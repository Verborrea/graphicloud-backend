import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List

import fitz
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keybert import KeyBERT
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from turbovec import TurboQuantIndex
from umap import UMAP

from stopwords import STOPWORDS

load_dotenv()

REGEX_CLEAN = [
    (re.compile(r"https?://\S+|www\.\S+"), " "),
    (re.compile(r"doi[:/]\s?\S+", re.IGNORECASE), " "),
    (re.compile(r"\[\d+(,\s*\d+)*\]"), " "),
    (re.compile(r"\b\d+\b"), " "),
    (re.compile(r"[^\w\s]"), " "),
    (re.compile(r"\s+"), " "),
]

ICONS_ENRICHED = os.getenv("ICONS_ENRICHED_PATH", "merged.json")
ICONS_SVG_DIR = os.getenv("ICONS_SVG_DIR", "icons")
FAISS_THRESHOLD = float(os.getenv("FAISS_THRESHOLD", "0.1"))
KEYBERT_MAX_CHARS = 4_000

# ── Modelos y Estructuras Globales ───────────────────────────────────────────
kw_model: KeyBERT | None = None
embed_model: SentenceTransformer | None = None
executor: ThreadPoolExecutor | None = None

# Reemplazo de FAISS por TurboVec
turbovec_index: TurboQuantIndex | None = None
local_icon_names: list[str] = []
local_icon_texts: list[str] = []


# ── Modelos de Datos ──────────────────────────────────────────────────────────
class KeywordItem(BaseModel):
    word: str
    score: float


class IconRequest(BaseModel):
    keywords: list[KeywordItem]


class BatchIconRequest(BaseModel):
    groups: list[list[KeywordItem]]


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global kw_model, embed_model, executor
    global turbovec_index, local_icon_names, local_icon_texts

    print("Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    kw_model = KeyBERT(model=embed_model)
    kw_model.extract_keywords("warmup", top_n=1)

    cpu_count = os.cpu_count() or 2
    executor = ThreadPoolExecutor(max_workers=cpu_count)

    print(f"Construyendo índice TurboVec con íconos locales ({ICONS_ENRICHED})...")
    try:
        with open(ICONS_ENRICHED, "r", encoding="utf-8") as f:
            enriched_data: list[dict] = json.load(f)

        local_icon_names = [item["name"] for item in enriched_data]
        local_icon_texts = [item["text"] for item in enriched_data]

        vectors = embed_model.encode(
            local_icon_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.array(vectors, dtype="float32")

        dim = vectors.shape[1]

        turbovec_index = TurboQuantIndex(dim=dim, bit_width=4)
        turbovec_index.add(vectors)

        print(f"TurboVec listo: {len(local_icon_names)} íconos locales indexados.")

    except FileNotFoundError:
        print(f"No se encontró {ICONS_ENRICHED} — Búsqueda local deshabilitada.")

    yield

    executor.shutdown(wait=False)
    print("Executor cerrado.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers Optimizados ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    for regex, replacement in REGEX_CLEAN:
        text = regex.sub(replacement, text)
    return text.strip()


def is_valid_keyword(kw: str) -> bool:
    kw_lower = kw.lower().strip()
    if len(kw_lower) < 3 or kw_lower in STOPWORDS:
        return False
    return not any(p in STOPWORDS for p in kw_lower.split())


def deduplicate_keywords(keywords: list, top_n: int = 30) -> list:
    seen_roots: dict[str, bool] = {}
    result = []
    for item in sorted(keywords, key=lambda x: x["score"], reverse=True):
        root = " ".join(w[:6] for w in item["word"].lower().split())
        if root not in seen_roots:
            seen_roots[root] = True
            result.append(item)
        if len(result) == top_n:
            break
    return result


def extract_text_from_bytes(file_bytes: bytes) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return " ".join(page.get_text() for page in doc)


def cosine_similarity_pair(mat) -> float:
    normed = normalize(mat, norm="l2")
    dense = normed.toarray() if hasattr(normed, "toarray") else normed
    return round(float((dense[0] * dense[1]).sum()), 4)


def extract_keybert_for_doc(text: str, top_n: int = 40) -> list[dict]:
    truncated = text[:KEYBERT_MAX_CHARS]
    raw = kw_model.extract_keywords(
        truncated,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_mmr=True,
        diversity=0.4,
        top_n=top_n,
    )
    return [
        {"word": kw, "score": round(float(score), 4)}
        for kw, score in raw
        if is_valid_keyword(kw)
    ]


def merge_keybert_results(per_doc: list[list[dict]], top_n: int = 30) -> list[dict]:
    merged: dict[str, float] = {}
    for doc_kws in per_doc:
        for item in doc_kws:
            word = item["word"].lower()
            merged[word] = max(merged.get(word, 0.0), item["score"])

    combined = [{"word": w, "score": s} for w, s in merged.items()]
    return deduplicate_keywords(
        sorted(combined, key=lambda x: x["score"], reverse=True),
        top_n=top_n,
    )


def _build_query_text(keywords: list[KeywordItem]) -> str:
    top_kws = sorted(keywords, key=lambda x: x.score, reverse=True)[:10]
    parts = []
    for kw in top_kws:
        reps = max(1, round(kw.score * 4))
        parts.extend([kw.word] * reps)
    return " ".join(parts)


def search_local_icon_turbovec(
    keywords: list[KeywordItem],
) -> tuple[str | None, float, list[dict]]:
    if turbovec_index is None or not local_icon_names:
        return None, 0.0, []

    query_text = _build_query_text(keywords)
    query_vec = embed_model.encode([query_text], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype="float32")  # TurboVec espera float32

    k = min(3, len(local_icon_names))

    # Búsqueda en TurboVec
    scores, indices = turbovec_index.search(query_vec, k=k)

    # TurboVec devuelve estructuras bidimensionales [queries, resultados]
    top3 = [
        {"name": local_icon_names[idx], "score": round(float(s), 4)}
        for s, idx in zip(scores[0], indices[0])
        if idx >= 0 and idx < len(local_icon_names)
    ]

    best_name = top3[0]["name"] if top3 else None
    best_score = top3[0]["score"] if top3 else 0.0

    if best_score < FAISS_THRESHOLD:
        return None, best_score, top3

    return best_name, best_score, top3


def load_local_svg(icon_name: str) -> str | None:
    path = os.path.join(ICONS_SVG_DIR, f"{icon_name}.svg")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None


# ── Rutas API ──────────────────────────────────────────────────────────────────


@app.post("/convert-pdfs/")
async def convert_pdfs(files: List[UploadFile] = File(...)):
    loop = asyncio.get_running_loop()

    # Carga directa en memoria (eficiente para un único usuario local)
    file_contents = [await f.read() for f in files]

    async def safe_extract(content: bytes, filename: str) -> str:
        try:
            return await loop.run_in_executor(
                executor, extract_text_from_bytes, content
            )
        except Exception as e:
            raise HTTPException(
                status_code=422, detail=f"Error procesando '{filename}': {e}"
            )

    t0 = time.perf_counter()
    raw_texts = await asyncio.gather(
        *[safe_extract(c, files[i].filename) for i, c in enumerate(file_contents)]
    )
    print(f"PDF text extraction: {time.perf_counter() - t0:.2f}s")

    clean_texts = [clean_text(t) for t in raw_texts]
    for i, text in enumerate(clean_texts):
        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail=f"'{files[i].filename}' no tiene texto extraíble.",
            )

    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    local_tfidf_keywords = []
    dense_tfidf = tfidf_matrix.toarray()

    for i in range(len(files)):
        row = dense_tfidf[i]
        top_idx = row.argsort()[-80:][::-1]
        candidates = [
            {"word": feature_names[j], "score": round(float(row[j]), 4)}
            for j in top_idx
            if row[j] > 0 and is_valid_keyword(feature_names[j])
        ]
        local_tfidf_keywords.append(deduplicate_keywords(candidates, top_n=30))
    print(f"TF-IDF: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    per_doc_keybert = []
    for text in clean_texts:
        kws = extract_keybert_for_doc(text)
        per_doc_keybert.append(kws)

    global_keywords = merge_keybert_results(per_doc_keybert, top_n=30)
    print(f"  KeyBERT ({len(files)} docs procesados): {time.perf_counter() - t0:.2f}s")

    n_samples = len(files)

    if n_samples == 1:
        return {"global": global_keywords, "locals": [], "similarity": None}

    elif n_samples == 2:
        similarity = cosine_similarity_pair(tfidf_matrix)
        locals_response = [
            {
                "filename": files[i].filename,
                "x": float(i),
                "y": float(i),
                "keywords": local_tfidf_keywords[i],
            }
            for i in range(2)
        ]
        return {
            "global": global_keywords,
            "locals": locals_response,
            "similarity": similarity,
        }

    else:
        t0 = time.perf_counter()
        tfidf_normalized = normalize(tfidf_matrix, norm="l2")

        if n_samples == 3:
            embedding = np.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.2]])
        else:
            # ── CAMBIO AQUÍ: Forzar a UMAP a ser estrictamente single-thread en C ──
            reducer = UMAP(
                n_components=2,
                n_neighbors=min(n_samples - 1, 15),
                min_dist=0.15,
                metric="euclidean",
                random_state=42,
                low_memory=True,
                n_jobs=1,  # <--- Crucial para que numba no rompa la memoria en Mac
            )
            embedding = reducer.fit_transform(tfidf_normalized)

        print(f"UMAP: {time.perf_counter() - t0:.2f}s")

        mins = embedding.min(axis=0)
        maxs = embedding.max(axis=0)
        ranges = np.where(maxs - mins == 0, 1, maxs - mins)
        embedding_norm = (embedding - mins) / ranges

        locals_response = [
            {
                "filename": files[i].filename,
                "x": round(float(embedding_norm[i, 0]), 4),
                "y": round(float(embedding_norm[i, 1]), 4),
                "keywords": local_tfidf_keywords[i],
            }
            for i in range(len(files))
        ]
        return {
            "global": global_keywords,
            "locals": locals_response,
            "similarity": None,
        }


@app.post("/select-icon/")
async def select_icon(body: IconRequest):
    if not body.keywords:
        raise HTTPException(status_code=400, detail="Se requiere al menos una keyword.")

    if turbovec_index is None:
        raise HTTPException(status_code=503, detail="Índice TurboVec no inicializado.")

    best_name, best_score, top3 = search_local_icon_turbovec(body.keywords)

    if best_name is None:
        raise HTTPException(
            status_code=404,
            detail={
                "message": "Ningún ícono local supera el umbral de similitud.",
                "threshold": FAISS_THRESHOLD,
                "best_score": best_score,
                "candidates": top3,
            },
        )

    svg = load_local_svg(best_name)
    if svg is None:
        raise HTTPException(
            status_code=500, detail=f"Ícono '{best_name}' no encontrado en disco."
        )

    avg_score = round(sum(k.score for k in body.keywords) / len(body.keywords), 4)

    return {
        "icon": f"ph:{best_name}",
        "svg": svg,
        "score": best_score,
        "avg_kw_score": avg_score,
        "source": "local",
        "candidates": [{"icon": f"ph:{c['name']}", "score": c["score"]} for c in top3],
    }


@app.post("/select-icons-batch/")
async def select_icons_batch(body: BatchIconRequest):
    if not body.groups:
        raise HTTPException(
            status_code=400, detail="No se enviaron grupos de palabras clave."
        )
    if turbovec_index is None:
        raise HTTPException(status_code=503, detail="Índice TurboVec no disponible.")

    loop = asyncio.get_running_loop()

    async def process_group(keywords: list[KeywordItem]) -> dict:
        if not keywords:
            return {"icon": None, "svg": None, "error": "Grupo vacío"}

        best_name, best_score, _ = await loop.run_in_executor(
            executor, search_local_icon_turbovec, keywords
        )
        if best_name is None:
            return {"icon": None, "svg": None, "error": "No superó el umbral"}

        svg = await loop.run_in_executor(executor, load_local_svg, best_name)
        if svg is None:
            return {
                "icon": None,
                "svg": None,
                "error": f"SVG '{best_name}' no encontrado",
            }

        return {"icon": f"ph:{best_name}", "svg": svg, "score": best_score}

    results = await asyncio.gather(*[process_group(kws) for kws in body.groups])
    return {"icons": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
