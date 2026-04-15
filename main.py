import asyncio
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List

import fitz
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keybert import KeyBERT
from openai import AsyncOpenAI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

# ── Env ────────────────────────────────────────────────────────────────────────
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

ICON_INDEX_PATH = os.getenv("ICON_INDEX_PATH", "icons_index.json")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Máx de caracteres que le pasamos a KeyBERT por documento.
# all-MiniLM-L6-v2 tiene ventana de 256 tokens (~1200 chars).
# Pasar más no mejora la calidad — solo enlentece.
KEYBERT_MAX_CHARS = 4_000


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global kw_model, executor, icon_index, openai_client

    print("⏳ Cargando modelo KeyBERT...")
    kw_model = KeyBERT(model="all-MiniLM-L6-v2")
    # Pre-warm: evita que el primer request pague el JIT del modelo
    kw_model.extract_keywords("warmup", top_n=1)
    print("✅ Modelo listo.")

    # Un hilo por CPU lógica — KeyBERT suelta el GIL durante la inferencia
    cpu_count = os.cpu_count() or 2
    executor = ThreadPoolExecutor(max_workers=cpu_count)

    print("⏳ Cargando índice de íconos...")
    try:
        with open(ICON_INDEX_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            icon_index.update(raw)
        total = sum(len(v) for v in icon_index.values())
        print(f"✅ Índice cargado: {len(icon_index)} colecciones, {total} íconos.")
    except FileNotFoundError:
        print(f"⚠️  No se encontró {ICON_INDEX_PATH}.")

    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    yield

    executor.shutdown(wait=False)
    print("🛑 Executor cerrado.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

kw_model: KeyBERT | None = None
executor: ThreadPoolExecutor | None = None
icon_index: dict[str, list[str]] = {}
openai_client: AsyncOpenAI | None = None

ACADEMIC_STOPWORDS = {
    "et",
    "al",
    "et al",
    "ibid",
    "op cit",
    "viz",
    "ie",
    "eg",
    "figure",
    "fig",
    "table",
    "section",
    "appendix",
    "chapter",
    "abstract",
    "introduction",
    "conclusion",
    "conclusions",
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "discussion",
    "results",
    "methods",
    "methodology",
    "related work",
    "background",
    "overview",
    "summary",
    "supplementary",
    "show",
    "shows",
    "shown",
    "propose",
    "proposed",
    "present",
    "presents",
    "use",
    "used",
    "using",
    "based",
    "paper",
    "work",
    "study",
    "approach",
    "method",
    "result",
    "experiment",
    "experiments",
    "evaluation",
    "performance",
    "analysis",
    "compare",
    "compared",
    "comparison",
    "dataset",
    "data",
    "task",
    "tasks",
    "model",
    "models",
    "training",
    "testing",
    "trained",
    "test",
    "train",
    "number",
    "set",
    "state",
    "also",
    "however",
    "therefore",
    "thus",
    "hence",
    "arxiv",
    "preprint",
    "journal",
    "conference",
    "proceedings",
    "vol",
    "volume",
    "pp",
    "page",
    "pages",
    "doi",
    "http",
    "https",
    "com",
    "org",
    "www",
    "university",
    "institute",
    "department",
    "email",
    "corresponding",
    "author",
    "authors",
    "ii",
    "iii",
    "iv",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers PDF
# ──────────────────────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"doi[:/]\s?\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\d+(,\s*\d+)*\]", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_keyword(kw: str) -> bool:
    kw_lower = kw.lower().strip()
    if len(kw_lower) < 3:
        return False
    if kw_lower in ACADEMIC_STOPWORDS:
        return False
    return not any(p in ACADEMIC_STOPWORDS for p in kw_lower.split())


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


# ──────────────────────────────────────────────────────────────────────────────
# KeyBERT por documento (corre en hilo del executor)
# ──────────────────────────────────────────────────────────────────────────────


def extract_keybert_for_doc(text: str, top_n: int = 40) -> list[dict]:
    """
    Extrae keywords de UN documento con KeyBERT.
    Se llama desde el executor para no bloquear el event loop.
    Trunca el texto a KEYBERT_MAX_CHARS para que sea rápido.
    """
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
    """
    Fusiona keywords de todos los documentos.
    Para cada palabra única, toma el score máximo entre documentos.
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers íconos
# ──────────────────────────────────────────────────────────────────────────────


def find_best_match(candidate: str, names: list[str]) -> str | None:
    c = candidate.lower().strip()
    if c in names:
        return c
    for name in names:
        if c in name or name in c:
            return name
    c_tokens = set(re.split(r"[-_\s]", c))
    best, best_score = None, 0
    for name in names:
        shared = len(c_tokens & set(re.split(r"[-_\s]", name)))
        if shared > best_score:
            best_score, best = shared, name
    return best if best_score > 0 else None


async def fetch_svg_from_iconify(prefix: str, icon: str) -> str | None:
    url = f"https://api.iconify.design/{prefix}/{icon}.svg"
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            r = await client.get(url)
            if r.status_code == 200 and "<svg" in r.text:
                return r.text
        except httpx.RequestError:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────


class KeywordItem(BaseModel):
    word: str
    score: float


class IconRequest(BaseModel):
    keywords: list[KeywordItem]


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint: convertir PDFs
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/convert-pdfs/")
async def convert_pdfs(files: List[UploadFile] = File(...)):
    if len(files) < 2:
        raise HTTPException(
            status_code=400,
            detail="Se necesitan al menos 2 documentos para la proyección UMAP.",
        )

    t_start = time.perf_counter()
    loop = asyncio.get_event_loop()

    # 1. Leer archivos
    file_contents = [await f.read() for f in files]

    # 2. Extracción de texto — paralela en hilos
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
    print(f"  PDF text extraction: {time.perf_counter() - t0:.2f}s")

    # 3. Limpieza
    clean_texts = [clean_text(t) for t in raw_texts]
    for i, text in enumerate(clean_texts):
        if not text.strip():
            raise HTTPException(
                status_code=422,
                detail=f"'{files[i].filename}' no tiene texto extraíble (¿PDF escaneado sin OCR?).",
            )

    # 4. TF-IDF — keywords LOCALES (rápido, en el event loop está bien)
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    local_tfidf_keywords = []
    for i in range(len(files)):
        row = tfidf_matrix.getrow(i).toarray()[0]
        top_idx = row.argsort()[-80:][::-1]
        candidates = [
            {"word": feature_names[j], "score": round(float(row[j]), 4)}
            for j in top_idx
            if row[j] > 0 and is_valid_keyword(feature_names[j])
        ]
        local_tfidf_keywords.append(deduplicate_keywords(candidates, top_n=30))
    print(f"  TF-IDF: {time.perf_counter() - t0:.2f}s")

    # 5. KeyBERT — UNO POR DOCUMENTO, en paralelo en el executor
    #    Lanzamos todos a la vez y esperamos juntos → se solapan en CPU
    t0 = time.perf_counter()
    keybert_tasks = [
        loop.run_in_executor(executor, extract_keybert_for_doc, text)
        for text in clean_texts
    ]
    per_doc_keybert: list[list[dict]] = await asyncio.gather(*keybert_tasks)
    global_keywords = merge_keybert_results(per_doc_keybert, top_n=30)
    print(f"  KeyBERT ({len(files)} docs en paralelo): {time.perf_counter() - t0:.2f}s")

    # 6. UMAP — euclidean es ~3x más rápido que cosine en CPU
    t0 = time.perf_counter()
    from sklearn.preprocessing import normalize

    tfidf_normalized = normalize(tfidf_matrix, norm="l2")

    n_samples = len(files)
    # UMAP necesita n_neighbors < n_samples, y el layout espectral necesita
    # al menos n_components+1 muestras. Con ≤3 docs usamos TruncatedSVD como fallback.
    if n_samples <= 3:
        from sklearn.decomposition import TruncatedSVD

        embedding = TruncatedSVD(n_components=2, random_state=42).fit_transform(
            tfidf_normalized
        )
    else:
        reducer = UMAP(
            n_components=2,
            n_neighbors=min(n_samples - 1, 15),
            min_dist=0.1,
            metric="euclidean",
            random_state=42,
            low_memory=True,
        )
        embedding = reducer.fit_transform(tfidf_normalized)
    print(f"  UMAP: {time.perf_counter() - t0:.2f}s")

    # 7. Normalización 0–1
    mins = embedding.min(axis=0)
    maxs = embedding.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1, maxs - mins)
    embedding_norm = (embedding - mins) / ranges

    # 8. Respuesta
    locals_response = [
        {
            "filename": files[i].filename,
            "x": round(float(embedding_norm[i, 0]), 4),
            "y": round(float(embedding_norm[i, 1]), 4),
            "keywords": local_tfidf_keywords[i],
        }
        for i in range(len(files))
    ]

    print(f"✅ Total {len(files)} docs: {time.perf_counter() - t_start:.2f}s")
    return {"global": global_keywords, "locals": locals_response}


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint: selección de ícono por keywords
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/select-icon/")
async def select_icon(body: IconRequest):
    if not icon_index:
        raise HTTPException(status_code=503, detail="Índice de íconos no cargado.")
    if not body.keywords:
        raise HTTPException(status_code=400, detail="Se requiere al menos una keyword.")

    sorted_kws = sorted(body.keywords, key=lambda x: x.score, reverse=True)
    kw_lines = "\n".join(f'  - "{k.word}" (weight: {k.score:.2f})' for k in sorted_kws)
    avg_score = round(sum(k.score for k in body.keywords) / len(body.keywords), 4)
    available_prefixes = list(icon_index.keys())

    prompt = f"""You are an icon selection expert. Given weighted concepts, choose the best icon collection and suggest 5 icon name candidates.

Concepts (higher weight = more important):
{kw_lines}

Available icon collections and their styles:
- streamline: outline/line icons, broad coverage
- tabler: clean outline icons, technical/UI focused
- mdi: Material Design, very broad coverage
- solar: bold/duotone modern icons
- lucide: minimal outline, developer-friendly
- heroicons: simple outline/solid, UI-focused
- carbon: IBM Carbon, enterprise/data focused
- fluent: Microsoft Fluent, modern UI
- material-symbols: Google Material, broad coverage
- openmoji: colorful emoji-style icons

Rules:
- Icon names use kebab-case (e.g. "bar-chart", "user-circle", "file-search")
- Suggest names likely to exist in the chosen collection
- Prioritize concepts with higher weight
- Return ONLY valid JSON, no explanation, no markdown

Output format:
{{"prefix": "<chosen_collection>", "candidates": ["name1", "name2", "name3", "name4", "name5"]}}
"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=150,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error llamando a OpenAI: {e}")

    raw = response.choices[0].message.content.strip()

    try:
        gpt_result: dict = json.loads(raw)
        chosen_prefix: str = gpt_result.get("prefix", "").strip()
        suggestions: list[str] = gpt_result.get("candidates", [])
        if not chosen_prefix or not suggestions:
            raise ValueError("Faltan campos")
    except Exception:
        raise HTTPException(
            status_code=502, detail=f"OpenAI devolvió formato inesperado: {raw}"
        )

    if chosen_prefix not in icon_index:
        chosen_prefix = available_prefixes[0]

    collection_icons = icon_index[chosen_prefix]
    matched, seen = [], set()

    for candidate in suggestions:
        match = find_best_match(candidate, collection_icons)
        if match and match not in seen:
            seen.add(match)
            matched.append(match)
        if len(matched) == 3:
            break

    if not matched:
        for prefix, icons in icon_index.items():
            if prefix == chosen_prefix:
                continue
            for candidate in suggestions:
                match = find_best_match(candidate, icons)
                if match and match not in seen:
                    seen.add(match)
                    matched.append(match)
                    chosen_prefix = prefix
                    break
            if matched:
                break

    if not matched:
        raise HTTPException(
            status_code=404, detail=f"No se encontraron íconos para: {suggestions}"
        )

    best = matched[0]
    svg = await fetch_svg_from_iconify(chosen_prefix, best)
    if svg is None:
        for fallback in matched[1:]:
            svg = await fetch_svg_from_iconify(chosen_prefix, fallback)
            if svg:
                best = fallback
                break

    if svg is None:
        raise HTTPException(
            status_code=502,
            detail=f"No se pudo obtener SVG para '{chosen_prefix}': {matched}",
        )

    return {
        "icon": f"{chosen_prefix}:{best}",
        "svg": svg,
        "score": avg_score,
        "candidates": [f"{chosen_prefix}:{c}" for c in matched],
        "gpt_suggestions": suggestions,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
