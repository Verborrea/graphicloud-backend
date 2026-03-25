# # import asyncio
# # import re
# # import time
# # from concurrent.futures import ThreadPoolExecutor
# # from typing import List

# # import fitz
# # import numpy as np
# # import spacy
# # from fastapi import FastAPI, File, HTTPException, UploadFile
# # from fastapi.middleware.cors import CORSMiddleware
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from umap import UMAP

# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Cargamos spaCy optimizado (solo lematización)
# # nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
# # executor = ThreadPoolExecutor()


# # def clean_paper_noise(text: str) -> str:
# #     """Limpieza quirúrgica de ruido en papers."""
# #     text = re.sub(r"https?://\S+|www\.\S+", "", text)
# #     text = re.sub(r"doi[:/]\s?\S+", "", text)
# #     text = re.sub(r"\[\d+(,\s*\d+)*\]", "", text)
# #     text = re.sub(r"\b\d+\b", "", text)
# #     return re.sub(r"\s+", " ", text).strip()


# # def extract_text(file_bytes: bytes) -> str:
# #     """Extracción rápida con PyMuPDF."""
# #     with fitz.open(stream=file_bytes, filetype="pdf") as doc:
# #         return " ".join([page.get_text() for page in doc])


# # @app.post("/convert-pdfs/")
# # async def convert_pdfs(
# #     files: List[UploadFile] = File(...),
# # ):
# #     # --- RESTRICCIÓN DE SEGURIDAD ---
# #     if len(files) < 3:
# #         raise HTTPException(
# #             status_code=400,
# #             detail="Se requieren al menos 3 documentos para la proyección UMAP.",
# #         )

# #     t_start = time.perf_counter()
# #     loop = asyncio.get_event_loop()

# #     # 1. Extracción paralela
# #     file_contents = [await f.read() for f in files]
# #     tasks = [
# #         loop.run_in_executor(executor, extract_text, content)
# #         for content in file_contents
# #     ]
# #     raw_texts = await asyncio.gather(*tasks)

# #     # 2. NLP (Lematización) en paralelo
# #     pre_cleaned = [clean_paper_noise(t) for t in raw_texts]
# #     processed_texts = []
# #     for doc in nlp.pipe(pre_cleaned, batch_size=5, n_process=-1):
# #         tokens = [
# #             t.lemma_.lower()
# #             for t in doc
# #             if t.is_alpha and not t.is_stop and len(t.text) > 2
# #         ]
# #         processed_texts.append(" ".join(tokens))

# #     # 3. Vectorización TF-IDF
# #     vectorizer = TfidfVectorizer(max_features=2100)
# #     matrix = vectorizer.fit_transform(processed_texts)
# #     feature_names = np.array(vectorizer.get_feature_names_out())

# #     # 4. Reducción UMAP (2D)
# #     # matrix_dense = matrix.toarray()
# #     n_neighbors = min(len(processed_texts) - 1, 15)

# #     reducer = UMAP(
# #         n_components=2, n_neighbors=n_neighbors, min_dist=0.1, metric="cosine"
# #     )
# #     embedding = reducer.fit_transform(matrix)

# #     # reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1)
# #     # embedding = reducer.fit_transform(matrix_dense)

# #     # 5. Normalización
# #     mins = embedding.min(axis=0)
# #     maxs = embedding.max(axis=0)
# #     # Evitar división por cero si todos los puntos caen en el mismo lugar
# #     ranges = np.where(maxs - mins == 0, 1, maxs - mins)
# #     embedding_norm = (embedding - mins) / ranges

# #     # 6. Construcción de respuesta (Ranking por documento + Coordenadas)
# #     results = []
# #     for i in range(len(files)):
# #         # Obtener pesos TF-IDF del documento i
# #         row = matrix.getrow(i).toarray()[0]
# #         top_indices = row.argsort()[-30:][::-1]

# #         results.append(
# #             {
# #                 "filename": files[i].filename,
# #                 "x": float(embedding_norm[i, 0]),
# #                 "y": float(embedding_norm[i, 1]),
# #                 "keywords": [
# #                     {"word": feature_names[idx], "score": round(float(row[idx]), 4)}
# #                     for idx in top_indices
# #                     if row[idx] > 0
# #                 ],
# #             }
# #         )

# #     t_end = time.perf_counter()
# #     print(f"✅ Análisis completo: {len(files)} docs en {t_end - t_start:.3f}s")

# #     return results

import asyncio
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List

import fitz
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

# ── Silenciar warnings de HuggingFace ─────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modelo cargado UNA SOLA VEZ al arrancar ────────────────────────────────────
print("⏳ Cargando modelo KeyBERT...")
kw_model = KeyBERT(model="all-MiniLM-L6-v2")
print("✅ Modelo listo.")

executor = ProcessPoolExecutor()

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
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
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
    """Filtra stopwords académicas y tokens muy cortos."""
    kw_lower = kw.lower().strip()
    if len(kw_lower) < 3:
        return False
    if kw_lower in ACADEMIC_STOPWORDS:
        return False
    # Filtra si alguna palabra del bigrama es stopword
    parts = kw_lower.split()
    if any(p in ACADEMIC_STOPWORDS for p in parts):
        return False
    return True


def deduplicate_keywords(keywords: list, top_n: int = 30) -> list:
    """
    Elimina variantes duplicadas (e.g. 'neural network' vs 'neural networks').
    Compara por root simple (primeros 6 chars de cada token).
    """
    seen_roots = {}
    result = []
    for item in sorted(keywords, key=lambda x: x["score"], reverse=True):
        word = item["word"].lower()
        root = " ".join(w[:6] for w in word.split())
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
# Endpoint principal
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

    # 2. Extracción paralela de texto
    tasks = [
        loop.run_in_executor(executor, extract_text_from_bytes, content)
        for content in file_contents
    ]
    raw_texts = await asyncio.gather(*tasks)

    # 3. Limpieza
    clean_texts = [clean_text(t) for t in raw_texts]

    # 4. TF-IDF — keywords LOCALES (por documento)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    local_keywords = []
    for i in range(len(files)):
        row = tfidf_matrix.getrow(i).toarray()[0]
        top_indices = row.argsort()[-80:][::-1]  # margen para filtrar
        candidates = [
            {"word": feature_names[idx], "score": round(float(row[idx]), 4)}
            for idx in top_indices
            if row[idx] > 0 and is_valid_keyword(feature_names[idx])
        ]
        local_keywords.append(deduplicate_keywords(candidates, top_n=30))

    # 5. KeyBERT — keywords GLOBALES (todo el corpus)
    corpus_text = " ".join(clean_texts)
    global_raw = kw_model.extract_keywords(
        corpus_text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        use_mmr=True,
        diversity=0.5,
        top_n=80,
    )
    global_candidates = [
        {"word": kw, "score": round(float(score), 4)}
        for kw, score in global_raw
        if is_valid_keyword(kw)
    ]
    global_keywords = deduplicate_keywords(global_candidates, top_n=30)

    # 6. UMAP — proyección 2D
    n_neighbors = min(len(files) - 1, 15)
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embedding = reducer.fit_transform(tfidf_matrix)

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
            "keywords": local_keywords[i],
        }
        for i in range(len(files))
    ]

    t_end = time.perf_counter()
    print(f"✅ {len(files)} docs en {t_end - t_start:.2f}s")

    return {"global": global_keywords, "locals": locals_response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
