"""
uvicorn project:app --reload --port 8001
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List

import fitz
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from umap import UMAP

MAX_CONCURRENT_READS = 16
SAMPLE_FRACTION: float | None | str = "auto"
HEAD_PAGES = 2

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
ENCODE_BATCH_SIZE = 128
SAMPLE_CHUNK_CHARS = 1_200
SAMPLE_POSITIONS = (0.0, 0.25, 0.5, 0.75)

REDUCER_BACKEND = "umap"  # "umap" | "pacmap" (requiere `pip install pacmap`)
UMAP_N_NEIGHBORS = 15  # más alto = estructura más global / menos ruido local
UMAP_MIN_DIST = 0.15  # qué tan apretados quedan puntos similares al dibujar
UMAP_METRIC = "cosine"
UMAP_N_EPOCHS: int | None = (
    None  # None = default de la librería; bajar acelera, baja calidad
)

# ── Clustering ──────────────────────────────────────────────────────────
# "raw_embeddings" (recomendado): HDBSCAN directo sobre los embeddings de
#     384 dims con distancia coseno. Es la opción de MEJOR calidad y la más
#     RÁPIDA de las tres, porque no necesita ninguna reducción extra: evita
#     la distorsión que mete cualquier proyección a 2D antes de clusterizar.
# "embedding_nd": reduce a CLUSTER_N_COMPONENTS dimensiones (con min_dist≈0,
#     como recomienda la guía oficial de UMAP+HDBSCAN) y clusteriza ahí.
#     Punto medio: menos distorsión que 2D, más barato que clusterizar en 384D
#     si tienes muchísimos documentos.
# "projection_2d": comportamiento original — clusteriza sobre las mismas
#     coords que se usan para dibujar. La más barata (una sola reducción),
#     pero la de peor calidad porque ya perdiste información al bajar a 2D.
CLUSTER_ON = "raw_embeddings"
CLUSTER_N_COMPONENTS = 8  # solo aplica si CLUSTER_ON == "embedding_nd"
HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_MIN_SAMPLES = 1

# ── Normalización de coordenadas para dibujar ───────────────────────────
# "none"    -> coords crudas de la proyección (rango arbitrario, cambia por corrida)
# "minmax"  -> cada eje escalado a [0,1] por separado (lo que hacía main.py).
#              OJO: si un eje tiene más rango que el otro, esto lo estira/achica
#              de forma distinta y deforma las proporciones del layout.
# "uniform" -> un solo factor de escala para ambos ejes (el mayor rango de los
#              dos), también a [0,1] pero conservando las proporciones reales
#              de la proyección. Generalmente la mejor opción para dibujar.
NORMALIZE_MODE = "uniform"

# ════════════════════════════════════════════════════════════════════════

REGEX_CLEAN = [
    (re.compile(r"https?://\S+|www\.\S+"), " "),
    (re.compile(r"doi[:/]\s?\S+", re.IGNORECASE), " "),
    (re.compile(r"\[\d+(,\s*\d+)*\]"), " "),
    (re.compile(r"\b\d+\b"), " "),
    (re.compile(r"[^\w\s]"), " "),
    (re.compile(r"\s+"), " "),
]

embed_model: SentenceTransformer | None = None
executor: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, executor
    print(f"Cargando modelo de embeddings ({EMBED_MODEL_NAME})...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embed_model.encode(["warmup"], show_progress_bar=False)
    executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 2)
    yield
    executor.shutdown(wait=False)
    print("Executor cerrado.")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Extracción de PDFs (idéntica a main.py: sin acumular bytes, lecturas
#    concurrentes con semáforo, muestreo adaptativo de páginas) ──────────
def _pick_sample_pages(total_pages: int, fraction: float) -> list[int]:
    if total_pages == 0:
        return []
    budget = max(1, round(total_pages * fraction))
    head_idx = list(range(min(HEAD_PAGES, total_pages)))
    if budget <= len(head_idx):
        return head_idx
    remaining = budget - len(head_idx)
    step = total_pages / remaining
    interleaved = {int(i * step) for i in range(remaining)}
    return sorted(set(head_idx) | interleaved)


def extract_text_from_bytes(
    file_bytes: bytes, sample_fraction: float | None = None
) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        total_pages = doc.page_count
        page_indices = (
            range(total_pages)
            if sample_fraction is None
            else _pick_sample_pages(total_pages, sample_fraction)
        )
        parts = [
            doc[i].get_text("text", flags=fitz.TEXTFLAGS_TEXT) for i in page_indices
        ]
        return " ".join(parts)


async def extract_all_texts(
    files,
    executor: ThreadPoolExecutor,
    max_concurrent_reads: int = 20,
    sample_fraction: float | None | str = "auto",
) -> tuple[list[str], list[str]]:
    n = len(files)
    fraction = sample_fraction
    if sample_fraction == "auto":
        if n <= 10:
            fraction = None
        elif n <= 50:
            fraction = 0.5
        elif n <= 100:
            fraction = 0.25
        else:
            fraction = 0.1

    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(max_concurrent_reads)
    filenames = [f.filename for f in files]

    async def _process(f) -> str:
        async with sem:
            content = await f.read()
            try:
                return await loop.run_in_executor(
                    executor, extract_text_from_bytes, content, fraction
                )
            except Exception as e:
                print(f"  [WARN] error procesando {f.filename}: {e}")
                return ""
            finally:
                del content

    results = await asyncio.gather(*[_process(f) for f in files])
    return filenames, list(results)


def clean_text(text: str) -> str:
    for regex, replacement in REGEX_CLEAN:
        text = regex.sub(replacement, text)
    return text.strip()


def sample_document(text: str) -> list[str]:
    total_needed = SAMPLE_CHUNK_CHARS * len(SAMPLE_POSITIONS)
    if len(text) <= total_needed:
        return [
            text[i : i + SAMPLE_CHUNK_CHARS]
            for i in range(0, len(text), SAMPLE_CHUNK_CHARS)
        ] or [text]
    chunks = []
    for p in SAMPLE_POSITIONS:
        start = int(p * (len(text) - SAMPLE_CHUNK_CHARS))
        chunks.append(text[start : start + SAMPLE_CHUNK_CHARS])
    return chunks


def compute_doc_embeddings(texts: list[str]) -> np.ndarray:
    assert embed_model is not None
    flat_chunks: list[str] = []
    spans: list[tuple[int, int]] = []
    for text in texts:
        chunks = sample_document(text)
        spans.append((len(flat_chunks), len(flat_chunks) + len(chunks)))
        flat_chunks.extend(chunks)

    chunk_embs = embed_model.encode(
        flat_chunks,
        normalize_embeddings=True,
        batch_size=ENCODE_BATCH_SIZE,
        show_progress_bar=False,
    )
    chunk_embs = np.asarray(chunk_embs, dtype="float32")

    doc_embs = np.vstack([chunk_embs[a:b].mean(axis=0) for a, b in spans]).astype(
        "float32"
    )
    return normalize(doc_embs, norm="l2")


# ── Reducción dimensional ────────────────────────────────────────────────
def _run_reducer(
    doc_embs: np.ndarray, n_components: int, n_neighbors: int, min_dist: float
) -> np.ndarray:
    if REDUCER_BACKEND == "pacmap":
        try:
            import pacmap

            reducer = pacmap.PaCMAP(
                n_components=n_components,
                n_neighbors=min(doc_embs.shape[0] - 1, n_neighbors),
            )
            return reducer.fit_transform(doc_embs)
        except ImportError:
            print("[WARN] pacmap no está instalado (pip install pacmap); usando UMAP.")

    reducer = UMAP(
        n_components=n_components,
        n_neighbors=min(doc_embs.shape[0] - 1, n_neighbors),
        min_dist=min_dist,
        metric=UMAP_METRIC,
        n_epochs=UMAP_N_EPOCHS,
        low_memory=True,
        n_jobs=-1,
    )
    return reducer.fit_transform(doc_embs)


def compute_2d(doc_embs: np.ndarray) -> np.ndarray | None:
    """Coords 2D crudas, sin normalizar, solo para dibujar."""
    n = doc_embs.shape[0]
    if n < 2:
        return None
    if n == 2:
        return np.array([[0.0, 0.5], [1.0, 0.5]], dtype="float32")
    if n <= 4:
        return PCA(n_components=2, random_state=42).fit_transform(doc_embs)
    return _run_reducer(
        doc_embs, n_components=2, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST
    )


def normalize_coords(coords: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return coords
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    if mode == "minmax":
        ranges = np.where(maxs - mins == 0, 1, maxs - mins)
        return (coords - mins) / ranges
    if mode == "uniform":
        span = np.where(maxs - mins == 0, 1, maxs - mins)
        scale = span.max()
        return (coords - mins) / scale
    raise ValueError(f"NORMALIZE_MODE inválido: {mode!r}")


# ── Clustering ────────────────────────────────────────────────────────────
def cluster_docs(
    doc_embs: np.ndarray, coords_2d: np.ndarray | None
) -> list[int | None]:
    n = doc_embs.shape[0]
    if n < 3:
        return [None] * n

    if CLUSTER_ON == "raw_embeddings":
        space = doc_embs.astype("float64")
        metric = "cosine"
    elif CLUSTER_ON == "embedding_nd":
        nd = _run_reducer(
            doc_embs,
            n_components=min(CLUSTER_N_COMPONENTS, n - 1),
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=0.0,  # apretado a propósito: queremos densidad fiel para HDBSCAN
        )
        space, metric = np.asarray(nd, dtype="float64"), "euclidean"
    elif CLUSTER_ON == "projection_2d":
        assert coords_2d is not None
        space, metric = coords_2d.astype("float64"), "euclidean"
    else:
        raise ValueError(f"CLUSTER_ON inválido: {CLUSTER_ON!r}")

    labels = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=metric,
        copy=False,
    ).fit_predict(space)
    return [int(label) for label in labels]


# ── Ruta ──────────────────────────────────────────────────────────────────
@app.post("/project-pdfs/")
async def project_pdfs(files: List[UploadFile] = File(...)):
    t0 = time.perf_counter()
    filenames, raw_texts = await extract_all_texts(
        files,
        executor,
        max_concurrent_reads=MAX_CONCURRENT_READS,
        sample_fraction=SAMPLE_FRACTION,
    )
    print(f"1. Extracción de texto: {time.perf_counter() - t0:.2f}s")

    clean_texts = [clean_text(t) for t in raw_texts]
    n = len(clean_texts)
    if n == 0:
        raise HTTPException(status_code=422, detail="Ningún PDF pudo procesarse.")

    t0 = time.perf_counter()
    doc_embs = compute_doc_embeddings(clean_texts)
    print(f"2. Doc embeddings ({n} docs): {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    coords_2d = compute_2d(doc_embs)
    clusters = cluster_docs(doc_embs, coords_2d)
    layout = (
        normalize_coords(coords_2d, NORMALIZE_MODE) if coords_2d is not None else None
    )
    print(f"3. Proyección + clustering: {time.perf_counter() - t0:.2f}s")

    locals_response = []
    if layout is not None:
        locals_response = [
            {
                "filename": filenames[i],
                "x": round(float(layout[i, 0]), 4),
                "y": round(float(layout[i, 1]), 4),
                "cluster": clusters[i],
            }
            for i in range(n)
        ]

    return {
        "locals": locals_response,
        "meta": {
            "n_files": n,
            "reducer": REDUCER_BACKEND,
            "cluster_on": CLUSTER_ON,
            "normalize_mode": NORMALIZE_MODE,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
