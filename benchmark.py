"""
Corre esto en tu Mac para medir exactamente dónde se va el tiempo
en compute_doc_embeddings. Ajusta n_docs si quieres simular otro volumen.
"""

import time

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize as sk_normalize

SAMPLE_CHUNK_CHARS = 1_200
SAMPLE_POSITIONS = (0.0, 0.25, 0.5, 0.75)
N_DOCS = 200  # simula tu caso real

print("Cargando modelo...")
t0 = time.perf_counter()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print(
    f"  Carga del modelo: {time.perf_counter() - t0:.2f}s  (solo ocurre 1 vez si es global)"
)

# --- Simular textos realistas (ajusta el largo si tus PDFs son más grandes/chicos) ---
words = (
    "the quick brown fox jumps over lazy dog research methodology "
    "results discussion conclusion data analysis significant model "
    "performance evaluation framework approach technique"
).split()
rng = np.random.default_rng(0)


def fake_doc(n_words=3000):
    return " ".join(rng.choice(words, n_words))


texts = [fake_doc() for _ in range(N_DOCS)]
print(f"\n{N_DOCS} documentos simulados de ~3000 palabras cada uno")


# --- Paso 1: sample_document (chunking) ---
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


t0 = time.perf_counter()
flat_chunks = []
spans = []
for text in texts:
    chunks = sample_document(text)
    spans.append((len(flat_chunks), len(flat_chunks) + len(chunks)))
    flat_chunks.extend(chunks)
t_chunk = time.perf_counter() - t0
print(f"\n1. Chunking (sample_document x{N_DOCS}): {t_chunk:.3f}s")
print(f"   Total chunks generados: {len(flat_chunks)}")

# --- Paso 2: encode ---
for batch_size in [32, 48, 64, 80, 96, 128]:
    t0 = time.perf_counter()
    embs = embed_model.encode(
        flat_chunks,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    t_encode = time.perf_counter() - t0
    print(f"2. Encode batch_size={batch_size}: {t_encode:.3f}s")

# --- Paso 3: promedio por doc + normalize ---
chunk_embs = np.asarray(embs, dtype="float32")
t0 = time.perf_counter()
doc_embs = np.vstack([chunk_embs[a:b].mean(axis=0) for a, b in spans]).astype("float32")
doc_embs = sk_normalize(doc_embs, norm="l2")
t_agg = time.perf_counter() - t0
print(f"3. Agregación (mean + normalize): {t_agg:.4f}s")

print(
    f"\nTOTAL compute_doc_embeddings (con batch_size=128): "
    f"{t_chunk + t_agg:.2f}s + tiempo de encode arriba"
)
