"""
Servidor nuevo, separado de main.py, con el pipeline de proyección 2D +
clustering (sin keywords) y un endpoint aparte para keywords por documento.

uvicorn main_v2:app --reload --port 8000
"""

import asyncio
import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List

import fitz
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from umap import UMAP

from stopwords import STOPWORDS

load_dotenv()

# ── Configuración ─────────────────────────────────────────────────────────────
ICON_PACKS_DIR = "icon-packs"
ICON_PACK = "material"
SCORE_THRESHOLD = 0.1
PHRASE_MODE = "words_bigrams"
EXACT_MATCH_BONUS = 1.0
NAME_MATCH_BONUS = 2.0
NAME_PARTIAL_BONUS = 0.6
HEAD_PAGES = 2
CACHE_VERSION = 2
SAMPLE_CHUNK_CHARS = 1_200
SAMPLE_POSITIONS = (0.0, 0.25, 0.5, 0.75)
MAX_CANDIDATES_PER_DOC = 100  # candidatos para keywords semánticas (MMR)
MMR_DIVERSITY = 0.4
QUERY_CACHE_SIZE = 1_024
TOP_K_TFIDF_LOCAL = 80

REGEX_CLEAN = [
    (re.compile(r"https?://\S+|www\.\S+"), " "),
    (re.compile(r"doi[:/]\s?\S+", re.IGNORECASE), " "),
    (re.compile(r"\[\d+(,\s*\d+)*\]"), " "),
    (re.compile(r"\b\d+\b"), " "),
    (re.compile(r"[^\w\s]"), " "),
    (re.compile(r"\s+"), " "),
]

_WS_RE = re.compile(r"\s+")

ALL_STOPWORDS = set(STOPWORDS)

# ── Estado global ─────────────────────────────────────────────────────────────
embed_model: SentenceTransformer | None = None
executor: ThreadPoolExecutor | None = None
active_pack: "IconPack | None" = None
_query_cache: "OrderedDict[tuple, dict]" = OrderedDict()


# ── Modelos de datos ──────────────────────────────────────────────────────────
class KeywordItem(BaseModel):
    word: str
    score: float


class IconRequest(BaseModel):
    keywords: list[KeywordItem]


class BatchIconRequest(BaseModel):
    groups: list[list[KeywordItem]]


# ── Helper de normalización de frases ─────────────────────────────────────────
def _norm_phrase(s: str) -> str:
    return _WS_RE.sub(" ", s.strip().lower())


# ── Packs de íconos (idéntico a main.py, mismas rutas) ────────────────────────
class IconPack:
    """Pack de íconos autocontenido en icon-packs/<nombre>/.

    Estructura esperada:
        icon-packs/phosphor/
            pack.json          -> {"prefix": "ph", "metadata": "merged.json",
                                   "svg_dir": "icons"}
            merged.json        -> [{"name": ..., "text": ...}, ...]
            icons/*.svg
            .cache/            -> generado automáticamente (vectores + hash)

    Representación de búsqueda, cada ícono se descompone en varias FRASES y
    se embede cada frase por separado. La búsqueda toma el máximo de
    similitud sobre las frases del ícono ("late interaction" / max-pooling),
    y suma un bonus si hay coincidencia léxica exacta.

    Los embeddings por frase se calculan una vez y se persisten a disco; solo
    se recalculan si cambia el hash del metadata o la CACHE_VERSION.
    """

    def __init__(self, name: str, packs_dir: str = ICON_PACKS_DIR):
        self.name = name
        self.root = os.path.join(packs_dir, name)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Pack '{name}' no existe en {packs_dir}/")

        cfg_path = os.path.join(self.root, "pack.json")
        cfg: dict = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        self.prefix: str = cfg.get("prefix", name[:2])
        self.metadata_path = os.path.join(self.root, cfg.get("metadata", "merged.json"))
        self.svg_dir = os.path.join(self.root, cfg.get("svg_dir", "icons"))
        self.cache_dir = os.path.join(self.root, ".cache")

        self.names: list[str] = []
        self.name_phrases: list[str] = []  # name normalizado (sin _ ni -)

        # Frases por ícono + estructuras derivadas para búsqueda híbrida.
        self.icon_phrases: list[list[str]] = []
        self.icon_phrase_sets: list[set[str]] = []
        self.phrase_to_icons: dict[
            str, list[int]
        ] = {}  # índice invertido léxico (tags)
        self.phrase_vectors: np.ndarray | None = None  # (n_frases_totales, dim)
        self.phrase_owner: np.ndarray | None = None  # (n_frases_totales,) -> ícono

        # Índices invertidos sobre el NAME (separados de los tags, para que el
        # bonus de nombre se pueda aplicar sin penalización y en O(1)).
        self.name_phrase_to_icon: dict[str, int] = {}  # name completo -> ícono único
        self.name_word_to_icons: dict[str, list[int]] = {}  # palabra del name -> íconos

        # Penalización por tamaño de bag de tags (no afecta al bonus de name).
        self.icon_tag_penalty: np.ndarray | None = None  # (n_icons,)

        self._svg_cache: dict[str, str | None] = {}

    # -- split del campo `text` en frases --------------------------------------
    def _split_phrases(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []

        if "," in text:
            # El pack ya marca las frases con comas: respetarlas SIEMPRE.
            raw = [_norm_phrase(p) for p in text.split(",")]
        elif PHRASE_MODE == "comma":
            raw = [_norm_phrase(text)]
        else:
            tokens = _norm_phrase(text).split()
            raw = list(tokens)
            if PHRASE_MODE == "words_bigrams":
                # Bigramas adyacentes: así "team members" existe como frase
                # aunque el texto venga como "...team team members user...".
                raw += [f"{a} {b}" for a, b in zip(tokens, tokens[1:])]

        seen: set[str] = set()
        out: list[str] = []
        for p in raw:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    # -- carga / cache de vectores por frase -----------------------------------
    def load(self, model: SentenceTransformer) -> None:
        with open(self.metadata_path, "rb") as f:
            raw = f.read()
        meta_hash = hashlib.md5(raw).hexdigest()
        enriched: list[dict] = json.loads(raw.decode("utf-8"))

        self.names = [item["name"] for item in enriched]
        self.name_phrases = [
            _norm_phrase(n.replace("_", " ").replace("-", " ")) for n in self.names
        ]

        # -- índices invertidos sobre el name (independientes de los tags) --
        self.name_phrase_to_icon = {}
        self.name_word_to_icons = {}
        for i, name_phrase in enumerate(self.name_phrases):
            self.name_phrase_to_icon.setdefault(name_phrase, i)
            for w in name_phrase.split():
                self.name_word_to_icons.setdefault(w, []).append(i)

        # -- frases por ícono: el name siempre entra como frase de búsqueda --
        self.icon_phrases = []
        for item, name_phrase in zip(enriched, self.name_phrases):
            phrases = self._split_phrases(item.get("text") or item["name"])
            if name_phrase not in phrases:
                phrases = [name_phrase] + phrases
            self.icon_phrases.append(phrases)
        self.icon_phrase_sets = [set(p) for p in self.icon_phrases]

        # -- penalización por tamaño de bag de tags --
        n_phrases_per_icon = np.array(
            [len(p) for p in self.icon_phrases], dtype="float32"
        )
        self.icon_tag_penalty = 1.0 / np.log2(n_phrases_per_icon + 2)

        # Aplanado de frases + índice invertido léxico (frase -> íconos que la tienen).
        self.phrase_to_icons = {}
        flat_phrases: list[str] = []
        owner: list[int] = []
        for i, phrases in enumerate(self.icon_phrases):
            for p in phrases:
                self.phrase_to_icons.setdefault(p, []).append(i)
                flat_phrases.append(p)
                owner.append(i)
        self.phrase_owner = np.asarray(owner, dtype="int32")
        n_phrases = len(flat_phrases)

        vec_path = os.path.join(self.cache_dir, "phrase_vectors.npy")
        meta_path = os.path.join(self.cache_dir, "meta.json")

        if os.path.exists(vec_path) and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (
                cached.get("hash") == meta_hash
                and cached.get("version") == CACHE_VERSION
                and cached.get("phrase_mode") == PHRASE_MODE
                and cached.get("n_phrases") == n_phrases
            ):
                self.phrase_vectors = np.load(vec_path)
                print(
                    f"Pack '{self.name}': {len(self.names)} íconos / "
                    f"{n_phrases} frases cargados desde cache (.npy)."
                )
                return

        # Cache inválido o inexistente -> encodear las frases y persistir.
        t0 = time.perf_counter()
        vectors = model.encode(
            flat_phrases,
            normalize_embeddings=True,
            batch_size=256,
            show_progress_bar=False,
        )
        self.phrase_vectors = np.asarray(vectors, dtype="float32")
        print(
            f"Pack '{self.name}': {len(self.names)} íconos / {n_phrases} frases "
            f"encodeadas en {time.perf_counter() - t0:.2f}s."
        )

        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(vec_path, self.phrase_vectors)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "hash": meta_hash,
                    "version": CACHE_VERSION,
                    "phrase_mode": PHRASE_MODE,
                    "n_phrases": n_phrases,
                    "names": self.names,
                },
                f,
            )

    # -- búsqueda híbrida: max sobre frases + bonus léxico jerárquico -----------
    def search_hybrid(
        self,
        query_matrix: np.ndarray,
        query_keywords: list[list[str]],
        k: int = 3,
    ):
        """query_matrix: (Q, dim) L2-normalizada (centroide semántico por grupo).
        query_keywords: Q listas de strings normalizados (para match léxico).
        Devuelve (scores, indices) con shape (Q, k), score = semántico + bonus.
        """
        assert self.phrase_vectors is not None and self.phrase_owner is not None
        assert self.icon_tag_penalty is not None
        Q = query_matrix.shape[0]
        n_icons = len(self.names)
        k = min(k, n_icons)

        sims = query_matrix @ self.phrase_vectors.T  # (Q, n_frases)
        icon_scores = np.full((Q, n_icons), -1.0, dtype="float32")
        for q in range(Q):
            np.maximum.at(icon_scores[q], self.phrase_owner, sims[q])

        for q in range(Q):
            for kw in query_keywords[q]:
                for ic in self.phrase_to_icons.get(kw, ()):
                    icon_scores[q, ic] += EXACT_MATCH_BONUS * self.icon_tag_penalty[ic]

                exact_icon = self.name_phrase_to_icon.get(kw)
                if exact_icon is not None:
                    icon_scores[q, exact_icon] += NAME_MATCH_BONUS

                for ic in self.name_word_to_icons.get(kw, ()):
                    icon_scores[q, ic] += NAME_PARTIAL_BONUS

        idx = np.argpartition(-icon_scores, kth=k - 1, axis=1)[:, :k]
        part = np.take_along_axis(icon_scores, idx, axis=1)
        order = np.argsort(-part, axis=1)
        idx_sorted = np.take_along_axis(idx, order, axis=1)
        scores_sorted = np.take_along_axis(part, order, axis=1)
        return scores_sorted, idx_sorted

    # -- SVGs (memoizados en RAM) ----------------------------------------------
    def get_svg(self, icon_name: str) -> str | None:
        if icon_name in self._svg_cache:
            return self._svg_cache[icon_name]
        path = os.path.join(self.svg_dir, f"{icon_name}.svg")
        try:
            with open(path, "r", encoding="utf-8") as f:
                svg = f.read()
        except FileNotFoundError:
            svg = None
        self._svg_cache[icon_name] = svg
        return svg

    def icon_id(self, icon_name: str) -> str:
        return f"{self.prefix}:{icon_name}"


def list_available_packs() -> list[str]:
    if not os.path.isdir(ICON_PACKS_DIR):
        return []
    return sorted(
        d
        for d in os.listdir(ICON_PACKS_DIR)
        if os.path.isdir(os.path.join(ICON_PACKS_DIR, d))
    )


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, executor, active_pack

    print("Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_model.encode(["warmup"], show_progress_bar=False)

    executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 2)

    try:
        pack = IconPack(ICON_PACK)
        pack.load(embed_model)
        active_pack = pack
    except FileNotFoundError as e:
        print(f"{e} — Búsqueda de íconos deshabilitada.")
        print(f"Packs disponibles: {list_available_packs() or 'ninguno'}")

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


# ── Extracción de PDFs ────────────────────────────────────────────────────────
def _pick_sample_pages(total_pages: int, fraction: float) -> list[int]:
    """
    Devuelve índices de página (0-based) a extraer:
    - Garantiza las primeras `HEAD_PAGES` páginas (si existen).
    - Rellena el resto del cupo con páginas intercaladas uniformemente
      en todo el documento, para capturar contenido del medio/final.
    """
    if total_pages == 0:
        return []

    budget = max(1, round(total_pages * fraction))
    head_idx = list(range(min(HEAD_PAGES, total_pages)))

    if budget <= len(head_idx):
        return head_idx

    remaining = budget - len(head_idx)
    step = total_pages / remaining
    interleaved = {int(i * step) for i in range(remaining)}

    picked = sorted(set(head_idx) | interleaved)
    return picked


def extract_text_from_bytes(
    file_bytes: bytes,
    sample_fraction: float | None = None,
) -> str:
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        total_pages = doc.page_count

        if sample_fraction is None:
            page_indices = range(total_pages)
        else:
            page_indices = _pick_sample_pages(total_pages, sample_fraction)

        parts = [
            doc[i].get_text("text", flags=fitz.TEXTFLAGS_TEXT) for i in page_indices
        ]
        return " ".join(parts)


async def extract_all_texts(
    files,
    executor,
    max_concurrent_reads: int = 20,
    sample_fraction: float | None | str = "auto",
) -> list[tuple[str, str | None, str | None]]:
    """
    Lee y extrae texto de una lista de UploadFile, sin acumular todos los
    bytes en memoria simultáneamente. Devuelve, por archivo, una tupla
    (filename, texto, error): si la extracción falla, texto es None y error
    trae el detalle (para poblar la lista `errors` de la respuesta en vez de
    tumbar todo el request por un solo PDF corrupto).

    sample_fraction:
        "auto" (default) -> decide según cantidad de archivos
        None             -> fuerza full siempre
        float            -> fuerza esa fracción siempre
    """
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

    async def _process(f) -> tuple[str, str | None, str | None]:
        async with sem:
            content = await f.read()
            try:
                text = await loop.run_in_executor(
                    executor, extract_text_from_bytes, content, fraction
                )
                return f.filename, text, None
            except Exception as e:
                return f.filename, None, str(e)
            finally:
                del content

    return await asyncio.gather(*[_process(f) for f in files])


def split_extraction_results(
    extracted: list[tuple[str, str | None, str | None]],
) -> tuple[list[str], list[str], list[dict]]:
    """Separa los resultados de extract_all_texts en (filenames válidos,
    textos limpios, errores). Un documento sin texto extraíble (PDF
    escaneado sin OCR, página en blanco, etc.) también cuenta como error."""
    filenames: list[str] = []
    clean_texts: list[str] = []
    errors: list[dict] = []

    for filename, text, err in extracted:
        if err is not None:
            errors.append({"filename": filename, "detail": err})
            continue
        cleaned = clean_text(text or "")
        if not cleaned:
            errors.append(
                {
                    "filename": filename,
                    "detail": "No se pudo extraer texto del documento.",
                }
            )
            continue
        filenames.append(filename)
        clean_texts.append(cleaned)

    return filenames, clean_texts, errors


# ── Helpers de texto ──────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    for regex, replacement in REGEX_CLEAN:
        text = regex.sub(replacement, text)
    return text.strip()


def is_valid_keyword(kw: str) -> bool:
    kw_lower = kw.lower().strip()
    if len(kw_lower) < 3 or kw_lower in ALL_STOPWORDS:
        return False
    return not any(p in ALL_STOPWORDS for p in kw_lower.split())


def deduplicate_keywords(keywords: list, top_n: int = 30) -> list:
    seen_roots: set[str] = set()
    result = []
    for item in sorted(keywords, key=lambda x: x["score"], reverse=True):
        root = " ".join(w[:6] for w in item["word"].lower().split())
        if root not in seen_roots:
            seen_roots.add(root)
            result.append(item)
        if len(result) == top_n:
            break
    return result


def sample_document(text: str) -> list[str]:
    """Chunks distribuidos a lo largo del documento (inicio, 25%, 50%, 75%)."""
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


# ── Embeddings de documento ───────────────────────────────────────────────────
def compute_doc_embeddings(texts: list[str]) -> np.ndarray:
    """Embede los chunks muestreados de todos los documentos en UN batch y
    promedia por documento. El resultado (n_docs, dim) alimenta keywords
    semánticas, layout 2D y clustering."""
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
        batch_size=128,
        show_progress_bar=False,
    )
    chunk_embs = np.asarray(chunk_embs, dtype="float32")

    doc_embs = np.vstack([chunk_embs[a:b].mean(axis=0) for a, b in spans]).astype(
        "float32"
    )
    return normalize(doc_embs, norm="l2")


# ── TF-IDF: top-k por fila SIN densificar la matriz completa ─────────────────
def top_k_per_row_sparse(sparse_matrix, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Devuelve, por cada fila de una matriz TF-IDF CSR, los índices y scores
    de sus top-k valores — sin nunca llamar a .toarray() sobre la matriz
    completa."""
    sparse_matrix = sparse_matrix.tocsr()
    out = []
    for i in range(sparse_matrix.shape[0]):
        row = sparse_matrix.getrow(i).toarray().ravel()
        if row.size == 0:
            out.append((np.array([], dtype=int), np.array([], dtype="float32")))
            continue
        kk = min(k, row.size)
        top_idx = np.argpartition(row, -kk)[-kk:]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        out.append((top_idx, row[top_idx]))
    return out


# ── Keywords semánticas (MMR vectorizado, sin recalcular top-k) ──────────────
def _mmr_select(
    doc_emb: np.ndarray,
    cand_embs: np.ndarray,
    top_n: int,
    diversity: float = MMR_DIVERSITY,
) -> list[tuple[int, float]]:
    """Maximal Marginal Relevance vectorizado. Devuelve [(idx, score)]."""
    sim_doc = cand_embs @ doc_emb
    n = len(sim_doc)
    if n == 0:
        return []
    sim_between = cand_embs @ cand_embs.T

    selected = [int(np.argmax(sim_doc))]
    remaining = [i for i in range(n) if i != selected[0]]

    while len(selected) < min(top_n, n) and remaining:
        rem = np.array(remaining)
        relevance = sim_doc[rem]
        redundancy = sim_between[np.ix_(rem, selected)].max(axis=1)
        mmr = (1 - diversity) * relevance - diversity * redundancy
        chosen = int(rem[int(np.argmax(mmr))])
        selected.append(chosen)
        remaining.remove(chosen)

    return [(i, float(sim_doc[i])) for i in selected]


def extract_semantic_keywords(
    doc_embs: np.ndarray,
    tfidf_topk: list[tuple[np.ndarray, np.ndarray]],
    feature_names: np.ndarray,
    top_n: int = 40,
) -> list[list[dict]]:
    """Keywords semánticas para todos los documentos con UNA sola llamada a
    encode() sobre el vocabulario de candidatos deduplicado globalmente."""
    assert embed_model is not None
    n_docs = len(tfidf_topk)

    per_doc_candidates: list[list[str]] = []
    vocab: dict[str, int] = {}
    for i in range(n_docs):
        top_idx, top_scores = tfidf_topk[i]
        cands = []
        for j, score in zip(top_idx, top_scores):
            if score <= 0:
                break
            word = feature_names[j]
            if is_valid_keyword(word):
                cands.append(word)
                if word not in vocab:
                    vocab[word] = len(vocab)
            if len(cands) == MAX_CANDIDATES_PER_DOC:
                break
        per_doc_candidates.append(cands)

    if not vocab:
        return [[] for _ in range(n_docs)]

    unique_words = list(vocab.keys())
    word_embs = embed_model.encode(
        unique_words,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=False,
    )
    word_embs = np.asarray(word_embs, dtype="float32")

    results: list[list[dict]] = []
    for i in range(n_docs):
        cands = per_doc_candidates[i]
        if not cands:
            results.append([])
            continue
        local_embs = word_embs[[vocab[w] for w in cands]]
        picked = _mmr_select(doc_embs[i], local_embs, top_n=top_n)
        results.append(
            [{"word": cands[idx], "score": round(score, 4)} for idx, score in picked]
        )
    return results


def merge_keyword_results(per_doc: list[list[dict]], top_n: int = 30) -> list[dict]:
    merged: dict[str, float] = {}
    for doc_kws in per_doc:
        for item in doc_kws:
            word = item["word"].lower()
            merged[word] = max(merged.get(word, 0.0), item["score"])
    combined = [{"word": w, "score": s} for w, s in merged.items()]
    return deduplicate_keywords(combined, top_n=top_n)


# ── Layout 2D (solo para dibujar) + clustering (sobre embeddings crudos) ─────
def compute_2d(doc_embs: np.ndarray) -> np.ndarray | None:
    """Coordenadas 2D reales, SOLO para dibujar. No se usan para clusterizar:
    proyectar a 2D antes de clusterizar pierde estructura y da peor calidad
    que clusterizar sobre los embeddings crudos (ver cluster_by_embedding)."""
    n = doc_embs.shape[0]
    if n < 2:
        return None
    if n == 2:
        return np.array([[0.0, 0.5], [1.0, 0.5]], dtype="float32")
    if n <= 4:
        return PCA(n_components=2, random_state=42).fit_transform(doc_embs)
    reducer = UMAP(
        n_components=2,
        n_neighbors=min(n - 1, 15),
        min_dist=0.15,
        metric="cosine",
        low_memory=True,
        n_jobs=-1,
    )
    return reducer.fit_transform(doc_embs)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Escala a [0,1] con UN solo factor para ambos ejes (el mayor rango de
    los dos), preservando las proporciones reales del layout. Escalar cada
    eje por separado (minmax) deforma el layout si un eje tiene más rango
    que el otro."""
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    span = np.where(maxs - mins == 0, 1, maxs - mins)
    return (coords - mins) / span.max()


def cluster_by_embedding(doc_embs: np.ndarray) -> list[int | None]:
    """HDBSCAN directo sobre los embeddings de documento (384 dims, distancia
    coseno) en vez de sobre la proyección 2D. Es la opción de mejor calidad
    (no hereda la distorsión de comprimir a 2D) y no cuesta una reducción
    extra, así que tampoco es más lenta."""
    n = doc_embs.shape[0]
    if n < 3:
        return [None] * n

    labels = HDBSCAN(
        min_cluster_size=2, min_samples=1, metric="cosine", copy=False
    ).fit_predict(doc_embs.astype("float64"))
    return [int(label) for label in labels]


# ── Queries de íconos (promedio ponderado, batcheado, cacheado) ──────────────
def _cache_key(keywords: list[KeywordItem]) -> tuple:
    return tuple(
        sorted((kw.word.lower().strip(), round(kw.score, 4)) for kw in keywords)
    )


def _cache_get(key: tuple) -> dict | None:
    if key in _query_cache:
        _query_cache.move_to_end(key)
        return _query_cache[key]
    return None


def _cache_put(key: tuple, value: dict) -> None:
    _query_cache[key] = value
    _query_cache.move_to_end(key)
    while len(_query_cache) > QUERY_CACHE_SIZE:
        _query_cache.popitem(last=False)


def build_query_vectors(groups: list[list[KeywordItem]]) -> np.ndarray:
    """Vector semántico de query por grupo = promedio de embeddings de
    keywords ponderado por score."""
    assert embed_model is not None
    unique: dict[str, int] = {}
    for group in groups:
        for kw in group:
            w = kw.word.lower().strip()
            if w and w not in unique:
                unique[w] = len(unique)

    words = list(unique.keys())
    word_embs = embed_model.encode(
        words,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=False,
    )
    word_embs = np.asarray(word_embs, dtype="float32")

    dim = word_embs.shape[1]
    vectors = np.zeros((len(groups), dim), dtype="float32")
    for gi, group in enumerate(groups):
        top = sorted(group, key=lambda x: x.score, reverse=True)[:10]
        acc = np.zeros(dim, dtype="float32")
        for kw in top:
            w = kw.word.lower().strip()
            if w in unique:
                acc += max(kw.score, 0.05) * word_embs[unique[w]]
        norm = np.linalg.norm(acc)
        vectors[gi] = acc / norm if norm > 0 else acc
    return vectors


def resolve_icon_groups(groups: list[list[KeywordItem]], k: int = 3) -> list[dict]:
    """Resuelve N grupos con 1 encode + 1 búsqueda híbrida matricial + cache LRU."""
    assert active_pack is not None

    results: list[dict | None] = [None] * len(groups)
    pending_idx: list[int] = []

    for i, group in enumerate(groups):
        if not group:
            results[i] = {"icon": None, "svg": None, "error": "Grupo vacío"}
            continue
        cached = _cache_get(_cache_key(group))
        if cached is not None:
            results[i] = cached
        else:
            pending_idx.append(i)

    if pending_idx:
        pending_groups = [groups[i] for i in pending_idx]
        query_matrix = build_query_vectors(pending_groups)
        query_keywords = [
            [_norm_phrase(kw.word) for kw in g if kw.word.strip()]
            for g in pending_groups
        ]
        scores, indices = active_pack.search_hybrid(query_matrix, query_keywords, k=k)

        for row, gi in enumerate(pending_idx):
            top = [
                {
                    "name": active_pack.names[idx],
                    "score": round(float(s), 4),
                }
                for s, idx in zip(scores[row], indices[row])
                if 0 <= idx < len(active_pack.names)
            ]
            best = top[0] if top else None

            if best is None or best["score"] < SCORE_THRESHOLD:
                result = {
                    "icon": None,
                    "svg": None,
                    "error": "No superó el umbral",
                    "best_score": best["score"] if best else 0.0,
                    "candidates": [
                        {"icon": active_pack.icon_id(c["name"]), "score": c["score"]}
                        for c in top
                    ],
                }
            else:
                svg = active_pack.get_svg(best["name"])
                if svg is None:
                    result = {
                        "icon": None,
                        "svg": None,
                        "error": f"SVG '{best['name']}' no encontrado",
                    }
                else:
                    result = {
                        "icon": active_pack.icon_id(best["name"]),
                        "svg": svg,
                        "score": best["score"],
                        "candidates": [
                            {
                                "icon": active_pack.icon_id(c["name"]),
                                "score": c["score"],
                            }
                            for c in top
                        ],
                    }

            _cache_put(_cache_key(groups[gi]), result)
            results[gi] = result

    return results  # type: ignore[return-value]


# ── Rutas: PDFs ───────────────────────────────────────────────────────────────
@app.post("/project-pdfs/")
async def project_pdfs(files: List[UploadFile] = File(...)):
    """Proyección 2D + clustering de una lista de documentos. Solo posición y
    cluster: las keywords viven aparte, en /extract-keywords/, para el
    subconjunto de documentos que el frontend quiera convertir en nube."""
    t_total = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    extracted = await extract_all_texts(
        files, executor, max_concurrent_reads=16, sample_fraction="auto"
    )
    timings["extraction"] = round(time.perf_counter() - t0, 4)
    print(f"1. PDF text extraction: {timings['extraction']:.2f}s")

    filenames, clean_texts, errors = split_extraction_results(extracted)
    n = len(clean_texts)
    if n == 0:
        timings["total"] = round(time.perf_counter() - t_total, 4)
        return {
            "global": [],
            "locals": [],
            "similarity": None,
            "errors": errors,
            "timings": timings,
        }

    t0 = time.perf_counter()
    doc_embs = compute_doc_embeddings(clean_texts)
    timings["embeddings"] = round(time.perf_counter() - t0, 4)
    print(f"Doc embeddings ({n} docs): {timings['embeddings']:.2f}s")

    t0 = time.perf_counter()
    coords = compute_2d(doc_embs)
    clusters = cluster_by_embedding(doc_embs)
    layout = normalize_coords(coords) if coords is not None else None
    timings["projection_clustering"] = round(time.perf_counter() - t0, 4)
    print(f"Layout + clustering: {timings['projection_clustering']:.2f}s")

    similarity = None
    if n == 2:
        similarity = round(float(doc_embs[0] @ doc_embs[1]), 4)

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

    timings["total"] = round(time.perf_counter() - t_total, 4)

    response = {
        "global": [],
        "locals": locals_response,
        "similarity": similarity,
        "timings": timings,
    }
    if errors:
        response["errors"] = errors
    return response


@app.post("/extract-keywords/")
async def extract_keywords(files: List[UploadFile] = File(...)):
    """Keywords (TF-IDF local + semánticas) para un subconjunto de documentos,
    usadas por el frontend para armar las nubes de palabras."""
    t_total = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    extracted = await extract_all_texts(
        files, executor, max_concurrent_reads=16, sample_fraction="auto"
    )
    timings["extraction"] = round(time.perf_counter() - t0, 4)
    print(f"1. PDF text extraction: {timings['extraction']:.2f}s")

    filenames, clean_texts, errors = split_extraction_results(extracted)
    n = len(clean_texts)
    if n == 0:
        timings["total"] = round(time.perf_counter() - t_total, 4)
        return {"global": [], "locals": [], "errors": errors, "timings": timings}

    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=5_000,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.7 if n >= 5 else 1.0,
        min_df=2 if n >= 10 else 1,
    )
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    tfidf_topk = top_k_per_row_sparse(tfidf_matrix, k=MAX_CANDIDATES_PER_DOC * 2)

    local_tfidf_keywords = []
    for i in range(n):
        top_idx, top_scores = tfidf_topk[i]
        candidates = [
            {"word": feature_names[j], "score": round(float(s), 4)}
            for j, s in zip(top_idx[:TOP_K_TFIDF_LOCAL], top_scores[:TOP_K_TFIDF_LOCAL])
            if s > 0 and is_valid_keyword(feature_names[j])
        ]
        local_tfidf_keywords.append(deduplicate_keywords(candidates, top_n=30))
    timings["tfidf"] = round(time.perf_counter() - t0, 4)
    print(f"TF-IDF: {timings['tfidf']:.2f}s")

    t0 = time.perf_counter()
    doc_embs = compute_doc_embeddings(clean_texts)
    timings["embeddings"] = round(time.perf_counter() - t0, 4)
    print(f"Doc embeddings ({n} docs): {timings['embeddings']:.2f}s")

    t0 = time.perf_counter()
    per_doc_semantic = extract_semantic_keywords(
        doc_embs, tfidf_topk, feature_names, top_n=40
    )
    global_keywords = merge_keyword_results(per_doc_semantic, top_n=30)
    timings["semantic_keywords"] = round(time.perf_counter() - t0, 4)
    print(f"Keywords semánticas ({n} docs): {timings['semantic_keywords']:.2f}s")

    locals_response = [
        {"filename": filenames[i], "keywords": local_tfidf_keywords[i]}
        for i in range(n)
    ]

    timings["total"] = round(time.perf_counter() - t_total, 4)

    response = {"global": global_keywords, "locals": locals_response, "timings": timings}
    if errors:
        response["errors"] = errors
    return response


# ── Rutas: íconos ─────────────────────────────────────────────────────────────
@app.post("/select-icon/")
async def select_icon(body: IconRequest):
    if not body.keywords:
        raise HTTPException(status_code=400, detail="Se requiere al menos una keyword.")
    if active_pack is None:
        raise HTTPException(status_code=503, detail="Ningún pack de íconos cargado.")

    result = resolve_icon_groups([body.keywords], k=3)[0]

    if result.get("icon") is None:
        if "umbral" in (result.get("error") or ""):
            raise HTTPException(
                status_code=404,
                detail={
                    "message": "Ningún ícono supera el umbral de similitud.",
                    "threshold": SCORE_THRESHOLD,
                    "best_score": result.get("best_score", 0.0),
                    "candidates": result.get("candidates", []),
                },
            )
        raise HTTPException(status_code=500, detail=result.get("error"))

    avg_score = round(sum(k.score for k in body.keywords) / len(body.keywords), 4)
    return {
        "icon": result["icon"],
        "svg": result["svg"],
        "score": result["score"],
        "avg_kw_score": avg_score,
        "source": "local",
        "pack": active_pack.name,
        "candidates": result.get("candidates", []),
    }


@app.post("/select-icons-batch/")
async def select_icons_batch(body: BatchIconRequest):
    if not body.groups:
        raise HTTPException(
            status_code=400, detail="No se enviaron grupos de palabras clave."
        )
    if active_pack is None:
        raise HTTPException(status_code=503, detail="Ningún pack de íconos cargado.")

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(executor, resolve_icon_groups, body.groups)
    return {"icons": results, "pack": active_pack.name}


# ── Rutas: gestión de packs ───────────────────────────────────────────────────
@app.get("/packs/")
async def get_packs():
    return {
        "active": active_pack.name if active_pack else None,
        "available": list_available_packs(),
    }


@app.post("/packs/{name}/activate")
async def activate_pack(name: str):
    """Cambia de pack en caliente, sin reiniciar el servidor."""
    global active_pack
    try:
        pack = IconPack(name)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Pack '{name}' no existe.",
                "available": list_available_packs(),
            },
        )

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, pack.load, embed_model)

    active_pack = pack
    _query_cache.clear()
    return {
        "active": pack.name,
        "prefix": pack.prefix,
        "icons": len(pack.names),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
