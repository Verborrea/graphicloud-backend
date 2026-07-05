"""
python local.py --input ./pdfs --output resultado.json


python local.py \
    --input ./pdfs \
    --output resultado.json \
    --top-n-global 30 \
    --top-n-local 30 \
    --top-n-semantic 40 \
    --max-features 5000 \
    --sample-fraction auto \
    --workers 8 \
    --batch-size 256 \
    --max-files 2000 \
    --recursive

Si tu archivo del servidor NO se llama main.py, cambia MODULE_NAME aquí
abajo (sin la extensión .py), o pasa --module nombre_del_archivo.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_MODULE_NAME = "main"


def load_server_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        print(
            f"ERROR: no pude importar '{module_name}.py'. Asegúrate de que "
            f"este script esté en la misma carpeta que tu servidor, o usa "
            f"--module nombre_del_archivo.\nDetalle: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def collect_pdf_paths(
    input_dir: Path, recursive: bool, max_files: int | None
) -> list[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    paths = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def read_pdfs_to_bytes(paths: list[Path]) -> list[tuple[str, bytes]]:
    """Lee todos los PDFs a memoria como (nombre, bytes). Con 2000 PDFs
    típicos (unos pocos MB cada uno) esto entra cómodo en RAM; si tus
    archivos son enormes, este es el punto a cambiar por lectura en
    streaming/chunks."""
    out = []
    for p in paths:
        try:
            out.append((p.name, p.read_bytes()))
        except Exception as e:
            print(f"  [WARN] no pude leer {p.name}: {e}")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Procesa una carpeta de PDFs localmente con el mismo pipeline que /convert-pdfs/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path, help="Carpeta con los PDFs."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="resultado.json",
        type=Path,
        help="Archivo JSON de salida.",
    )
    parser.add_argument(
        "--module",
        default=DEFAULT_MODULE_NAME,
        help="Nombre del módulo del servidor (sin .py) a importar.",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Buscar PDFs en subcarpetas también."
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Límite de archivos a procesar (debug rápido).",
    )

    # Equivalentes a las constantes del módulo, pero overrideables por CLI.
    parser.add_argument(
        "--top-n-global",
        type=int,
        default=30,
        help="Cantidad de keywords globales finales.",
    )
    parser.add_argument(
        "--top-n-local",
        type=int,
        default=30,
        help="Cantidad de keywords TF-IDF locales por doc.",
    )
    parser.add_argument(
        "--top-n-semantic",
        type=int,
        default=40,
        help="Cantidad de keywords semánticas (MMR) por doc antes de mergear.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="max_features de TfidfVectorizer.",
    )
    parser.add_argument(
        "--sample-fraction",
        default="auto",
        help="'auto' (igual que el server), 'full' (sin muestreo) o un float 0-1 (ej: 0.3).",
    )
    parser.add_argument(
        "--max-concurrent-reads",
        type=int,
        default=32,
        help="Lecturas/extracciones PDF concurrentes (asyncio).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Hilos del ThreadPoolExecutor (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=256,
        help="Batch size para SentenceTransformer.encode().",
    )
    parser.add_argument(
        "--no-pretty",
        action="store_true",
        help="Guardar JSON compacto en vez de indentado.",
    )

    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"ERROR: '{args.input}' no es una carpeta válida.", file=sys.stderr)
        sys.exit(1)

    # Parseo de sample_fraction: 'auto' | 'full' (-> None) | float
    sample_fraction = args.sample_fraction
    if sample_fraction == "full":
        sample_fraction = None
    elif sample_fraction != "auto":
        try:
            sample_fraction = float(sample_fraction)
        except ValueError:
            print(
                "ERROR: --sample-fraction debe ser 'auto', 'full' o un número 0-1.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Importando pipeline desde '{args.module}.py'...")
    srv = load_server_module(args.module)

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    # ── 0. Modelo de embeddings (igual que en el lifespan del server) ──────
    print("Cargando modelo de embeddings (all-MiniLM-L6-v2)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_model.encode(["warmup"], show_progress_bar=False)
    srv.embed_model = embed_model  # las funciones del módulo leen esta global

    executor = ThreadPoolExecutor(max_workers=args.workers)

    # ── 1. Recolectar PDFs ──────────────────────────────────────────────────
    pdf_paths = collect_pdf_paths(args.input, args.recursive, args.max_files)
    if not pdf_paths:
        print(
            f"No se encontraron PDFs en '{args.input}'"
            + (" (recursivo)" if args.recursive else ""),
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Encontrados {len(pdf_paths)} PDFs.")

    # ── 2. Extracción de texto (reutiliza extract_text_from_bytes) ─────────
    t0 = time.perf_counter()

    n_files = len(pdf_paths)
    fraction = sample_fraction
    if sample_fraction == "auto":
        if n_files <= 10:
            fraction = None
        elif n_files <= 50:
            fraction = 0.5
        elif n_files <= 100:
            fraction = 0.25
        else:
            fraction = 0.1
        print(f"sample_fraction='auto' -> {fraction} para {n_files} archivos.")

    async def extract_all():
        sem = asyncio.Semaphore(args.max_concurrent_reads)
        loop = asyncio.get_running_loop()

        async def _process(path: Path):
            async with sem:
                try:
                    content = await loop.run_in_executor(None, path.read_bytes)
                    text = await loop.run_in_executor(
                        executor, srv.extract_text_from_bytes, content, fraction
                    )
                    return path.name, text
                except Exception as e:
                    print(f"  [WARN] error procesando {path.name}: {e}")
                    return path.name, ""

        return await asyncio.gather(*[_process(p) for p in pdf_paths])

    extracted = asyncio.run(extract_all())
    filenames = [name for name, _ in extracted]
    raw_texts = [text for _, text in extracted]
    print(f"1. Extracción de texto: {time.perf_counter() - t0:.2f}s")

    clean_texts = [srv.clean_text(t) for t in raw_texts]
    n = len(clean_texts)

    empty_count = sum(1 for t in clean_texts if not t.strip())
    if empty_count:
        print(
            f"  [INFO] {empty_count} documento(s) quedaron con texto vacío tras la extracción."
        )

    # ── 3. TF-IDF (igual lógica que /convert-pdfs/) ─────────────────────────
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.7 if n >= 5 else 1.0,
        min_df=2 if n >= 10 else 1,
    )
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    tfidf_topk = srv.top_k_per_row_sparse(
        tfidf_matrix, k=srv.MAX_CANDIDATES_PER_DOC * 2
    )

    local_tfidf_keywords = []
    for i in range(n):
        top_idx, top_scores = tfidf_topk[i]
        candidates = [
            {"word": feature_names[j], "score": round(float(s), 4)}
            for j, s in zip(top_idx[: args.top_n_local], top_scores[: args.top_n_local])
            if s > 0 and srv.is_valid_keyword(feature_names[j])
        ]
        local_tfidf_keywords.append(
            srv.deduplicate_keywords(candidates, top_n=args.top_n_local)
        )
    print(f"2. TF-IDF: {time.perf_counter() - t0:.2f}s")

    # ── 4. Embeddings de documento ──────────────────────────────────────────
    t0 = time.perf_counter()
    doc_embs = srv.compute_doc_embeddings(clean_texts)
    print(f"3. Doc embeddings ({n} docs): {time.perf_counter() - t0:.2f}s")

    # ── 5. Keywords semánticas (MMR) + merge global ────────────────────────
    t0 = time.perf_counter()
    per_doc_semantic = srv.extract_semantic_keywords(
        doc_embs, tfidf_topk, feature_names, top_n=args.top_n_semantic
    )
    global_keywords = srv.merge_keyword_results(
        per_doc_semantic, top_n=args.top_n_global
    )
    print(f"4. Keywords semánticas ({n} docs): {time.perf_counter() - t0:.2f}s")

    # ── 6. Layout 2D + clustering ────────────────────────────────────────────
    t0 = time.perf_counter()
    coords = srv.compute_2d(doc_embs)
    clusters = srv.cluster_2d(coords)
    layout = srv.normalize_xy(coords) if coords is not None else None
    print(f"5. Layout + clustering: {time.perf_counter() - t0:.2f}s")

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
                "keywords": local_tfidf_keywords[i],
            }
            for i in range(n)
        ]

    result = {
        "global": global_keywords,
        "locals": locals_response,
        "similarity": similarity,
        "meta": {
            "n_files": n,
            "sample_fraction": fraction,
            "max_features": args.max_features,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        if args.no_pretty:
            json.dump(result, f, ensure_ascii=False)
        else:
            json.dump(result, f, ensure_ascii=False, indent=2)

    executor.shutdown(wait=False)
    print(f"\nListo. Resultado guardado en: {args.output}")


if __name__ == "__main__":
    main()
