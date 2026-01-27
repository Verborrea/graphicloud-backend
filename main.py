import asyncio
import re
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List

import fitz
import numpy as np
import spacy
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargamos spaCy optimizado (solo lematización)
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
executor = ProcessPoolExecutor()


def clean_paper_noise(text: str) -> str:
    """Limpieza quirúrgica de ruido en papers."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"doi[:/]\s?\S+", "", text)
    text = re.sub(r"\[\d+(,\s*\d+)*\]", "", text)
    text = re.sub(r"\b\d+\b", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text(file_bytes: bytes) -> str:
    """Extracción rápida con PyMuPDF."""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        return " ".join([page.get_text() for page in doc])


@app.post("/convert-pdfs/")
async def convert_pdfs(
    files: List[UploadFile] = File(...),
    n_top: int = Form(10),
):
    # --- RESTRICCIÓN DE SEGURIDAD ---
    if len(files) < 3:
        raise HTTPException(
            status_code=400,
            detail="Se requieren al menos 3 documentos para la proyección UMAP.",
        )

    t_start = time.perf_counter()
    loop = asyncio.get_event_loop()

    # 1. Extracción paralela
    file_contents = [await f.read() for f in files]
    tasks = [
        loop.run_in_executor(executor, extract_text, content)
        for content in file_contents
    ]
    raw_texts = await asyncio.gather(*tasks)

    # 2. NLP (Lematización) en paralelo
    pre_cleaned = [clean_paper_noise(t) for t in raw_texts]
    processed_texts = []
    for doc in nlp.pipe(pre_cleaned, batch_size=5, n_process=-1):
        tokens = [
            t.lemma_.lower()
            for t in doc
            if t.is_alpha and not t.is_stop and len(t.text) > 2
        ]
        processed_texts.append(" ".join(tokens))

    # 3. Vectorización TF-IDF
    vectorizer = TfidfVectorizer(max_features=2100)
    matrix = vectorizer.fit_transform(processed_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # 4. Reducción UMAP (2D)
    matrix_dense = matrix.toarray()
    n_neighbors = min(len(processed_texts) - 1, 15)

    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1)
    embedding = reducer.fit_transform(matrix_dense)

    # 5. Normalización 0-1
    mins = embedding.min(axis=0)
    maxs = embedding.max(axis=0)
    # Evitar división por cero si todos los puntos caen en el mismo lugar
    ranges = np.where(maxs - mins == 0, 1, maxs - mins)
    embedding_norm = (embedding - mins) / ranges

    # 6. Construcción de respuesta (Ranking por documento + Coordenadas)
    results = []
    for i in range(len(files)):
        # Obtener pesos TF-IDF del documento i
        row = matrix.getrow(i).toarray()[0]
        top_indices = row.argsort()[-n_top:][::-1]

        results.append(
            {
                "filename": files[i].filename,
                "x": float(embedding_norm[i, 0]),
                "y": float(embedding_norm[i, 1]),
                "keywords": [
                    {"word": feature_names[idx], "score": round(float(row[idx]), 4)}
                    for idx in top_indices
                    if row[idx] > 0
                ],
            }
        )

    t_end = time.perf_counter()
    print(f"✅ Análisis completo: {len(files)} docs en {t_end - t_start:.3f}s")

    return results
