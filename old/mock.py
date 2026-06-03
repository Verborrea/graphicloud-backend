import random
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ACADEMIC_WORDS = [
    "analysis",
    "framework",
    "dynamic",
    "systems",
    "intelligence",
    "neural",
    "learning",
    "stochastic",
    "model",
    "optimization",
    "data",
    "quantum",
    "algorithm",
    "theory",
    "distribution",
    "variance",
    "heuristic",
    "entropy",
    "gradient",
    "topology",
    "validation",
    "accuracy",
    "performance",
    "scalability",
    "latency",
    "protocol",
    "architecture",
    "integration",
    "paradigm",
    "inference",
    "simulation",
    "matrix",
    "regression",
    "correlation",
    "clustering",
    "classification",
    "supervision",
    "synthesis",
    "protocol",
    "robustness",
    "efficiency",
    "bandwidth",
    "node",
    "cluster",
    "storage",
    "compute",
    "parallel",
    "concurrency",
    "semantics",
    "logic",
]


def generate_keywords(n: int) -> List[dict]:
    """Genera N keywords con scores aleatorios descendentes."""
    words = random.sample(ACADEMIC_WORDS, min(n, len(ACADEMIC_WORDS)))
    return sorted(
        [{"word": w, "score": round(random.uniform(0.1, 0.9), 4)} for w in words],
        key=lambda x: x["score"],
        reverse=True,
    )


@app.post("/convert-pdfs/")
async def convert_pdfs_mock(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    global_keywords = generate_keywords(30)

    locals_response = []
    for file in files:
        num_keywords = 30

        locals_response.append(
            {
                "filename": file.filename,
                "x": round(random.random(), 4),
                "y": round(random.random(), 4),
                "keywords": generate_keywords(num_keywords),
            }
        )

    return {"global": global_keywords, "locals": locals_response}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
