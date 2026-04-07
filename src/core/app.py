import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.routes.inference import router as inference_router
from src.routes.pages import router as pages_router
from src.routes.train import router as train_router


def create_app() -> FastAPI:
    # Required by anomalib 2.x to allow loading .pt models saved with pickle
    os.environ.setdefault("TRUST_REMOTE_CODE", "1")

    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    app = FastAPI(title="Anomaly Detection Pipeline")
    app.mount("/static", StaticFiles(directory="static"), name="static")

    app.include_router(pages_router)
    app.include_router(train_router)
    app.include_router(inference_router)

    return app
