import os
import zipfile

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.anomaly.trainer import MODEL_REGISTRY, train_model

router = APIRouter()
templates = Jinja2Templates(directory="templates")

USER_MODELS = list(MODEL_REGISTRY.keys())
DATA_DIR = "./data"
MODELS_DIR = "./models"
RESULTS_DIR = "./results"
UPLOAD_DIR = "static/uploads"


@router.get("/train", response_class=HTMLResponse)
async def train_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="train.html",
        context={"user_models": USER_MODELS, "message": None, "success": False},
    )


@router.post("/train", response_class=HTMLResponse)
async def train_post(
    request: Request,
    zip_input: UploadFile = File(...),
    user_model: str = Form("patchcore"),
    user_backbone: str = Form("resnet18"),
):
    if not zip_input.filename:
        return templates.TemplateResponse(
            request=request,
            name="train.html",
            context={
                "user_models": USER_MODELS,
                "message": "Please attach a zip file for training.",
                "success": False,
            },
        )

    dataset_name = zip_input.filename.rsplit(".", 1)[0]
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    zip_path = os.path.join(UPLOAD_DIR, zip_input.filename)

    contents = await zip_input.read()
    with open(zip_path, "wb") as f:
        f.write(contents)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    dataset_root = os.path.join(DATA_DIR, dataset_name)

    try:
        ckpt_path = train_model(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            model_name=user_model,
            backbone=user_backbone,
            output_dir=RESULTS_DIR,
            models_dir=MODELS_DIR,
        )
        message = f"Training successful! Model saved to {ckpt_path}"
        success = True
    except Exception as exc:
        message = f"Training failed: {exc}"
        success = False

    return templates.TemplateResponse(
        request=request,
        name="train.html",
        context={"user_models": USER_MODELS, "message": message, "success": success},
    )
