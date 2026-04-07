import datetime
import os

import cv2
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.anomaly.inferencer import AnomalyInferencer

router = APIRouter()
templates = Jinja2Templates(directory="templates")

MODELS_DIR = "./models"
UPLOAD_DIR = "static/uploads"


def _list_models() -> list[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    return [f.rsplit(".", 1)[0] for f in os.listdir(MODELS_DIR) if f.endswith(".pt")]


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%d%m%Y_%H%M%S")


@router.get("/inference", response_class=HTMLResponse)
async def inference_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="inference.html",
        context={"user_select": _list_models(), "message": None},
    )


@router.post("/inference", response_class=HTMLResponse)
async def inference_post(
    request: Request,
    image_input: UploadFile = File(...),
    user_category: str = Form(...),
):
    if not image_input.filename:
        return templates.TemplateResponse(
            request=request,
            name="inference.html",
            context={"user_select": _list_models(), "message": "Please attach an image file."},
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    now = _timestamp()
    stem = image_input.filename.rsplit(".", 1)[0]
    image_upload_path = f"{UPLOAD_DIR}/{now}_{stem}.jpg"

    contents = await image_input.read()
    with open(image_upload_path, "wb") as f:
        f.write(contents)

    ckpt_path = os.path.join(MODELS_DIR, f"{user_category}.pt")
    if not os.path.exists(ckpt_path):
        return templates.TemplateResponse(
            request=request,
            name="inference.html",
            context={
                "user_select": _list_models(),
                "message": f"Model checkpoint not found: {ckpt_path}",
            },
        )

    inferencer = AnomalyInferencer(ckpt_path)
    result = inferencer.predict(image_upload_path)

    base_path = f"{UPLOAD_DIR}/{now}_{stem}"
    heatmap_path = base_path + "_heatmap.jpg"
    mask_path = base_path + "_mask.jpg"
    segment_path = base_path + "_segment.jpg"

    cv2.imwrite(heatmap_path, result["heat_map"])
    cv2.imwrite(mask_path, (result["pred_mask"] * 255).astype("uint8"))
    cv2.imwrite(segment_path, result["segmentation"])

    return templates.TemplateResponse(
        request=request,
        name="result_inference.html",
        context={
            "image_upload": image_upload_path,
            "anomaly_score": result["pred_score"],
            "target": result["pred_label"],
            "heatmap_predict": heatmap_path,
            "mask_predict": mask_path,
            "segment_predict": segment_path,
        },
    )
