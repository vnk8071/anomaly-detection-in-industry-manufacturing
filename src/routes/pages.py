from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/index")


@router.get("/index", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@router.get("/document", response_class=HTMLResponse)
async def document(request: Request):
    return templates.TemplateResponse(request=request, name="document.html")
