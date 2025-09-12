from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routes import cuda, infer, misc, uploads

# Template and static file paths
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="NeuroServe")

# Serve static files under /static if available
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """
    Serve the homepage template.

    Args:
        request (Request): The HTTP request object.

    Returns:
        TemplateResponse: Rendered index.html template.
    """
    return templates.TemplateResponse("index.html", {"request": request})


# Include routers for CUDA and file uploads
app.include_router(cuda.router, tags=["cuda"])
app.include_router(uploads.router, prefix="/upload", tags=["uploads"])
app.include_router(misc.router, tags=["misc"])
app.include_router(infer.router)
