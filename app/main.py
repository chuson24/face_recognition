import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, pipeline
from .face_db import face_db

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = (BASE_DIR / "templates" / "index.html").read_text(encoding="utf-8")

app = FastAPI(title="Nhan dien nguoi qua CCTV")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/captures", StaticFiles(directory=str(config.ENROLL_DIR)), name="captures")


@app.on_event("startup")
def _startup():
    pipeline.start()


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


def _mjpeg_generator():
    # Chi gui khi co frame moi (JPEG da duoc encode san trong pipeline), khong
    # encode lai cho moi trinh duyet.
    boundary = b"--frame"
    last_seq = -1
    while True:
        jpg, seq = pipeline.wait_for_jpeg(last_seq, timeout=1.0)
        if jpg is None or seq == last_seq:
            continue
        last_seq = seq
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/rtsp")
def get_rtsp():
    return {"source": config.read_rtsp_source(), "connected": pipeline.camera_connected()}


@app.post("/api/rtsp")
async def set_rtsp(request: Request):
    data = await request.json()
    source = str(data.get("source", "")).strip()
    if not source:
        return JSONResponse({"error": "Thieu duong dan RTSP"}, status_code=400)
    config.write_rtsp_source(source)
    pipeline.reload_camera(source)
    return {"ok": True}


@app.get("/api/identities")
def list_identities():
    return {"people": face_db.list_people()}


@app.delete("/api/identities/{name}")
def delete_identity(name: str):
    return {"ok": face_db.delete_person(name)}


@app.post("/api/enroll/start")
async def enroll_start(request: Request):
    data = await request.json()
    name = str(data.get("name", "")).strip()
    target = data.get("target", 15)
    auto = bool(data.get("auto", True))
    try:
        pipeline.start_enrollment(name, target, auto)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"ok": True}


@app.post("/api/enroll/capture")
def enroll_capture():
    # Ham dong bo (chay trong threadpool): cho luong infer chup o frame ke tiep.
    return pipeline.request_manual_capture()


@app.post("/api/enroll/auto")
async def enroll_auto(request: Request):
    data = await request.json()
    pipeline.set_enroll_auto(bool(data.get("auto", True)))
    return {"ok": True}


@app.post("/api/enroll/cancel")
def enroll_cancel():
    pipeline.cancel_enrollment()
    return {"ok": True}


@app.get("/api/enroll/status")
def enroll_status():
    return pipeline.enrollment_status()


@app.get("/api/settings")
def get_settings():
    return config.settings.get_all()


@app.post("/api/settings")
async def update_settings(request: Request):
    data = await request.json()
    return config.settings.update(data)
