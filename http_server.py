from __future__ import annotations

import io
import os
import tempfile
import traceback
from threading import Lock, Semaphore
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device


app = FastAPI()
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"

MODEL_ID = os.environ.get("TTS_MODEL_ID", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("OMNIVOICE_DEVICE") or get_best_device()

torch.set_num_threads(int(os.environ.get("TTS_TORCH_THREADS", "1")))
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


def _dtype() -> torch.dtype:
    configured = os.environ.get("OMNIVOICE_DTYPE")
    if configured:
        return getattr(torch, configured)
    return torch.float16 if DEVICE.startswith(("cuda", "xpu", "mps")) else torch.float32


MODEL = OmniVoice.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=_dtype())
MODEL_LOCK = Lock()
QUEUE = Semaphore(int(os.environ.get("OMNIVOICE_WORKER_BACKLOG", "1")))


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_value(value: str | None, default: int) -> int:
    return default if value is None or value == "" else int(value)


def _float_value(value: str | None, default: float | None) -> float | None:
    return default if value is None or value == "" else float(value)


def _string_value(value: str | None, default: str | None = None) -> str | None:
    value = value.strip() if value is not None else None
    return value if value else default


def _env_float(name: str, default: float | None) -> float | None:
    return _float_value(os.environ.get(name), default)


def _env_int(name: str, default: int) -> int:
    return _int_value(os.environ.get(name), default)


def _env_bool(name: str, default: bool) -> bool:
    return _bool(os.environ.get(name), default)


def _write_wav(audio: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    sf.write(buf, audio, MODEL.sampling_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


async def _save_upload(upload: UploadFile | None) -> str | None:
    if upload is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await upload.read())
        return tmp.name


def _remove(path: str | None) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL_ID, "device": DEVICE}


@app.exception_handler(Exception)
async def _exception_handler(_request: Request, exc: Exception):
    if DEBUG_MODE:
        return JSONResponse({"error": str(exc), "traceback": traceback.format_exc()}, status_code=500)
    return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form("en"),
    audio: UploadFile | None = File(None),
    ref_text: str | None = Form(None),
    instruct: str | None = Form(None),
    num_step: str | None = Form(None),
    guidance_scale: str | None = Form(None),
    speed: str | None = Form(None),
    duration: str | None = Form(None),
    t_shift: str | None = Form(None),
    denoise: str | None = Form(None),
    postprocess_output: str | None = Form(None),
    layer_penalty_factor: str | None = Form(None),
    position_temperature: str | None = Form(None),
    class_temperature: str | None = Form(None),
):
    if not text.strip():
        return JSONResponse({"error": "Missing text"}, status_code=400)

    upload_path = await _save_upload(audio)
    ref_audio = upload_path or _string_value(os.environ.get("OMNIVOICE_REF_AUDIO"))
    if ref_audio and not os.path.exists(ref_audio):
        _remove(upload_path)
        return JSONResponse({"error": f"Reference audio not found: {ref_audio}"}, status_code=400)

    kwargs: dict[str, Any] = {
        "text":                  text,
        "language":              _string_value(language),
        "ref_audio":             ref_audio,
        "ref_text":              _string_value(ref_text, os.environ.get("OMNIVOICE_REF_TEXT")),
        "instruct":              _string_value(instruct, os.environ.get("OMNIVOICE_INSTRUCT")),
        "num_step":              _int_value(num_step, _env_int("OMNIVOICE_NUM_STEP", 32)),
        "guidance_scale":        _float_value(guidance_scale, _env_float("OMNIVOICE_GUIDANCE_SCALE", 2.0)),
        "speed":                 _float_value(speed, _env_float("OMNIVOICE_SPEED", 0.92)),
        "duration":              _float_value(duration, _env_float("OMNIVOICE_DURATION", None)),
        "t_shift":               _float_value(t_shift, _env_float("OMNIVOICE_T_SHIFT", 0.1)),
        "denoise":               _bool(denoise, _env_bool("OMNIVOICE_DENOISE", True)),
        "postprocess_output":    _bool(postprocess_output, _env_bool("OMNIVOICE_POSTPROCESS_OUTPUT", True)),
        "layer_penalty_factor":  _float_value(layer_penalty_factor, _env_float("OMNIVOICE_LAYER_PENALTY_FACTOR", 5.0)),
        "position_temperature":  _float_value(position_temperature, _env_float("OMNIVOICE_POSITION_TEMPERATURE", 5.0)),
        "class_temperature":     _float_value(class_temperature, _env_float("OMNIVOICE_CLASS_TEMPERATURE", 0.0)),
    }

    if kwargs["ref_audio"]:
        kwargs["instruct"] = None

    try:
        QUEUE.acquire()
        with MODEL_LOCK, torch.inference_mode():
            audios = MODEL.generate(**kwargs)
        wav = _write_wav(audios[0])
        return StreamingResponse(
            wav,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="output.wav"'},
        )
    finally:
        QUEUE.release()
        _remove(upload_path)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "10440"))
    uvicorn.run(app, host=host, port=port)
