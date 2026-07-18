from __future__ import annotations

import io
import base64
import hashlib
import json
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
from cachetools import TTLCache
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
PROMPT_CACHE = TTLCache(
    maxsize=int(os.environ.get("OMNIVOICE_PROMPT_CACHE_SIZE", "8")),
    ttl=int(os.environ.get("OMNIVOICE_PROMPT_CACHE_TTL", "86400")),
)
PROMPT_CACHE_HITS = 0
PROMPT_CACHE_MISSES = 0


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _string_value(value: str | None, default: str | None = None) -> str | None:
    value = value.strip() if value is not None else None
    return value if value else default


def _generation_options(
    num_step: str | None,
    guidance_scale: str | None,
    speed: str | None,
    duration: str | None,
    t_shift: str | None,
    denoise: str | None,
    postprocess_output: str | None,
    layer_penalty_factor: str | None,
    position_temperature: str | None,
    class_temperature: str | None,
) -> dict[str, Any]:
    options = {}
    values = {
        "num_step": num_step,
        "guidance_scale": guidance_scale,
        "speed": speed,
        "duration": duration,
        "t_shift": t_shift,
        "denoise": denoise,
        "postprocess_output": postprocess_output,
        "layer_penalty_factor": layer_penalty_factor,
        "position_temperature": position_temperature,
        "class_temperature": class_temperature,
    }
    converters = {
        "num_step": int,
        "denoise": _bool,
        "postprocess_output": _bool,
    }
    for name, value in values.items():
        if value is not None and value.strip():
            options[name] = converters.get(name, float)(value)
    return options


def _write_wav(audio: np.ndarray) -> io.BytesIO:
    buf = io.BytesIO()
    sf.write(buf, audio, MODEL.sampling_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf


def _wav_base64(audio: np.ndarray) -> str:
    return base64.b64encode(_write_wav(audio).getvalue()).decode("ascii")


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _voice_clone_prompt(ref_audio: str, ref_text: str | None):
    global PROMPT_CACHE_HITS, PROMPT_CACHE_MISSES

    key = (_file_sha256(ref_audio), ref_text or "")
    prompt = PROMPT_CACHE.get(key)
    if prompt is not None:
        PROMPT_CACHE_HITS += 1
        return prompt

    PROMPT_CACHE_MISSES += 1
    prompt = MODEL.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    PROMPT_CACHE[key] = prompt
    return prompt


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
    return {
        "ok": True,
        "model": MODEL_ID,
        "device": DEVICE,
        "prompt_cache": {
            "size": len(PROMPT_CACHE),
            "maxsize": PROMPT_CACHE.maxsize,
            "ttl": PROMPT_CACHE.ttl,
            "hits": PROMPT_CACHE_HITS,
            "misses": PROMPT_CACHE_MISSES,
        },
    }


@app.exception_handler(Exception)
async def _exception_handler(_request: Request, exc: Exception):
    if DEBUG_MODE:
        return JSONResponse(
            {"error": str(exc), "traceback": traceback.format_exc()}, status_code=500
        )
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
        return JSONResponse(
            {"error": f"Reference audio not found: {ref_audio}"}, status_code=400
        )

    kwargs: dict[str, Any] = {
        "text": text,
        "language": _string_value(language),
        "ref_audio": ref_audio,
        "ref_text": _string_value(ref_text, os.environ.get("OMNIVOICE_REF_TEXT")),
        "instruct": _string_value(instruct, os.environ.get("OMNIVOICE_INSTRUCT")),
    }
    kwargs.update(
        _generation_options(
            num_step,
            guidance_scale,
            speed,
            duration,
            t_shift,
            denoise,
            postprocess_output,
            layer_penalty_factor,
            position_temperature,
            class_temperature,
        )
    )

    ref_audio_for_prompt = None
    ref_text_for_prompt = None
    if kwargs["ref_audio"]:
        ref_audio_for_prompt = kwargs.pop("ref_audio")
        ref_text_for_prompt = kwargs.pop("ref_text")
        kwargs["instruct"] = None

    try:
        QUEUE.acquire()
        with MODEL_LOCK, torch.inference_mode():
            if ref_audio_for_prompt:
                kwargs["voice_clone_prompt"] = _voice_clone_prompt(
                    ref_audio_for_prompt,
                    ref_text_for_prompt,
                )
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


@app.post("/synthesize_batch")
async def synthesize_batch(
    items: str = Form(...),
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
    try:
        parsed_items = json.loads(items)
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": f"Invalid items JSON: {exc}"}, status_code=400)

    if not isinstance(parsed_items, list) or not parsed_items:
        return JSONResponse(
            {"error": "items must be a non-empty array"}, status_code=400
        )

    texts = []
    languages = []
    for idx, item in enumerate(parsed_items):
        text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
        if not text:
            return JSONResponse(
                {"error": f"Missing text for item {idx}"}, status_code=400
            )
        texts.append(text)
        item_language = item.get("language") if isinstance(item, dict) else None
        languages.append(_string_value(item_language, _string_value(language)))

    upload_path = await _save_upload(audio)
    ref_audio = upload_path or _string_value(os.environ.get("OMNIVOICE_REF_AUDIO"))
    if ref_audio and not os.path.exists(ref_audio):
        _remove(upload_path)
        return JSONResponse(
            {"error": f"Reference audio not found: {ref_audio}"}, status_code=400
        )

    kwargs: dict[str, Any] = {
        "text": texts,
        "language": languages,
        "instruct": _string_value(instruct, os.environ.get("OMNIVOICE_INSTRUCT")),
    }
    kwargs.update(
        _generation_options(
            num_step,
            guidance_scale,
            speed,
            duration,
            t_shift,
            denoise,
            postprocess_output,
            layer_penalty_factor,
            position_temperature,
            class_temperature,
        )
    )

    try:
        QUEUE.acquire()
        with MODEL_LOCK, torch.inference_mode():
            if ref_audio:
                kwargs["voice_clone_prompt"] = _voice_clone_prompt(
                    ref_audio,
                    _string_value(ref_text, os.environ.get("OMNIVOICE_REF_TEXT")),
                )
                kwargs["instruct"] = None
            audios = MODEL.generate(**kwargs)
        return JSONResponse(
            {"items": [{"audio": _wav_base64(audio)} for audio in audios]}
        )
    finally:
        QUEUE.release()
        _remove(upload_path)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "10440"))
    uvicorn.run(app, host=host, port=port)
