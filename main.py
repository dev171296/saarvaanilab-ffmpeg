from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
import subprocess
import requests
import os
import tempfile
import shutil
import logging
import urllib.parse
import time
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SaarVaaniLab FFmpeg Service")


class VideoRequest(BaseModel):
    image_prompts: List[str]       # 7 image prompts (raw text, service encodes)
    audio_url: str                  # Google Drive webContentLink for the MP3
    video_number: str               # e.g. "001"
    duration_per_image: float = 9.0   # seconds per image (reduced for faster FFmpeg)


@app.get("/")
def root():
    return {"status": "alive", "service": "SaarVaaniLab FFmpeg", "version": "1.1"}


@app.get("/ping")
def ping():
    return {"pong": True}


def _download_single_image(args):
    """Download one image from Pollinations — called inside a thread pool."""
    i, prompt, work_dir = args
    seed = 1001 + i
    encoded = urllib.parse.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width=1080&height=1920&model=flux&seed={seed}&nologo=true"
    )
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            img_path = os.path.join(work_dir, f"img_{i:02d}.jpg")
            with open(img_path, "wb") as f:
                f.write(r.content)
            logger.info(f"  Image {i+1} ✓ ({len(r.content)//1024} KB)")
            return i, img_path
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Image {i+1} failed after 3 attempts: {e}")
            time.sleep(3)


@app.post("/assemble")
async def assemble_video(req: VideoRequest):
    work_dir = tempfile.mkdtemp()
    t0 = time.time()
    logger.info(f"[{req.video_number}] Starting assembly in {work_dir}")

    try:
        # ── Step 1: Download all 7 images IN PARALLEL ──────────────────────────
        # Sequential downloads took ~90-120s, hitting Cloudflare's ~100s proxy
        # timeout → 502. Parallel downloads finish in ~20-30s.
        logger.info(f"[{req.video_number}] Downloading {len(req.image_prompts)} images in parallel …")
        t1 = time.time()

        args_list = [(i, p, work_dir) for i, p in enumerate(req.image_prompts)]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            futures = {pool.submit(_download_single_image, a): a[0] for a in args_list}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except RuntimeError as e:
                    raise HTTPException(status_code=502, detail=str(e))

        results.sort(key=lambda x: x[0])           # ensure correct scene order 0-6
        image_paths = [path for _, path in results]
        logger.info(f"[{req.video_number}] All images done in {time.time()-t1:.1f}s")

        # ── Step 2: Download audio from Google Drive ───────────────────────────
        logger.info(f"[{req.video_number}] Downloading audio …")
        audio_url = req.audio_url

        # Normalise Google Drive URLs to direct download
        if "drive.google.com" in audio_url:
            if "/file/d/" in audio_url:
                file_id = audio_url.split("/file/d/")[1].split("/")[0]
            elif "id=" in audio_url:
                file_id = audio_url.split("id=")[1].split("&")[0]
            else:
                file_id = None
            if file_id:
                audio_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"

        session = requests.Session()
        r = session.get(audio_url, timeout=60, allow_redirects=True)
        # Handle Google Drive "virus scan warning" redirect
        if b"download_warning" in r.content[:2000] or b"Google Drive" in r.content[:200]:
            for k, v in r.cookies.items():
                if "download_warning" in k:
                    r = session.get(f"{audio_url}&confirm={v}", timeout=60)
                    break

        audio_path = os.path.join(work_dir, "audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(r.content)
        logger.info(f"[{req.video_number}] Audio saved ({len(r.content)//1024} KB)")

        # ── Step 3: Build FFmpeg command ────────────────────────────────────────
        n = len(image_paths)
        d = req.duration_per_image      # seconds per image (input loop)
        fade = 0.5                       # crossfade duration in seconds
        output_path = os.path.join(work_dir, f"SaarVaaniLab_{req.video_number}.mp4")

        # Each image is looped for (d + 1) seconds to give xfade enough material
        input_args = []
        for img in image_paths:
            input_args += ["-loop", "1", "-t", str(d + 1.0), "-i", img]
        input_args += ["-i", audio_path]

        # Scale + pad each image to exactly 1080×1920 (black bars if needed)
        scale_parts = []
        for i in range(n):
            scale_parts.append(
                f"[{i}:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,fps=25[s{i}]"
            )

        # Xfade transitions chained together
        # Offset formula: transition i starts at (i+1)*(d - fade)
        xfade_parts = []
        prev = "s0"
        for i in range(1, n):
            offset = round(i * (d - fade), 3)
            out = f"xf{i}" if i < n - 1 else "outv"
            xfade_parts.append(
                f"[{prev}][s{i}]xfade=transition=fade:duration={fade}:offset={offset}[{out}]"
            )
            prev = f"xf{i}"

        if n == 1:
            filter_complex = scale_parts[0] + ";[s0]copy[outv]"
        else:
            filter_complex = ";".join(scale_parts) + ";" + ";".join(xfade_parts)

        cmd = [
            "ffmpeg", "-y",
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", f"{n}:a",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "28",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",                 # stop when audio ends
            "-movflags", "+faststart",   # web-optimised
            output_path
        ]

        logger.info(f"[{req.video_number}] Running FFmpeg …")
        t2 = time.time()
        result = subprocess.run(cmd, capture_output=True, timeout=240)

        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            logger.error(f"[{req.video_number}] FFmpeg FAILED:\n{err}")
            raise HTTPException(status_code=500, detail=f"FFmpeg error: {err[-800:]}")

        logger.info(f"[{req.video_number}] FFmpeg done in {time.time()-t2:.1f}s")

        # ── Step 4: Return the video ───────────────────────────────────────────
        with open(output_path, "rb") as f:
            video_bytes = f.read()

        logger.info(f"[{req.video_number}] Total: {time.time()-t0:.1f}s — {len(video_bytes)//1024} KB")

        return Response(
            content=video_bytes,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f'attachment; filename="SaarVaaniLab_{req.video_number}.mp4"',
                "Content-Length": str(len(video_bytes)),
            },
        )

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info(f"[{req.video_number}] Temp dir cleaned up")
