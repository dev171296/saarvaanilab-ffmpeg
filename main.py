from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
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
import random
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SaarVaaniLab FFmpeg Service")


class VideoRequest(BaseModel):
    image_prompts: List[str]       # 7 image prompts (raw text, service encodes)
    audio_url: str                  # Google Drive webContentLink for the MP3
    video_number: str               # e.g. "001"
    duration_per_image: float = 7.5   # seconds per image


@app.get("/")
def root():
    return {"status": "alive", "service": "SaarVaaniLab FFmpeg", "version": "1.2"}


@app.get("/ping")
def ping():
    return {"pong": True}


def _download_single_image(args):
    """Download one image from Pollinations — called inside a thread pool."""
    i, prompt, work_dir = args
    # Tiny random jitter; max_workers=3 already limits concurrency
    time.sleep(random.uniform(0, 0.5))
    seed = 1001 + i
    encoded = urllib.parse.quote(prompt)
    # 720×1280 saves ~56% memory vs 1080×1920; still fine for Reels/Shorts
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width=720&height=1280&model=flux&seed={seed}"
    )
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)   # 10s, 20s, 30s
                logger.info(f"  Image {i+1} rate-limited (429), retrying in {wait}s …")
                time.sleep(wait)
                continue
            r.raise_for_status()
            img_path = os.path.join(work_dir, f"img_{i:02d}.jpg")
            with open(img_path, "wb") as f:
                f.write(r.content)
            logger.info(f"  Image {i+1} ✓ ({len(r.content)//1024} KB)")
            return i, img_path
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"Image {i+1} failed after 4 attempts: {e}")
            time.sleep(5)


@app.post("/assemble")
async def assemble_video(req: VideoRequest, background_tasks: BackgroundTasks):
    work_dir = tempfile.mkdtemp()
    t0 = time.time()
    logger.info(f"[{req.video_number}] Starting assembly in {work_dir}")

    try:
        # ── Step 1: Download all images IN PARALLEL ────────────────────────────
        logger.info(f"[{req.video_number}] Downloading {len(req.image_prompts)} images in parallel …")
        t1 = time.time()

        args_list = [(i, p, work_dir) for i, p in enumerate(req.image_prompts)]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_download_single_image, a): a[0] for a in args_list}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except RuntimeError as e:
                    raise HTTPException(status_code=502, detail=str(e))

        results.sort(key=lambda x: x[0])
        image_paths = [path for _, path in results]
        logger.info(f"[{req.video_number}] All images done in {time.time()-t1:.1f}s")

        # ── Step 2: Download audio from Google Drive ───────────────────────────
        logger.info(f"[{req.video_number}] Downloading audio …")
        audio_url = req.audio_url

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
        d = req.duration_per_image
        fade = 0.5
        output_path = os.path.join(work_dir, f"SaarVaaniLab_{req.video_number}.mp4")

        input_args = []
        for img in image_paths:
            input_args += ["-loop", "1", "-t", str(d + 1.0), "-i", img]
        input_args += ["-i", audio_path]

        # Scale + pad to 720×1280 (9:16)
        scale_parts = []
        for i in range(n):
            scale_parts.append(
                f"[{i}:v]scale=720:1280:force_original_aspect_ratio=decrease,"
                f"pad=720:1280:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"setsar=1,fps=25[s{i}]"
            )

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
            "-threads", "2",            # cap thread count → lower peak RAM
            *input_args,
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", f"{n}:a",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "28",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
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
        logger.info(f"[{req.video_number}] Total: {time.time()-t0:.1f}s")

        # ── Step 4: Stream video — no f.read() into RAM ───────────────────────
        # BackgroundTask deletes temp dir AFTER response is fully sent
        background_tasks.add_task(shutil.rmtree, work_dir, True)
        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename=f"SaarVaaniLab_{req.video_number}.mp4",
        )

    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.error(f"[{req.video_number}] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
