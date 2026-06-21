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
    return {"status": "alive", "service": "SaarVaaniLab FFmpeg", "version": "1.3"}


@app.get("/ping")
def ping():
    return {"pong": True}


def _download_single_image(args):
    """Download one image from Pollinations — called inside a thread pool."""
    i, prompt, work_dir = args
    time.sleep(random.uniform(0, 0.5))   # tiny jitter to spread requests
    seed = 1001 + i
    encoded = urllib.parse.quote(prompt)
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
    logger.info(f"[{req.video_number}] Starting assembly v1.3 in {work_dir}")

    try:
        # ── Step 1: Download all images IN PARALLEL ────────────────────────────
        logger.info(f"[{req.video_number}] Downloading {len(req.image_prompts)} images …")
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
        logger.info(f"[{req.video_number}] Images done in {time.time()-t1:.1f}s")

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

        # ── Step 3: Encode each image to a short .ts clip (1 FFmpeg per image) ──
        # v1.2 used 7 simultaneous inputs + 6 chained xfade → 400-600MB filter graph → OOM
        # v1.3: one FFmpeg per image, ultrafast, 1 thread → peak ~30-50MB per call
        n = len(image_paths)
        d = req.duration_per_image
        clip_paths = []
        t2 = time.time()
        for idx, img_path in enumerate(image_paths):
            clip_path = os.path.join(work_dir, f"clip_{idx:02d}.ts")
            cmd_clip = [
                "ffmpeg", "-y",
                "-loop", "1", "-t", str(d),
                "-i", img_path,
                "-vf", (
                    "scale=720:1280:force_original_aspect_ratio=decrease,"
                    "pad=720:1280:(ow-iw)/2:(oh-ih)/2:color=black,"
                    "setsar=1,fps=25"
                ),
                "-c:v", "libx264",
                "-preset", "ultrafast",   # fastest encode, fine for Reels
                "-crf", "28",
                "-threads", "1",           # 1 thread per clip → low RAM
                clip_path
            ]
            res = subprocess.run(cmd_clip, capture_output=True, timeout=60)
            if res.returncode != 0:
                err = res.stderr.decode(errors="replace")
                raise HTTPException(status_code=500, detail=f"Clip {idx} error: {err[-600:]}")
            clip_paths.append(clip_path)
            logger.info(f"[{req.video_number}] Clip {idx+1}/{n} encoded")
        logger.info(f"[{req.video_number}] All clips encoded in {time.time()-t2:.1f}s")

        # ── Step 4: Write concat list ──────────────────────────────────────────
        concat_file = os.path.join(work_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        # ── Step 5: Concat clips + audio (-c:v copy = no re-encode, ~2s, <50MB) ─
        output_path = os.path.join(work_dir, f"SaarVaaniLab_{req.video_number}.mp4")
        cmd_final = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-i", audio_path,
            "-c:v", "copy",           # stream-copy: zero re-encoding RAM
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            output_path
        ]
        t3 = time.time()
        res_final = subprocess.run(cmd_final, capture_output=True, timeout=120)
        if res_final.returncode != 0:
            err = res_final.stderr.decode(errors="replace")
            logger.error(f"[{req.video_number}] Concat FAILED:\n{err}")
            raise HTTPException(status_code=500, detail=f"Concat error: {err[-800:]}")
        logger.info(f"[{req.video_number}] Concat done in {time.time()-t3:.1f}s")
        logger.info(f"[{req.video_number}] Total: {time.time()-t0:.1f}s")

        # ── Step 6: Stream video — FileResponse sends without loading into RAM ─
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
