from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
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
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SaarVaaniLab FFmpeg Service")

# ── Font setup ─────────────────────────────────────────────────────────────────
# fonts-noto-core is installed in Dockerfile; path on Debian/Ubuntu:
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari[wdth,wght].ttf",
    "/usr/share/fonts/opentype/noto/NotoSansDevanagari-Bold.otf",
    "/usr/share/fonts/noto/NotoSansDevanagari-Bold.ttf",
]
_FONT_PATH: Optional[str] = None
_FONT_CACHE: dict = {}

def _resolve_font() -> Optional[str]:
    global _FONT_PATH
    if _FONT_PATH:
        return _FONT_PATH
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            logger.info(f"Font found: {p}")
            _FONT_PATH = p
            return p
    # Fallback: download once at runtime
    url = (
        "https://fonts.gstatic.com/s/notosansdevanagari/v30/"
        "TuGoUUFzXI5FBtUq5a8bjKYTZjtRU6Sgv3NaV_SNmI0b8QQCQmHn6B2OHjbL_08AlZMiy-A.ttf"
    )
    dest = "/tmp/NotoSansDevanagari-Bold.ttf"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        logger.info(f"Font downloaded ({len(r.content)//1024} KB) → {dest}")
        _FONT_PATH = dest
        return dest
    except Exception as e:
        logger.warning(f"Font unavailable: {e}  — overlays will be skipped")
        return None

def _get_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    path = _resolve_font()
    if not path:
        return None
    try:
        f = ImageFont.truetype(path, size)
        _FONT_CACHE[size] = f
        return f
    except Exception as e:
        logger.warning(f"Could not load font at size {size}: {e}")
        return None

def _wrap_text(draw: ImageDraw.Draw, text: str, font, max_w: int) -> List[str]:
    """Split text into lines that fit within max_w pixels."""
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip()
        try:
            tw = draw.textbbox((0, 0), candidate, font=font)[2]
        except Exception:
            tw = len(candidate) * 20
        if tw > max_w and cur:
            lines.append(cur)
            cur = w
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return lines

def _add_text_overlays(img_path: str, hook_text: str) -> None:
    """
    Burn hook text (lower-third) + SaarVaaniLab branding (top-right) onto
    the image in-place.  Fails silently so a missing font never kills the video.
    """
    try:
        img = Image.open(img_path).convert("RGBA")
        W, H = img.size                     # 720 × 1280
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ov = ImageDraw.Draw(overlay)
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

        # ── Hook text — lower-third dark strip ───────────────────────────────
        if hook_text:
            hook_font = _get_font(42)
            if hook_font:
                lines = _wrap_text(dummy, hook_text, hook_font, W - 48)
                LINE_H = 56
                total_h = len(lines) * LINE_H + 20
                sy = H - total_h - 90          # top of the strip
                # semi-transparent dark background
                ov.rectangle([(0, sy - 10), (W, sy + total_h + 4)],
                             fill=(0, 0, 0, 175))
                for j, line in enumerate(lines):
                    try:
                        tw = dummy.textbbox((0, 0), line, font=hook_font)[2]
                    except Exception:
                        tw = W // 2
                    tx = (W - tw) // 2
                    ty = sy + j * LINE_H
                    # drop shadow
                    ov.text((tx + 2, ty + 2), line, font=hook_font, fill=(0, 0, 0, 220))
                    # white text
                    ov.text((tx, ty), line, font=hook_font, fill=(255, 255, 255, 255))

        # ── SaarVaaniLab branding — top-right ────────────────────────────────
        brand_font = _get_font(26)
        if brand_font:
            brand = "SaarVaaniLab"
            try:
                bw = dummy.textbbox((0, 0), brand, font=brand_font)[2]
                bh = dummy.textbbox((0, 0), brand, font=brand_font)[3]
            except Exception:
                bw, bh = 160, 28
            bx = W - bw - 16
            by = 16
            # pill background
            ov.rectangle([(bx - 8, by - 4), (bx + bw + 8, by + bh + 4)],
                         fill=(0, 0, 0, 155))
            # shadow + golden text
            ov.text((bx + 1, by + 1), brand, font=brand_font, fill=(0, 0, 0, 200))
            ov.text((bx, by), brand, font=brand_font, fill=(255, 210, 60, 255))

        # composite and save
        Image.alpha_composite(img, overlay).convert("RGB").save(
            img_path, "JPEG", quality=92
        )
        logger.info("Text overlays applied ✓")

    except Exception as e:
        logger.warning(f"Text overlay skipped: {e}")


# ── Request model ──────────────────────────────────────────────────────────────

class VideoRequest(BaseModel):
    image_prompts: List[str]          # 7 image prompts (raw text, service encodes)
    audio_url: str                    # Google Drive webContentLink for the MP3
    video_number: str                 # e.g. "001"
    hook_text: str = ""               # Column D from Sheet — displayed throughout video
    duration_per_image: float = 7.5  # seconds per image


@app.get("/")
def root():
    return {"status": "alive", "service": "SaarVaaniLab FFmpeg", "version": "1.4"}


@app.get("/ping")
def ping():
    return {"pong": True}


def _download_single_image(args):
    """Download one image from Pollinations — called inside a thread pool."""
    i, prompt, work_dir = args
    time.sleep(random.uniform(0, 0.5))
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
                wait = 10 * (attempt + 1)
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
    logger.info(f"[{req.video_number}] Starting assembly v1.4 in {work_dir}")

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

        # ── Step 1b: Burn text overlays onto every image ───────────────────────
        # hook_text stays visible throughout the entire video; adds branding too
        if req.hook_text:
            logger.info(f"[{req.video_number}] Applying text overlays …")
            for img_path in image_paths:
                _add_text_overlays(img_path, req.hook_text)

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
        # One FFmpeg call per image → peak RAM ~30-50 MB each (vs 400-600 MB for
        # a single filter_complex with 7 inputs + 6 chained xfade filters)
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
                "-preset", "ultrafast",
                "-crf", "28",
                "-threads", "1",
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

        # ── Step 5: Concat clips + audio (-c:v copy = no re-encode, <50 MB) ───
        output_path = os.path.join(work_dir, f"SaarVaaniLab_{req.video_number}.mp4")
        cmd_final = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-i", audio_path,
            "-c:v", "copy",
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

        # ── Step 6: Stream video without loading into RAM ─────────────────────
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
