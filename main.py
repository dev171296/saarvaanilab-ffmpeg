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

# ── Font ────────────────────────────────────────────────────────────────────────
FONT_PATH = "/tmp/NotoSansDevanagari-Bold.ttf"
FONT_URL = (
    "https://fonts.gstatic.com/s/notosansdevanagari/v30/"
    "TuGoUUFzXI5FBtUq5a8bjKYTZjtRU6Sgv3NaV_SNmI0b8QQCQmHn6B2OHjbL_08AlZMiy-A.ttf"
)
HOOK_TEXT_FILE = "/tmp/saarvaani_hook.txt"


def _ensure_font() -> bool:
    """Download Noto Devanagari Bold to /tmp at startup. Render has internet at runtime."""
    if os.path.exists(FONT_PATH) and os.path.getsize(FONT_PATH) > 50_000:
        logger.info(f"Font ready ✓  ({os.path.getsize(FONT_PATH)//1024} KB)")
        return True
    try:
        logger.info("Downloading Noto Devanagari font …")
        r = requests.get(FONT_URL, timeout=30)
        r.raise_for_status()
        with open(FONT_PATH, "wb") as f:
            f.write(r.content)
        logger.info(f"Font downloaded ✓  ({len(r.content)//1024} KB) → {FONT_PATH}")
        return True
    except Exception as e:
        logger.error(f"Font download FAILED: {e}  — overlays will be skipped")
        return False


@app.on_event("startup")
def startup_event():
    _ensure_font()


def _font_ready() -> bool:
    return os.path.exists(FONT_PATH) and os.path.getsize(FONT_PATH) > 50_000


def _wrap_hook(text: str, max_chars: int = 14) -> str:
    """Word-wrap for FFmpeg textfile (supports Hindi + Latin mix)."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        candidate = (cur + " " + w).strip()
        if len(candidate) > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = candidate
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _build_vf_hook(hook_text: str) -> str:
    """VF filter for Scene 1 — base scale + hook text overlay + branding."""
    base = (
        "scale=720:1280:force_original_aspect_ratio=decrease,"
        "pad=720:1280:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setsar=1,fps=25"
    )
    if not hook_text or not _font_ready():
        logger.warning("Overlays skipped — font not ready or hook_text empty")
        return base

    wrapped = _wrap_hook(hook_text)
    with open(HOOK_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(wrapped)
    logger.info(f"Hook text written ({len(wrapped)} chars, {wrapped.count(chr(10))+1} lines)")

    # Cinematic centered hook — large, dominant, screen-filling
    hook_dt = (
        f"drawtext=fontfile={FONT_PATH}"
        f":textfile={HOOK_TEXT_FILE}"
        f":fontcolor=white:fontsize=64"
        f":x=(w-text_w)/2:y=(h-text_h)/2"
        f":shadowcolor=black@0.9:shadowx=3:shadowy=3"
        f":box=1:boxcolor=black@0.60:boxborderw=22"
        f":line_spacing=12"
    )
    brand_dt = (
        f"drawtext=fontfile={FONT_PATH}"
        f":text=SaarVaaniLab"
        f":fontcolor=yellow:fontsize=24"
        f":x=w-text_w-15:y=15"
        f":box=1:boxcolor=black@0.45:boxborderw=8"
    )
    return f"{base},{hook_dt},{brand_dt}"


def _build_vf_plain() -> str:
    """VF filter for Scenes 2-7 — base scale + branding only, no hook text."""
    base = (
        "scale=720:1280:force_original_aspect_ratio=decrease,"
        "pad=720:1280:(ow-iw)/2:(oh-ih)/2:color=black,"
        "setsar=1,fps=25"
    )
    brand_dt = (
        f"drawtext=fontfile={FONT_PATH}"
        f":text=SaarVaaniLab"
        f":fontcolor=yellow:fontsize=24"
        f":x=w-text_w-15:y=15"
        f":box=1:boxcolor=black@0.45:boxborderw=8"
    )
    if not _font_ready():
        return base
    return f"{base},{brand_dt}"


# ── Request model ───────────────────────────────────────────────────────────────

class VideoRequest(BaseModel):
    image_prompts: List[str]
    audio_url: str
    video_number: str
    hook_text: str = ""
    duration_per_image: float = 7.5


@app.get("/")
def root():
    return {
        "status": "alive",
        "service": "SaarVaaniLab FFmpeg",
        "version": "2.2",
        "font_ready": _font_ready(),
    }


@app.get("/ping")
def ping():
    return {"pong": True}


def _download_single_image(args):
    i, prompt, work_dir = args
    time.sleep(random.uniform(0, 0.5))
    seed = 1001 + i

    # Minimal cinematic style prefix — no gender or character assumptions
    style_prefix = (
        "photorealistic cinematic 4K ultra-detailed, "
        "ancient India Ramayana era, "
        "traditional period-accurate setting and clothing, "
        "dramatic cinematic lighting, "
    )

    encoded = urllib.parse.quote(style_prefix + prompt)
    negative = urllib.parse.quote(
        "cleavage,deep neck,deep neckline,deep V-neck,low cut neckline,off shoulder,"
        "revealing clothes,bare skin,bare chest,bare shoulders,sexual,nsfw,nude,semi-nude,"
        "inappropriate,modern clothing,western outfit,bikini,lingerie,exposed midriff,"
        "tight clothes,skimpy,provocative pose,ugly,deformed,blurry,watermark"
    )
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width=720&height=1280&model=flux&seed={seed}&negative={negative}&nologo=true"
    )
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 429:
                wait = 10 * (attempt + 1)
                logger.info(f"  Image {i+1} 429 — retry in {wait}s")
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
    # Decode URL-encoded hook text (Make.com encodeURL() encodes Hindi to ASCII-safe)
    hook_text = urllib.parse.unquote(req.hook_text)
    logger.info(f"[{req.video_number}] v2.2 — hook_text={repr(hook_text[:40])} font={_font_ready()}")

    try:
        # ── Step 1: Download images in parallel ────────────────────────────────
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
        image_paths = [p for _, p in results]
        logger.info(f"[{req.video_number}] Images done in {time.time()-t1:.1f}s")

        # ── Step 2: Download audio ─────────────────────────────────────────────
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

        # ── Step 3: Measure audio duration → set per-image duration ───────────
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10
        )
        try:
            audio_duration = float(probe.stdout.strip())
        except Exception:
            audio_duration = None

        n = len(image_paths)
        if audio_duration and audio_duration > 0:
            d = audio_duration / n
            logger.info(f"[{req.video_number}] Audio duration={audio_duration:.1f}s → {d:.2f}s/clip")
        else:
            d = req.duration_per_image
            logger.warning(f"[{req.video_number}] ffprobe failed, using default {d}s/clip")

        # ── Step 4: Encode each image to clip ─────────────────────────────────
        # Scene 1 gets hook text overlay; Scenes 2-7 get branding only.
        vf_hook = _build_vf_hook(hook_text)
        vf_plain = _build_vf_plain()
        logger.info(f"[{req.video_number}] VF filters ready (hook={'yes' if _font_ready() and hook_text else 'no'})")

        clip_paths = []
        t2 = time.time()
        for idx, img_path in enumerate(image_paths):
            clip_path = os.path.join(work_dir, f"clip_{idx:02d}.ts")
            vf = vf_hook if idx == 0 else vf_plain
            cmd_clip = [
                "ffmpeg", "-y",
                "-loop", "1", "-t", str(d),
                "-i", img_path,
                "-vf", vf,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                "-threads", "1",
                clip_path
            ]
            res = subprocess.run(cmd_clip, capture_output=True, timeout=60)
            if res.returncode != 0:
                err = res.stderr.decode(errors="replace")
                logger.error(f"Clip {idx} stderr:\n{err[-600:]}")
                raise HTTPException(status_code=500, detail=f"Clip {idx} error: {err[-400:]}")
            clip_paths.append(clip_path)
            logger.info(f"[{req.video_number}] Clip {idx+1}/{n} encoded")
        logger.info(f"[{req.video_number}] All clips in {time.time()-t2:.1f}s")

        # ── Step 5: Concat + audio (video duration matches audio exactly) ──────
        concat_file = os.path.join(work_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        output_path = os.path.join(work_dir, f"SaarVaaniLab_{req.video_number}.mp4")
        cmd_final = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "128k",
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
