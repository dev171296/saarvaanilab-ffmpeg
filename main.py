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
import edge_tts
from PIL import Image, ImageDraw, ImageFont

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


def _render_hook_png(hook_text: str, work_dir: str):
    """Render the centered hook text to a transparent 720x1280 PNG using Pillow +
    RAQM shaping. FFmpeg's drawtext lacks HarfBuzz in this build, so it mis-places
    Devanagari matras (e.g. फिर -> फरि). Pillow+RAQM reorders them correctly.
    Returns the PNG path, or None if there's no hook / the font isn't ready."""
    if not hook_text or not _font_ready():
        logger.warning("Hook overlay skipped — font not ready or hook_text empty")
        return None

    W, H = 720, 1280
    lines = _wrap_hook(hook_text).split("\n")
    try:
        font = ImageFont.truetype(FONT_PATH, 64, layout_engine=ImageFont.Layout.RAQM)
    except Exception:
        # Fallback: default layout (still shapes if Pillow's default is RAQM-capable)
        font = ImageFont.truetype(FONT_PATH, 64)

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    line_spacing = 12
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    widths = [draw.textbbox((0, 0), ln, font=font)[2] for ln in lines]
    block_w = max(widths) if widths else 0
    block_h = line_h * len(lines) + line_spacing * (len(lines) - 1)

    pad = 22
    draw.rectangle(
        [(W - block_w) // 2 - pad, (H - block_h) // 2 - pad,
         (W + block_w) // 2 + pad, (H + block_h) // 2 + pad],
        fill=(0, 0, 0, 153),  # black @ 0.60
    )

    y = (H - block_h) // 2
    for ln in lines:
        w = draw.textbbox((0, 0), ln, font=font)[2]
        x = (W - w) // 2
        draw.text((x + 3, y + 3), ln, font=font, fill=(0, 0, 0, 230))  # shadow
        draw.text((x, y), ln, font=font, fill=(255, 255, 255, 255))    # text
        y += line_h + line_spacing

    out = os.path.join(work_dir, "hook_overlay.png")
    img.save(out)
    img.close()
    logger.info(f"Hook PNG rendered ({len(lines)} lines) -> {out}")
    return out


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
        "version": "2.5",
        "font_ready": _font_ready(),
    }


@app.get("/ping")
def ping():
    return {"pong": True}


# ── Text-to-Speech (free, edge-tts / Microsoft neural voices) ────────────────────
# Replaces the old Azure TTS. No API key, no cost. hi-IN-MadhurNeural = male Hindi
# voice (closest free twin of Azure's Kunal). hi-IN-SwaraNeural = female Hindi voice.

class TTSRequest(BaseModel):
    text: str = ""       # optional: full text directly
    hook: str = ""       # optional: hook line (spoken first)
    script: str = ""     # optional: full script (spoken after hook)
    voice: str = "hi-IN-MadhurNeural"
    rate: str = "+0%"    # e.g. "-10%" slower, "+10%" faster
    pitch: str = "+0Hz"  # e.g. "-2Hz" lower, "+2Hz" higher
    url_encoded: bool = True   # Make sends fields URL-encoded to keep JSON valid


@app.post("/tts")
async def tts(req: TTSRequest, background_tasks: BackgroundTasks):
    def _dec(s: str) -> str:
        return urllib.parse.unquote(s or "") if req.url_encoded else (s or "")

    if req.text:
        text = _dec(req.text).strip()
    else:
        hook = _dec(req.hook).strip()
        script = _dec(req.script).strip()
        if hook and script:
            text = f"{hook}। {script}"
        else:
            text = hook or script
        text = text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="No text provided for TTS")

    logger.info(f"[TTS] voice={req.voice} rate={req.rate} pitch={req.pitch} chars={len(text)}")
    work_dir = tempfile.mkdtemp()
    out_path = os.path.join(work_dir, "voice.mp3")

    try:
        communicate = edge_tts.Communicate(
            text, req.voice, rate=req.rate, pitch=req.pitch
        )
        await communicate.save(out_path)

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            raise HTTPException(status_code=502, detail="TTS produced empty audio")

        logger.info(f"[TTS] Audio ready ({os.path.getsize(out_path)//1024} KB)")
        background_tasks.add_task(shutil.rmtree, work_dir, True)
        return FileResponse(out_path, media_type="audio/mpeg", filename="voice.mp3")

    except HTTPException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.error(f"[TTS] Failed: {e}")
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")


def _download_single_image(args):
    i, prompt, work_dir = args
    time.sleep(random.uniform(0, 0.5))
    seed = 42000 + i

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
            if len(r.content) < 5000:
                logger.warning(f"  Image {i+1} empty ({len(r.content)} bytes) — retry {attempt+1}/4")
                time.sleep(10 * (attempt + 1))
                continue
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
        # Scene 1 gets the hook text as a properly-shaped PNG overlay; Scenes 2-7
        # get branding only.
        hook_png = _render_hook_png(hook_text, work_dir)
        vf_plain = _build_vf_plain()
        logger.info(f"[{req.video_number}] Overlay ready (hook={'yes' if hook_png else 'no'})")

        clip_paths = []
        t2 = time.time()
        for idx, img_path in enumerate(image_paths):
            clip_path = os.path.join(work_dir, f"clip_{idx:02d}.ts")
            if idx == 0 and hook_png:
                # scale/pad/brand the image, then overlay the shaped hook PNG on top
                cmd_clip = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-t", str(d),
                    "-i", img_path,
                    "-i", hook_png,
                    "-filter_complex",
                    f"[0:v]{vf_plain}[bg];[bg][1:v]overlay=0:0[out]",
                    "-map", "[out]",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "28",
                    "-threads", "1",
                    clip_path
                ]
            else:
                cmd_clip = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-t", str(d),
                    "-i", img_path,
                    "-vf", vf_plain,
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
