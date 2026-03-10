import base64
import gc
import io
import logging
import os
import shutil
import sys

import torch
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils import download_files_from_urls, rp_cleanup

import whisperx
from whisperx.alignment import DEFAULT_ALIGN_MODELS_TORCH, DEFAULT_ALIGN_MODELS_HF
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.utils import WriteSRT, WriteVTT

from rp_schema import INPUT_VALIDATIONS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("rp_handler")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
WHISPER_ARCH = os.environ.get("WHISPER_MODEL", "/models/faster-whisper-large-v3")
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

# ---------------------------------------------------------------------------
# Model warm-up: load Whisper once at container start
# ---------------------------------------------------------------------------
logger.info(f"Loading Whisper model: {WHISPER_ARCH}  device={DEVICE}  compute={COMPUTE_TYPE}")
ASR_MODEL = whisperx.load_model(
    WHISPER_ARCH,
    DEVICE,
    compute_type=COMPUTE_TYPE,
    asr_options={"temperatures": [0]},
)
logger.info("Whisper model loaded and ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_audio(audio_input: str, job_id: str) -> str:
    """Download from URL or decode base64 to a temp file. Returns file path."""
    if "://" in audio_input:
        paths = download_files_from_urls(job_id, [audio_input])
        return paths[0]

    if "," in audio_input:
        audio_input = audio_input.split(",", 1)[1]
    audio_bytes = base64.b64decode(audio_input)

    job_dir = f"/tmp/jobs/{job_id}"
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, "audio_input")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    logger.info(f"Decoded base64 audio -> {path} ({len(audio_bytes)} bytes)")
    return path


def _cleanup(job_id: str):
    job_dir = f"/tmp/jobs/{job_id}"
    if os.path.isdir(job_dir):
        shutil.rmtree(job_dir, ignore_errors=True)
    try:
        rp_cleanup.clean(["input_objects"])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(job):
    job_id = job["id"]
    job_input = job["input"]

    validated = validate(job_input, INPUT_VALIDATIONS)
    if "errors" in validated:
        return {"error": validated["errors"]}

    audio_input = job_input["audio"]
    language = job_input.get("language")
    batch_size = job_input.get("batch_size", 16)
    align_output = job_input.get("align_output", False)
    diarization = job_input.get("diarization", False)
    min_speakers = job_input.get("min_speakers")
    max_speakers = job_input.get("max_speakers")
    initial_prompt = job_input.get("initial_prompt")
    temperature = job_input.get("temperature", 0)
    vad_onset = job_input.get("vad_onset", 0.500)
    vad_offset = job_input.get("vad_offset", 0.363)
    output_format = job_input.get("output_format", ["json"])
    if isinstance(output_format, str):
        output_format = [output_format]
    output_format = [f.lower() for f in output_format]

    try:
        audio_path = _resolve_audio(audio_input, job_id)
        audio = whisperx.load_audio(audio_path)
        logger.info(f"Audio loaded: {len(audio) / 16000:.1f}s")

        result = ASR_MODEL.transcribe(audio, batch_size=batch_size)
        detected_language = result["language"]
        logger.info(f"Transcription done. Language: {detected_language}, segments: {len(result['segments'])}")

        if align_output:
            supported = (
                detected_language in DEFAULT_ALIGN_MODELS_TORCH
                or detected_language in DEFAULT_ALIGN_MODELS_HF
            )
            if supported:
                align_model, align_metadata = whisperx.load_align_model(
                    detected_language, DEVICE
                )
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    DEVICE,
                    return_char_alignments=False,
                )
                del align_model
                gc.collect()
                torch.cuda.empty_cache()
                logger.info("Alignment done.")
            else:
                logger.warning(
                    f"No alignment model for language '{detected_language}', skipping alignment."
                )

        if diarization:
            token = HF_TOKEN or None
            diarize_model = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-community-1",
                token=token,
                device=DEVICE,
            )
            diarize_segments = diarize_model(
                audio, min_speakers=min_speakers, max_speakers=max_speakers
            )
            result = assign_word_speakers(diarize_segments, result)
            del diarize_model
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Diarization done.")

        output = {"detected_language": detected_language}

        subtitle_options = {
            "max_line_width": None,
            "max_line_count": None,
            "highlight_words": False,
        }

        for fmt in output_format:
            if fmt == "json":
                output["json"] = {"segments": result["segments"]}
            elif fmt == "srt":
                sio = io.StringIO()
                WriteSRT(".").write_result(result, file=sio, options=subtitle_options)
                output["srt"] = sio.getvalue()
            elif fmt == "vtt":
                sio = io.StringIO()
                WriteVTT(".").write_result(result, file=sio, options=subtitle_options)
                output["vtt"] = sio.getvalue()
            else:
                logger.warning(f"Unknown output format '{fmt}', skipping.")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        output = {"error": str(e)}

    finally:
        _cleanup(job_id)

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
