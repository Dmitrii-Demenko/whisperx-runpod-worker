# whisperx-runpod-worker

![Platform](https://img.shields.io/badge/platform-RunPod%20Serverless-7c3aed)
![CUDA](https://img.shields.io/badge/CUDA-12.8-76b900)
![Python](https://img.shields.io/badge/python-3.12-3776ab)
![License](https://img.shields.io/badge/license-MIT-22c55e)

RunPod Serverless worker for [WhisperX](https://github.com/m-bain/whisperX) — fast speech transcription with word-level timestamps and speaker diarization.

## Features

- **Transcription** via `faster-whisper large-v3` — 30+ languages, automatic language detection
- **Word-level timestamps** via forced alignment (`wav2vec2`)
- **Speaker diarization** via `pyannote/speaker-diarization-community-1`
- **Flexible audio input** — public URL or base64-encoded file
- **Supported formats** — MP3, WAV, FLAC, AAC, OGG, MP4, MKV, MOV, WebM, and any format supported by ffmpeg
- **Zero cold-start** — all models (~15 GB) are baked into the image at build time

## Prerequisites

- Docker with [BuildKit](https://docs.docker.com/build/buildkit/) enabled (Docker Desktop enables it by default)
- A [Hugging Face account](https://huggingface.co/join) with a [read token](https://huggingface.co/settings/tokens)
- Accepted terms of use for the following gated models (click → **Agree**):
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- ~25 GB of free disk space for the build (final image is ~15 GB)

> **Note:** The image targets `linux/amd64` with NVIDIA CUDA. It cannot run locally on macOS — it is designed for deployment on RunPod GPU instances.

## Build

The build is split into two images to keep code iteration fast:

- **`Dockerfile.base`** — heavy base: CUDA, Python, PyTorch, all models (~15 GB). Build once, rarely rebuild.
- **`Dockerfile`** — thin app layer: just copies `rp_handler.py` + `rp_schema.py` on top of the base. Builds in seconds.

### Step 1 — Build and push the base image (once, or when dependencies change)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx   # your Hugging Face read token

docker build \
  --platform linux/amd64 \
  --secret id=hf_token,env=HF_TOKEN \
  -t your-dockerhub-username/whisperx-base:latest \
  -f Dockerfile.base \
  . && \
docker push your-dockerhub-username/whisperx-base:latest
```

This takes **10–20 minutes** and downloads:
- `Systran/faster-whisper-large-v3`
- `pyannote/speaker-diarization-community-1` + `pyannote/segmentation-3.0`
- `jonatasgrosman/wav2vec2-large-xlsr-53-russian` (alignment model pre-cached for Russian)
- NLTK `punkt_tab` tokenizer

### Step 2 — Build and push the worker image (on every code change)

```bash
docker build \
  --platform linux/amd64 \
  -t your-dockerhub-username/whisperx-worker:latest \
  . && \
docker push your-dockerhub-username/whisperx-worker:latest
```

This takes **~5 seconds** and pushes only the handler layer (~50 KB).

## Push & Deploy

```bash
# Push to registry
docker push your-dockerhub-username/whisperx-worker:latest
```

**Deploy on RunPod:**

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless) → **New Endpoint**
2. Select **Custom Image** → enter your Docker image URL
3. Recommended GPU: **RTX 4090** or **A100 40GB**
4. Add environment variable: `HF_TOKEN=<your-huggingface-token>` (required for diarization at runtime)
5. Save and wait for the worker to become active

## API

### Endpoints

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run        # async job
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync    # sync (short audio only, <30s)
GET  https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}
```

### Input schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `audio` | `string` | **required** | Public URL or base64-encoded audio/video file |
| `language` | `string` | `null` | Language code (`ru`, `en`, `de`, ...). `null` = auto-detect |
| `batch_size` | `int` | `16` | Whisper batch size. Reduce if running out of VRAM |
| `align_output` | `bool` | `false` | Add word-level timestamps to each segment |
| `diarization` | `bool` | `false` | Identify and label individual speakers |
| `min_speakers` | `int` | `null` | Minimum speaker count hint (diarization only) |
| `max_speakers` | `int` | `null` | Maximum speaker count hint (diarization only) |
| `initial_prompt` | `string` | `null` | Hint text passed to Whisper (improves domain-specific accuracy) |
| `temperature` | `float` | `0` | Sampling temperature (0 = greedy, deterministic) |
| `vad_onset` | `float` | `0.500` | VAD onset threshold |
| `vad_offset` | `float` | `0.363` | VAD offset threshold |
| `output_format` | `list[string]` | `["json"]` | Output formats to return. Any combination of `"json"`, `"srt"`, `"vtt"` |

### Example requests

**Transcription only (default):**

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio": "https://example.com/audio.mp3",
      "language": "ru"
    }
  }'
```

**Full pipeline with multiple output formats:**

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio": "https://example.com/audio.mp3",
      "language": "ru",
      "align_output": true,
      "diarization": true,
      "max_speakers": 3,
      "output_format": ["json", "srt", "vtt"]
    }
  }'
```

### Example responses

**`output_format: ["json"]` (default) with `align_output: true` and `diarization: true`:**

```json
{
  "detected_language": "ru",
  "json": {
    "segments": [
      {
        "start": 0.0,
        "end": 4.5,
        "text": "Привет, как дела?",
        "speaker": "SPEAKER_00",
        "words": [
          { "word": "Привет,", "start": 0.0, "end": 0.7, "score": 0.99 },
          { "word": "как",     "start": 0.8, "end": 1.1, "score": 0.97 },
          { "word": "дела?",   "start": 1.2, "end": 1.8, "score": 0.95 }
        ]
      }
    ]
  }
}
```

**`output_format: ["srt", "vtt"]` с `diarization: true`:**

```json
{
  "detected_language": "ru",
  "srt": "1\n00:00:00,000 --> 00:00:04,500\n[SPEAKER_00]: Привет, как дела?\n\n2\n00:00:05,100 --> 00:00:09,200\n[SPEAKER_01]: Всё хорошо, спасибо.\n\n",
  "vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:04.500\n<v SPEAKER_00>Привет, как дела?</v>\n\n00:00:05.100 --> 00:00:09.200\n<v SPEAKER_01>Всё хорошо, спасибо.</v>\n\n"
}
```

> Speaker метки в SRT/VTT появляются только при `diarization: true`. Без диаризации текст сегментов остаётся без изменений.

## Test script

The included `test_runpod.py` script submits a job to a running endpoint and polls for the result.

```bash
export RUNPOD_API_KEY=your_api_key
export RUNPOD_ENDPOINT_ID=your_endpoint_id

# Transcription only
python3 test_runpod.py --audio "https://example.com/audio.mp3"

# Full pipeline: align + diarize
python3 test_runpod.py \
  --audio "https://example.com/audio.mp3" \
  --align --diarize

# Local file (auto base64-encoded)
python3 test_runpod.py \
  --audio /path/to/audio.wav \
  --align --diarize --language ru
```

Results are saved to `results/<job_id>.json`.

## Environment variables (runtime)

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes (diarization) | Hugging Face read token — required to load the pyannote model |
| `WHISPER_MODEL` | No | Path or name of the Whisper model (default: `/models/faster-whisper-large-v3`) |

## Project structure

```
whisperx-runpod-worker/
├── Dockerfile                   # Container definition
├── builder/
│   ├── requirements.txt         # Extra pip deps: runpod, hf_transfer, nltk
│   └── download_models.sh       # Model pre-download script (runs at build time)
├── rp_handler.py                # RunPod serverless handler
├── rp_schema.py                 # Input validation schema
├── test_runpod.py               # CLI test script with job polling
└── test_input.json              # Sample request payload
```

## Known limitations

- **Alignment models** are only pre-cached for Russian (`ru`). For other languages, the alignment model is downloaded on the first request that uses `align_output: true` — which adds latency on the first call. Subsequent calls use the cached model.
- **Diarization** requires `HF_TOKEN` to be set at runtime as an environment variable on your RunPod endpoint.
- The image does not support CPU-only inference — CUDA is required.

## License

MIT
