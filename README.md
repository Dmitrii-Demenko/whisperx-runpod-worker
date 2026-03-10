# whisperx-runpod-worker

RunPod Serverless worker for [WhisperX](https://github.com/m-bain/whisperX) — fast transcription with word-level timestamps and speaker diarization.

## Features

- **Transcription** via `faster-whisper large-v3` (30+ languages, auto-detect)
- **Word-level timestamps** via forced alignment (`wav2vec2`)
- **Speaker diarization** via `pyannote/speaker-diarization-community-1`
- **Audio input**: public URL or base64-encoded file
- **Supported formats**: MP3, WAV, FLAC, AAC, OGG, MP4, MKV, MOV, WebM and any other format supported by ffmpeg

## Prerequisites

- Docker with BuildKit enabled
- Hugging Face account with accepted terms for:
  - [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)
  - [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
- Docker Hub account (or any other container registry)

## Build

```bash
# Build for RunPod (linux/amd64)
docker build \
  --platform linux/amd64 \
  --secret id=hf_token,env=HF_TOKEN \
  -t your-dockerhub-username/whisperx-worker:latest \
  .

# Push to registry
docker push your-dockerhub-username/whisperx-worker:latest
```

> The build pre-downloads all models into the image (~15 GB) to eliminate cold-start latency.

## Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless) → **New Endpoint**
2. Select **Custom Image** and enter your Docker image URL
3. Set GPU: **RTX 4090** or **A100** recommended
4. Add environment variable: `HF_TOKEN=<your-huggingface-token>`
5. Save and wait for the worker to become active

## API

### Endpoint

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run      # async
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync  # sync (short audio only)
GET  https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}
```

### Input schema

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `audio` | `string` | required | Public URL or base64-encoded audio/video file |
| `language` | `string` | `null` | Language code (`ru`, `en`, ...). `null` = auto-detect |
| `batch_size` | `int` | `16` | Whisper batch size. Lower if OOM |
| `align_output` | `bool` | `false` | Enable word-level timestamps |
| `diarization` | `bool` | `false` | Enable speaker diarization |
| `min_speakers` | `int` | `null` | Minimum number of speakers (diarization hint) |
| `max_speakers` | `int` | `null` | Maximum number of speakers (diarization hint) |
| `initial_prompt` | `string` | `null` | Whisper initial prompt |
| `temperature` | `float` | `0` | Sampling temperature |
| `vad_onset` | `float` | `0.500` | VAD onset threshold |
| `vad_offset` | `float` | `0.363` | VAD offset threshold |

### Example request

```bash
curl -X POST "https://api.runpod.ai/v2/{ENDPOINT_ID}/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "audio": "https://example.com/audio.mp3",
      "language": "ru",
      "align_output": true,
      "diarization": true
    }
  }'
```

### Example response

```json
{
  "detected_language": "ru",
  "segments": [
    {
      "start": 0.0,
      "end": 4.5,
      "text": "Привет, как дела?",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Привет,", "start": 0.0, "end": 0.7, "score": 0.99},
        {"word": "как", "start": 0.8, "end": 1.1, "score": 0.97},
        {"word": "дела?", "start": 1.2, "end": 1.8, "score": 0.95}
      ]
    }
  ]
}
```

## Test script

```bash
# Transcription only
python3 test_runpod.py \
  --audio "https://example.com/audio.mp3"

# Full pipeline: align + diarize
python3 test_runpod.py \
  --audio "https://example.com/audio.mp3" \
  --align --diarize

# Local file
python3 test_runpod.py \
  --audio /path/to/audio.wav \
  --align --diarize --language ru
```

Set `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID` as environment variables or pass via `--api-key` / `--endpoint-id` flags.

Results are saved to `results/<job_id>.json`.

## Project structure

```
whisperx-runpod-worker/
├── Dockerfile                  # Container definition
├── builder/
│   ├── requirements.txt        # Extra pip deps (runpod, hf_transfer)
│   └── download_models.sh      # Model pre-download script (build time)
├── rp_handler.py               # RunPod serverless handler
├── rp_schema.py                # Input validation schema
├── test_runpod.py              # CLI test script with polling
└── test_input.json             # Sample request payload
```

## Environment variables (runtime)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face token (required for diarization model) |
| `WHISPER_MODEL` | Path or name of the Whisper model (default: `/models/faster-whisper-large-v3`) |
