#!/usr/bin/env python3
"""
RunPod WhisperX — test script with polling and result saving.

Usage:
    python3 test_runpod.py --audio <URL or local file path>

Options:
    --audio           Public URL or path to local audio/video file (required)
    --language        Language code (ru, en, ...). Default: auto-detect
    --align           Enable word-level timestamps (flag)
    --diarize         Enable speaker diarization (flag)
    --min-speakers    Minimum speakers for diarization
    --max-speakers    Maximum speakers for diarization
    --batch-size      Whisper batch size (default: 16)
    --output          Path to save JSON result (default: results/<job_id>.json)
    --poll-interval   Polling interval in seconds (default: 5)
    --endpoint-id     RunPod endpoint ID (overrides env)
    --api-key         RunPod API key (overrides env)

Environment variables (loaded from .env if present):
    RUNPOD_API_KEY        RunPod API key
    RUNPOD_ENDPOINT_ID    RunPod endpoint ID
"""

import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error

# Load .env from the same directory as this script
_ROOT = Path(__file__).parent
_ENV_FILE = _ROOT / ".env"


def _load_env(path: Path) -> dict:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


_env = _load_env(_ENV_FILE)

RUNPOD_API_KEY = _env.get("RUNPOD_API_KEY") or os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = _env.get("RUNPOD_ENDPOINT_ID") or os.environ.get("RUNPOD_ENDPOINT_ID", "")
RUNPOD_API_URL = _env.get("RUNPOD_API_URL", "https://api.runpod.ai/v2").rstrip("/")


def _http_post(url: str, payload: dict, api_key: str) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _http_get(url: str, api_key: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _encode_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _resolve_audio(audio: str) -> str:
    if audio.startswith("http://") or audio.startswith("https://"):
        return audio
    p = Path(audio)
    if not p.exists():
        print(f"[ERROR] File not found: {audio}", file=sys.stderr)
        sys.exit(1)
    size_mb = p.stat().st_size / 1024 / 1024
    print(f"[INFO] Encoding file to base64: {p.name} ({size_mb:.1f} MB)...")
    return _encode_file(audio)


def run_job(
    audio: str,
    language: Optional[str],
    align_output: bool,
    diarization: bool,
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    batch_size: int,
    endpoint_id: str,
    api_key: str,
) -> str:
    payload: dict = {
        "input": {
            "audio": audio,
            "align_output": align_output,
            "diarization": diarization,
            "batch_size": batch_size,
        }
    }
    if language:
        payload["input"]["language"] = language
    if min_speakers is not None:
        payload["input"]["min_speakers"] = min_speakers
    if max_speakers is not None:
        payload["input"]["max_speakers"] = max_speakers

    url = f"{RUNPOD_API_URL}/{endpoint_id}/run"
    print(f"[INFO] Submitting job → {url}")
    resp = _http_post(url, payload, api_key)
    job_id = resp.get("id")
    if not job_id:
        print(f"[ERROR] Failed to get job_id. Response: {resp}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Job ID: {job_id}")
    return job_id


def poll_job(
    job_id: str,
    endpoint_id: str,
    api_key: str,
    poll_interval: float,
) -> dict:
    url = f"{RUNPOD_API_URL}/{endpoint_id}/status/{job_id}"
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = _http_get(url, api_key)
        except urllib.error.URLError as e:
            print(f"[WARN] Polling error (attempt {attempt}): {e}. Retrying...")
            time.sleep(poll_interval)
            continue

        status = resp.get("status", "UNKNOWN")
        elapsed = resp.get("delayTime", 0) + resp.get("executionTime", 0)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Status: {status:12s}  elapsed: {elapsed/1000:.1f}s")

        if status in ("COMPLETED", "FAILED", "CANCELLED"):
            return resp

        time.sleep(poll_interval)


def save_result(result: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    whisper_output = result.get("output", result)
    output_path.write_text(json.dumps(whisper_output, ensure_ascii=False, indent=2))
    print(f"[INFO] Result saved → {output_path}")


def print_summary(result: dict, wall_time_ms: int):
    status = result.get("status")
    output = result.get("output", {})

    print()
    print("=" * 60)
    print(f"  Status:           {status}")
    print(f"  Wall time:        {wall_time_ms / 1000:.1f}s")
    if "delayTime" in result:
        print(f"  RunPod delay:     {result['delayTime'] / 1000:.1f}s")
    if "executionTime" in result:
        print(f"  Execution time:   {result['executionTime'] / 1000:.1f}s")

    if status == "COMPLETED" and isinstance(output, dict):
        lang = output.get("detected_language", "—")
        segments = output.get("segments", [])
        print(f"  Language:         {lang}")
        print(f"  Segments:         {len(segments)}")
        if segments:
            duration = segments[-1].get("end", 0)
            print(f"  Duration:         {duration:.1f}s ({duration/60:.1f} min)")
        words_total = sum(len(s.get("words", [])) for s in segments)
        print(f"  Words:            {words_total}")
    elif status == "FAILED":
        print(f"  Error:            {result.get('error', output)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RunPod WhisperX test with polling")
    parser.add_argument("--audio", required=True, help="URL or path to audio/video file")
    parser.add_argument("--language", default=None, help="Language code (ru, en, ...)")
    parser.add_argument("--align", action="store_true", help="Enable word-level timestamps")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default=None, help="Path to save result JSON")
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Polling interval (sec)")
    parser.add_argument("--endpoint-id", default=None, help="RunPod endpoint ID")
    parser.add_argument("--api-key", default=None, help="RunPod API key")
    args = parser.parse_args()

    api_key = args.api_key or RUNPOD_API_KEY
    endpoint_id = args.endpoint_id or RUNPOD_ENDPOINT_ID

    if not api_key:
        print("[ERROR] RUNPOD_API_KEY not set. Use .env or --api-key", file=sys.stderr)
        sys.exit(1)
    if not endpoint_id:
        print("[ERROR] RUNPOD_ENDPOINT_ID not set. Use .env or --endpoint-id", file=sys.stderr)
        sys.exit(1)

    audio = _resolve_audio(args.audio)
    wall_start = time.monotonic()

    job_id = run_job(
        audio=audio,
        language=args.language,
        align_output=args.align,
        diarization=args.diarize,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        batch_size=args.batch_size,
        endpoint_id=endpoint_id,
        api_key=api_key,
    )

    print(f"[INFO] Polling every {args.poll_interval}s...")
    result = poll_job(job_id, endpoint_id, api_key, args.poll_interval)

    wall_ms = int((time.monotonic() - wall_start) * 1000)

    output_path = Path(args.output) if args.output else _ROOT / "results" / f"{job_id}.json"
    save_result(result, output_path)
    print_summary(result, wall_ms)


if __name__ == "__main__":
    main()
