#!/usr/bin/env bash
set -euo pipefail

# Read HF token from Docker build secret
HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || echo "")

if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: No HF_TOKEN secret provided. Gated models (pyannote) will fail to download."
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

# --- 1. Faster-Whisper large-v3 (CTranslate2 format) ---
echo "==> Downloading faster-whisper-large-v3..."
huggingface-cli download Systran/faster-whisper-large-v3 \
    --local-dir /models/faster-whisper-large-v3 \
    --token "$HF_TOKEN"

# --- 2. Pyannote speaker-diarization model ---
echo "==> Downloading pyannote/speaker-diarization-community-1..."
python3 -c "
from huggingface_hub import login
login(token='$HF_TOKEN', add_to_git_credential=False)
from pyannote.audio import Pipeline
Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token='$HF_TOKEN')
print('Diarization model cached successfully')
"

# --- 3. Wav2vec2 alignment model for Russian (pre-cached, others downloaded on demand) ---
echo "==> Downloading wav2vec2 alignment model (ru)..."
python3 -c "
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from huggingface_hub import login
login(token='$HF_TOKEN', add_to_git_credential=False)
for model_name in [
    'jonatasgrosman/wav2vec2-large-xlsr-53-russian',
]:
    print(f'Downloading {model_name}...')
    Wav2Vec2ForCTC.from_pretrained(model_name)
    Wav2Vec2Processor.from_pretrained(model_name)
print('Alignment models cached successfully')
"

# --- 4. NLTK punkt_tab tokenizer ---
echo "==> Downloading NLTK punkt_tab..."
python3 -c "import nltk; nltk.download('punkt_tab')"

echo "==> All models downloaded successfully."
