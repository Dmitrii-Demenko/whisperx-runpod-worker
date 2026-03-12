#!/usr/bin/env bash
# Downloads public (non-gated) models at Docker build time.
# Gated models (pyannote) are downloaded at container startup via HF_TOKEN env var.
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1

# --- 1. Faster-Whisper large-v3 (public, no token needed) ---
echo "==> Downloading faster-whisper-large-v3..."
huggingface-cli download Systran/faster-whisper-large-v3 \
    --local-dir /models/faster-whisper-large-v3

# --- 2. Wav2vec2 alignment model for Russian (public, no token needed) ---
echo "==> Downloading wav2vec2 alignment model (ru)..."
python3 -c "
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
for model_name in ['jonatasgrosman/wav2vec2-large-xlsr-53-russian']:
    print(f'Downloading {model_name}...')
    Wav2Vec2ForCTC.from_pretrained(model_name)
    Wav2Vec2Processor.from_pretrained(model_name)
print('Alignment models cached successfully')
"

# --- 3. NLTK punkt_tab tokenizer ---
echo "==> Downloading NLTK punkt_tab..."
python3 -c "import nltk; nltk.download('punkt_tab')"

echo "==> Public models downloaded. Pyannote will be downloaded at runtime via HF_TOKEN."
