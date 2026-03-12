FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-c"]
WORKDIR /

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        ffmpeg wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Extra pip dependencies (runpod SDK, hf_transfer, nltk)
COPY builder/requirements.txt /builder/requirements.txt
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    -r /builder/requirements.txt

# Install CUDA PyTorch 2.9.x (lowest version on cu128 that satisfies pyannote-audio).
# whisperx is installed with --no-deps to bypass its torch~=2.8.0 pin; all
# transitive dependencies are listed explicitly in the next step.
RUN python3 -m pip install --break-system-packages --no-cache-dir \
    "torch>=2.9.0,<2.11.0" "torchaudio>=2.9.0,<2.11.0" \
    --index-url https://download.pytorch.org/whl/cu128 && \
    python3 -m pip install --break-system-packages --no-cache-dir \
    --no-deps whisperx && \
    python3 -m pip install --break-system-packages --no-cache-dir \
    "ctranslate2>=4.5.0" "faster-whisper>=1.1.1" "nltk>=3.9.1" "numpy>=2.1.0" \
    "omegaconf>=2.3.0" "pandas>=2.2.3" "pyannote-audio>=4.0.0" \
    "huggingface-hub<1.0.0" "transformers>=4.48.0"

# Pre-download public models (no token needed).
# Gated models (pyannote) are downloaded on first startup via HF_TOKEN runtime env var.
COPY builder/download_models.sh /builder/download_models.sh
RUN chmod +x /builder/download_models.sh && /builder/download_models.sh

# Copy the VAD model to torch cache (whisperx looks there at runtime)
RUN mkdir -p /root/.cache/torch && \
    VAD_SRC=$(python3 -c "import whisperx, pathlib; print(pathlib.Path(whisperx.__file__).parent / 'assets' / 'pytorch_model.bin')") && \
    cp "$VAD_SRC" /root/.cache/torch/whisperx-vad-segmentation.bin

# Copy handler source
COPY rp_handler.py /rp_handler.py
COPY rp_schema.py /rp_schema.py

CMD ["python3", "-u", "/rp_handler.py"]
