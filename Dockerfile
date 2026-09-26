FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch wheels - much smaller than the default CUDA build, and this
# service only ever runs on plain VPS CPU (no GPU available). torchaudio is
# required by DeepFilterNet's own audio load/save code (df.io) - installing
# only torch leaves that import failing at runtime with no build-time warning.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir deepfilternet

COPY app/ ./app/

ENV DATA_DIR=/data/jobs
ENV DB_PATH=/data/jobs.db
RUN mkdir -p /data/jobs

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
