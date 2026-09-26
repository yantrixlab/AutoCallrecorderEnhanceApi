FROM python:3.11-slim

# git: DeepFilterNet's logger shells out to `git rev-parse` for a diagnostic
# commit-hash log line - its own code already handles "not a git repo"
# (catches CalledProcessError) but not "git binary missing" (FileNotFoundError),
# which is what a git-less base image hits every time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch wheels - much smaller than the default CUDA build, and this
# service only ever runs on plain VPS CPU (no GPU available). Pinned to 2.0.x:
# DeepFilterNet's df.io still imports the old torchaudio.backend.common module,
# which torchaudio 2.2+ removed entirely (unfixed as of DeepFilterNet 5.0.6 -
# see https://github.com/Rikorose/DeepFilterNet/issues/662) - confirmed via a
# live ModuleNotFoundError against the deployed API on "latest".
RUN pip install --no-cache-dir torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir deepfilternet

COPY app/ ./app/

ENV DATA_DIR=/data/jobs
ENV DB_PATH=/data/jobs.db
RUN mkdir -p /data/jobs

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
