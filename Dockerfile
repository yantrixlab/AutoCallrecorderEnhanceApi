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

# DeepFilterNet hardcodes `DataLoader(ds, num_workers=2, pin_memory=True)` in
# its own enhance.py with no CLI flag to change it. Docker's default /dev/shm
# is only 64MB, which multiprocessing DataLoader workers can exhaust on longer
# recordings (~10min), crashing with "Bus error... out of shared memory" -
# confirmed via a live crash against the deployed API. num_workers=0 runs the
# loader in the main process instead, needing no shared memory at all; safe
# performance-wise since jobs already run one at a time on a single thread.
RUN sed -i 's/num_workers=2/num_workers=0/' \
    /usr/local/lib/python3.11/site-packages/df/enhance.py

COPY app/ ./app/

ENV DATA_DIR=/data/jobs
ENV DB_PATH=/data/jobs.db
RUN mkdir -p /data/jobs

# PyTorch defaults to one CPU thread per core for its internal math kernels,
# each with its own working buffers - on a memory-constrained VPS that adds up
# fast for no real benefit here (jobs already run one at a time, single audio
# file, not a batch). Capping this reduces PyTorch's own memory footprint.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
