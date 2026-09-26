"""The actual enhancement pipeline: ffmpeg loudness normalization -> DeepFilterNet
denoising -> ffmpeg re-encode back to the original format. Jobs run one at a time
on a single background thread, to avoid CPU/memory contention on a modest VPS."""

import logging
import queue
import shutil
import subprocess
import threading
from pathlib import Path

from app import job_store

logger = logging.getLogger("enhance_api")

DATA_DIR = Path(__import__("os").environ.get("DATA_DIR", "/data/jobs"))

_CODEC_FOR_EXTENSION = {
    "m4a": "aac",
    "mp3": "libmp3lame",
    "wav": "pcm_s16le",
}

_job_queue: "queue.Queue[str]" = queue.Queue()


def job_dir(job_id: str) -> Path:
    return DATA_DIR / job_id


def input_path(job_id: str, extension: str) -> Path:
    return job_dir(job_id) / f"input.{extension}"


def output_path(job_id: str, extension: str) -> Path:
    return job_dir(job_id) / f"output.{extension}"


def enqueue(job_id: str) -> None:
    _job_queue.put(job_id)


def _run(cmd: list) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({cmd[0]}): {result.stderr.strip()[-2000:]}")


def _process_job(job_id: str) -> None:
    row = job_store.get_job(job_id)
    if row is None:
        logger.warning("Job %s vanished before processing", job_id)
        return

    extension = row["extension"]
    directory = job_dir(job_id)
    src = input_path(job_id, extension)
    normalized = directory / "normalized.wav"
    denoised_dir = directory / "denoised"
    dest = output_path(job_id, extension)

    if not src.exists():
        job_store.set_status(job_id, "failed", "Uploaded file missing on server")
        return

    try:
        job_store.set_status(job_id, "processing")

        # Loudness normalization (EBU R128) + resample to 48kHz mono, which is
        # what DeepFilterNet expects/works best with.
        _run([
            "ffmpeg", "-y", "-i", str(src),
            "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
            "-ar", "48000", "-ac", "1",
            str(normalized),
        ])

        # DeepFilterNet noise suppression - the `deepFilter` CLI writes its
        # output into --output-dir using the same base filename as the input.
        denoised_dir.mkdir(parents=True, exist_ok=True)
        _run(["deepFilter", str(normalized), "--output-dir", str(denoised_dir)])
        denoised_file = denoised_dir / normalized.name
        if not denoised_file.exists():
            raise RuntimeError("DeepFilterNet did not produce an output file")

        # Re-encode back to the original format so the app can drop this
        # straight in to replace the original recording, extension unchanged.
        codec = _CODEC_FOR_EXTENSION.get(extension, "aac")
        _run(["ffmpeg", "-y", "-i", str(denoised_file), "-c:a", codec, str(dest)])

        if not dest.exists() or dest.stat().st_size == 0:
            raise RuntimeError("Final re-encode produced an empty file")

        job_store.set_status(job_id, "done")
        logger.info("Job %s done", job_id)

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        job_store.set_status(job_id, "failed", str(e))
    finally:
        # Only the final output needs to survive for download - intermediate
        # files are pure clutter once we're done (or failed) with them.
        for temp in (normalized, denoised_dir):
            if temp.is_dir():
                shutil.rmtree(temp, ignore_errors=True)
            elif temp.exists():
                temp.unlink(missing_ok=True)


def _worker_loop() -> None:
    while True:
        job_id = _job_queue.get()
        try:
            _process_job(job_id)
        except Exception:
            logger.exception("Unhandled error processing job %s", job_id)
        finally:
            _job_queue.task_done()


def start_worker() -> None:
    thread = threading.Thread(target=_worker_loop, name="enhance-worker", daemon=True)
    thread.start()
