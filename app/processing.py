"""The actual enhancement pipeline: ffmpeg resample -> DeepFilterNet denoising ->
ffmpeg loudness boost/normalize + re-encode back to the original format. Jobs run
one at a time on a single background thread, to avoid CPU/memory contention on a
modest VPS."""

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
    resampled = directory / "resampled.wav"
    denoised_dir = directory / "denoised"
    dest = output_path(job_id, extension)

    if not src.exists():
        job_store.set_status(job_id, "failed", "Uploaded file missing on server")
        return

    try:
        job_store.set_status(job_id, "processing")

        # Resample to 48kHz mono only - no loudness boost here. Boosting a quiet,
        # noisy call recording BEFORE DeepFilterNet sees it raises the noise floor
        # right along with the speech, which confuses the model's speech/noise
        # separation and was the actual cause of soft speech getting chopped out
        # as if it were noise. Loudness work happens after denoising instead, on
        # the already-clean signal.
        _run([
            "ffmpeg", "-y", "-i", str(src),
            "-ar", "48000", "-ac", "1",
            str(resampled),
        ])

        # DeepFilterNet noise suppression.
        # --no-suffix: by default the `deepFilter` CLI appends the model name as
        # a filename suffix (e.g. "..._DeepFilterNet3.wav") rather than reusing
        # the input's exact basename; this makes the output land at the
        # predictable path we expect below.
        # --atten-lim 20: caps how much the model is allowed to attenuate a
        # frame by mixing some of the original signal back in. Without this,
        # DeepFilterNet can fully zero out quiet speech on low-SNR call audio
        # because it looks similar to noise - capping the attenuation keeps
        # that speech audible while still cutting the noise floor substantially.
        denoised_dir.mkdir(parents=True, exist_ok=True)
        _run([
            "deepFilter", str(resampled),
            "--output-dir", str(denoised_dir),
            "--no-suffix",
            "--atten-lim", "20",
        ])
        denoised_file = denoised_dir / resampled.name
        if not denoised_file.exists():
            raise RuntimeError("DeepFilterNet did not produce an output file")

        # Now that the signal is clean, do the actual loudness work and
        # re-encode back to the original format:
        #   - acompressor: gently boosts quiet passages relative to loud ones
        #     (mild 3:1 downward compression) so speech is more consistently
        #     audible, not just louder on average.
        #   - loudnorm: normalizes to a louder target (-14 LUFS, up from the
        #     previous -16) now that it's operating on clean audio instead of
        #     noisy audio.
        #   - alimiter: brick-wall safety ceiling just under 0 dBFS in case
        #     compression + normalization pushes any transient over the top.
        codec = _CODEC_FOR_EXTENSION.get(extension, "aac")
        _run([
            "ffmpeg", "-y", "-i", str(denoised_file),
            "-af", "acompressor=threshold=0.1:ratio=3:attack=5:release=60,"
                   "loudnorm=I=-14:LRA=9:TP=-1.0,"
                   "alimiter=limit=0.9",
            "-c:a", codec, str(dest),
        ])

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
        for temp in (resampled, denoised_dir):
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
