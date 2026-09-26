"""The actual enhancement pipeline, following the standard professional order for
cleaning up noisy speech - static cleanup, then the adaptive/ML denoiser, then a
classical spectral mop-up pass, and only then loudness work on the now-clean
signal (boosting before denoising just amplifies the noise floor along with the
speech):

  ffmpeg (highpass rumble removal + notch for any fixed-frequency whine) ->
  DeepFilterNet (ML denoising, the main noise-removal stage) ->
  ffmpeg (afftdn spectral mop-up + loudness boost/normalize + re-encode)

Jobs run one at a time on a single background thread, to avoid CPU/memory
contention on a modest VPS."""

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

# Lossy codecs only - ffmpeg's default bitrate for mono AAC/MP3 is far too low
# (~69kbps) and its own quantization noise becomes clearly audible once the
# signal is normalized louder, which is exactly the "huge noise" that was
# actually a low-bitrate re-encode, not leftover background noise.
_BITRATE_FOR_CODEC = {
    "aac": "128k",
    "libmp3lame": "128k",
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

        # Static cleanup pass, before any denoising and with no loudness boost:
        #   - highpass=80: standard first step of any noise-reduction chain -
        #     removes sub-80Hz rumble/handling noise that sits below the voice
        #     fundamental and just adds to the noise floor.
        #   - bandreject centered on 6890Hz: a real test recording showed a
        #     constant single-frequency whine at ~6890Hz present even during
        #     total silence - a classic electrical/hardware tone, not the kind
        #     of noise an ML model or spectral denoiser targets. A narrow notch
        #     removes it directly; on recordings that don't have a tone there
        #     it's a no-op (nothing to cut).
        # No loudness boost here - boosting a quiet, noisy call recording
        # BEFORE DeepFilterNet sees it raises the noise floor right along with
        # the speech, which confuses the model's speech/noise separation and
        # was the actual cause of soft speech getting chopped out as if it
        # were noise. Loudness work happens after denoising instead, on the
        # already-clean signal.
        _run([
            "ffmpeg", "-y", "-i", str(src),
            "-af", "highpass=f=80,bandreject=f=6890:w=15:t=q",
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
        # --pf: post-filter that pushes noise suppression harder on the
        # noisiest sections. Safe to enable now that --atten-lim puts a floor
        # under how much speech it can remove in the process.
        denoised_dir.mkdir(parents=True, exist_ok=True)
        _run([
            "deepFilter", str(resampled),
            "--model-base-dir", "DeepFilterNet3",
            "--output-dir", str(denoised_dir),
            "--no-suffix",
            "--atten-lim", "20",
            "--pf",
        ])
        denoised_file = denoised_dir / resampled.name
        if not denoised_file.exists():
            raise RuntimeError("DeepFilterNet did not produce an output file")

        # Now that the signal is clean, mop up any residual noise DeepFilterNet
        # left behind and do the actual loudness work, then re-encode back to
        # the original format:
        #   - afftdn: classical FFT spectral-subtraction denoiser, adaptively
        #     tracking the noise floor (tn=1). This is a second, different
        #     noise-reduction technique layered on top of the ML model - it
        #     cleans up steady residual hiss the neural model didn't fully
        #     remove, the standard "polish pass" in professional noise-reduction
        #     workflows (never rely on a single denoising technique alone).
        #   - acompressor: gently boosts quiet passages relative to loud ones
        #     (mild 3:1 downward compression) so speech is more consistently
        #     audible, not just louder on average.
        #   - loudnorm: normalizes to a louder target (-14 LUFS, up from the
        #     previous -16) now that it's operating on clean audio instead of
        #     noisy audio. Note: loudnorm internally resamples for true-peak
        #     detection (observed output at 96kHz from a 48kHz input) - the
        #     explicit -ar 48000 below forces it back afterward.
        #   - alimiter: brick-wall safety ceiling in case compression +
        #     normalization pushes any transient close to full scale - left
        #     a bit more headroom (0.85, ~-1.4dB) than loudnorm's own TP
        #     target since lossy re-encoding below can overshoot the true
        #     peak of the PCM by a fraction of a dB.
        codec = _CODEC_FOR_EXTENSION.get(extension, "aac")
        cmd = [
            "ffmpeg", "-y", "-i", str(denoised_file),
            "-af", "afftdn=nr=15:nf=-40:tn=1,"
                   "acompressor=threshold=0.1:ratio=3:attack=5:release=60,"
                   "loudnorm=I=-14:LRA=9:TP=-1.5,"
                   "alimiter=limit=0.85",
            "-ar", "48000",
            "-c:a", codec,
        ]
        bitrate = _BITRATE_FOR_CODEC.get(codec)
        if bitrate:
            cmd += ["-b:a", bitrate]
        cmd.append(str(dest))
        _run(cmd)

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
