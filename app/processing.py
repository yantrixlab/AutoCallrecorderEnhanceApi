"""The actual enhancement pipeline, following the standard professional order for
cleaning up noisy speech - static cleanup, then the adaptive/ML denoiser, then a
classical spectral mop-up pass, and only then loudness work on the now-clean
signal (boosting before denoising just amplifies the noise floor along with the
speech):

  ffmpeg (highpass rumble removal + notch for any fixed-frequency whine,
          split into 30s chunks) ->
  DeepFilterNet (ML denoising per chunk, bounded memory regardless of the
                 recording's total length) ->
  ffmpeg (stitch chunks back together, afftdn spectral mop-up, loudness
          boost/normalize, re-encode)

Jobs run one at a time on a single background thread, to avoid CPU/memory
contention on a modest VPS."""

import json
import logging
import queue
import re
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


def _run(cmd: list) -> subprocess.CompletedProcess:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Some tools (DeepFilterNet included) print harmless diagnostic lines
        # to stderr - like a failed `git rev-parse` in a git-less Docker image
        # - that have nothing to do with why the process actually exited
        # non-zero. Surfacing only stderr, as before, can point straight at
        # that noise instead of the real cause. Include both streams and the
        # exit code so a real failure is diagnosable from the error alone.
        raise RuntimeError(
            f"Command failed ({cmd[0]}), exit code {result.returncode}\n"
            f"stdout: {result.stdout.strip()[-2000:]}\n"
            f"stderr: {result.stderr.strip()[-2000:]}"
        )
    return result


def _measure_loudness(input_file: Path, pre_filters: str) -> dict:
    """Runs loudnorm in measurement-only mode and returns its JSON stats. Needed
    for two-pass (linear) normalization, which actually hits the target loudness
    accurately - unlike single-pass "dynamic" mode, which is only a heuristic and
    was observed silently undershooting badly on an unusually quiet real call,
    leaving the output below the downstream noise gate's threshold and getting
    the entire recording muted."""
    result = _run([
        "ffmpeg", "-i", str(input_file),
        "-af", f"{pre_filters},loudnorm=I=-14:LRA=9:TP=-1.5:print_format=json",
        "-f", "null", "-",
    ])
    match = re.search(r"\{[^{}]*\}", result.stderr, re.DOTALL)
    if not match:
        raise RuntimeError("Could not parse loudnorm measurement output")
    return json.loads(match.group(0))


def _process_job(job_id: str) -> None:
    row = job_store.get_job(job_id)
    if row is None:
        logger.warning("Job %s vanished before processing", job_id)
        return

    extension = row["extension"]
    directory = job_dir(job_id)
    src = input_path(job_id, extension)
    chunks_dir = directory / "chunks"
    denoised_dir = directory / "denoised"
    concat_list = directory / "concat_list.txt"
    denoised_full = directory / "denoised_full.wav"
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
        #
        # Also splits into 30-second chunks (-f segment) instead of one whole
        # file: DeepFilterNet loads and processes an entire file in memory in
        # a single pass, with no streaming. A ~15 minute real call recording
        # got SIGKILLed (exit code -9) by the Linux OOM killer on this VPS
        # after the model loaded successfully - confirmed via server logs,
        # not a guess - while an 11-second test clip worked fine. Since this
        # app records real phone calls that can easily run this long, memory
        # has to stay bounded regardless of recording length, not just work
        # for short test clips.
        chunks_dir.mkdir(parents=True, exist_ok=True)
        _run([
            "ffmpeg", "-y", "-i", str(src),
            "-af", "highpass=f=80,bandreject=f=6890:w=15:t=q",
            "-ar", "48000", "-ac", "1",
            "-f", "segment", "-segment_time", "30",
            str(chunks_dir / "chunk%04d.wav"),
        ])

        # DeepFilterNet noise suppression, run once over the whole chunks
        # directory (--noisy-dir) rather than per-chunk subprocess calls -
        # it loads and processes one file at a time internally either way
        # (confirmed from its own source: a DataLoader iterating file paths
        # one at a time), so batching them into a single invocation keeps
        # memory bounded per-chunk while only paying model-load overhead once.
        # --no-suffix: by default the `deepFilter` CLI appends the model name as
        # a filename suffix (e.g. "..._DeepFilterNet3.wav") rather than reusing
        # each input's exact basename; this makes the outputs land at the
        # predictable paths we expect below.
        # --atten-lim 20: caps how much the model is allowed to attenuate a
        # frame by mixing some of the original signal back in. Without this,
        # DeepFilterNet can fully zero out quiet speech on low-SNR call audio
        # because it looks similar to noise - capping the attenuation keeps
        # that speech audible while still cutting the noise floor substantially.
        # Briefly raised to 30 without being able to test it locally (no
        # DeepFilterNet install here) - that produced near-total silence on a
        # real low-SNR test recording, almost certainly from over-attenuating
        # actual speech combined with --pf. Reverted to the one value that's
        # actually been confirmed safe end-to-end. Don't raise this again
        # without a real test against the deployed server first.
        # --pf: post-filter that pushes noise suppression harder on the
        # noisiest sections. Safe to enable now that --atten-lim puts a floor
        # under how much speech it can remove in the process.
        denoised_dir.mkdir(parents=True, exist_ok=True)
        _run([
            "deepFilter",
            "--noisy-dir", str(chunks_dir),
            "--model-base-dir", "DeepFilterNet3",
            "--output-dir", str(denoised_dir),
            "--no-suffix",
            "--atten-lim", "20",
            "--pf",
        ])
        denoised_chunks = sorted(denoised_dir.glob("chunk*.wav"))
        if not denoised_chunks:
            raise RuntimeError("DeepFilterNet did not produce any output chunks")

        # Stitch the denoised chunks back into one file before continuing.
        # ffmpeg's concat demuxer just needs an ordered list of paths - chunk
        # filenames are zero-padded (chunk0000.wav, chunk0001.wav, ...) so a
        # plain sort already puts them back in chronological order.
        with open(concat_list, "w", encoding="utf-8") as f:
            for chunk in denoised_chunks:
                f.write(f"file '{chunk.resolve().as_posix()}'\n")
        _run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(denoised_full),
        ])
        denoised_file = denoised_full

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
        #   - loudnorm: normalizes to -14 LUFS, two-pass/"linear" (measured
        #     first via _measure_loudness, applied here with measured_* +
        #     linear=true) rather than single-pass "dynamic" mode. Dynamic
        #     mode is only a heuristic - on an unusually quiet real call
        #     (Flipkart IVR test) it silently undershot the target badly,
        #     landing the whole output below the noise gate's threshold below
        #     and getting the ENTIRE recording muted, not just some speech.
        #     Two-pass measurement makes hitting -14 LUFS reliable regardless
        #     of how quiet or loud the source material is, which is what
        #     actually makes a fixed gate threshold downstream valid at all.
        #     Note: loudnorm internally resamples for true-peak detection
        #     (observed output at 96kHz from a 48kHz input) - the explicit
        #     -ar 48000 below forces it back afterward.
        #   - agate: noise gate, placed AFTER loudnorm rather than before.
        #     Its threshold was calibrated by measuring real speech (~-18dB)
        #     vs noise-only pauses (~-37dB) on an already-normalized -14 LUFS
        #     reference file - a fixed threshold only means anything once the
        #     signal is reliably at that loudness (see the two-pass note above).
        #   - alimiter: brick-wall safety ceiling in case compression,
        #     normalization or the gate's release edge pushes any transient
        #     close to full scale - left a bit more headroom (0.85, ~-1.4dB)
        #     than loudnorm's own TP target since lossy re-encoding below can
        #     overshoot the true peak of the PCM by a fraction of a dB.
        pre_loudnorm_filters = (
            "afftdn=nr=15:nf=-40:tn=1,"
            "acompressor=threshold=0.1:ratio=3:attack=5:release=60"
        )
        measured = _measure_loudness(denoised_file, pre_loudnorm_filters)
        loudnorm_filter = (
            "loudnorm=I=-14:LRA=9:TP=-1.5:"
            f"measured_I={measured['input_i']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_thresh={measured['input_thresh']}:"
            "linear=true"
        )

        codec = _CODEC_FOR_EXTENSION.get(extension, "aac")
        cmd = [
            "ffmpeg", "-y", "-i", str(denoised_file),
            "-af", f"{pre_loudnorm_filters},"
                   f"{loudnorm_filter},"
                   "agate=threshold=0.04:ratio=4:attack=5:release=150:range=0.03,"
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

        # Safety net: if something (the gate or otherwise) still ends up
        # muting the whole recording, that's a much worse outcome than
        # simply failing the job - the user would get back a file that looks
        # successful but is silent. -50dB mean is comfortably below any real
        # speech but well above true digital silence (which measured -91dB
        # on the file that triggered this fix), so this only catches genuine
        # whole-file wipeouts, not just quiet content.
        volume_check = _run([
            "ffmpeg", "-i", str(dest), "-af", "volumedetect", "-f", "null", "-",
        ])
        mean_volume_match = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", volume_check.stderr)
        if mean_volume_match and float(mean_volume_match.group(1)) < -50:
            raise RuntimeError(
                f"Final output is unexpectedly silent (mean volume {mean_volume_match.group(1)}dB) - "
                "aborting rather than returning a broken result"
            )

        job_store.set_status(job_id, "done")
        logger.info("Job %s done", job_id)

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        job_store.set_status(job_id, "failed", str(e))
    finally:
        # Only the final output needs to survive for download - intermediate
        # files are pure clutter once we're done (or failed) with them.
        for temp in (chunks_dir, denoised_dir, concat_list, denoised_full):
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
