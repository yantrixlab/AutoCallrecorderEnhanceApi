"""The /v1/enhance pipeline - now unified with /v1/remove-background-noise's
approach (see processing_denoise.py), after that pipeline's per-segment
leveling, EQ correction, gentle noise gate, and loudness fix were all
validated against real DeepFilterNet output over several rounds this
session. The two stay separate functions/files rather than one shared
implementation, on purpose: this lets /v1/remove-background-noise keep
being tuned more aggressively (e.g. its noise-gate depth is still being
dialed up incrementally) without risking this stable, already-shipped path
the Android app depends on - if the two ever need genuinely different
settings again, they can diverge again from here.

Pipeline: ffmpeg static cleanup (highpass + notch, chunked) -> DeepFilterNet
-> ffmpeg afftdn/EQ mop-up -> per-segment VAD-based speech leveling -> gentle
VAD-gated noise attenuation -> ffmpeg final compressor/loudnorm/limiter ->
re-encode.

Jobs run one at a time on a single background thread, to avoid CPU/memory
contention on a modest VPS."""

import logging
import queue
import re
import shutil
import subprocess
import threading
from pathlib import Path

import soundfile as sf

from app import job_store, vad_gate

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

# How much non-speech is attenuated - kept in sync with
# processing_denoise.py's own starting point (a small, barely-noticeable
# reduction, meant to be nudged up gradually against real listening
# feedback) rather than the deeper -18dB tried and reverted earlier in that
# pipeline's evolution. Speech detection on real DeepFilterNet output tops
# out around 40-47% even with the richest detection filter tried, so
# there's always some real speech VAD doesn't catch.
NOISE_ATTENUATION_DB = -6.0

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


def _process_enhance_job(row) -> None:
    job_id = row["job_id"]
    extension = row["extension"]
    directory = job_dir(job_id)
    src = input_path(job_id, extension)
    chunks_dir = directory / "chunks"
    denoised_dir = directory / "denoised"
    concat_list = directory / "concat_list.txt"
    denoised_full = directory / "denoised_full.wav"
    detection_copy = directory / "detection_copy.wav"
    cleaned_full = directory / "cleaned_full.wav"
    leveled_full = directory / "leveled_full.wav"
    gated_full = directory / "gated_full.wav"
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

        # Cheap, throwaway copy purely to make speech detectable to VAD -
        # discarded immediately after detection, never touches the real
        # output. Needs the full afftdn/EQ/speechnorm/acompressor/loudnorm
        # chain to reliably expose quiet speech - lighter filters (plain
        # loudnorm, or loudnorm+speechnorm alone) were tried and validated
        # against real DeepFilterNet output via a temporary debug endpoint:
        # they only reached ~20-23% coverage (no better than raw audio),
        # while this full chain reached ~39-47%.
        _run([
            "ffmpeg", "-y", "-i", str(denoised_full),
            "-af", "afftdn=nr=15:nf=-40:tn=1,"
                   "bass=g=-3:f=200:width_type=h:width=200,"
                   "treble=g=4:f=3000:width_type=h:width=3000,"
                   "speechnorm=e=15:r=0.0004:l=1,"
                   "acompressor=threshold=0.1:ratio=3:attack=5:release=60,"
                   "loudnorm=I=-14:LRA=7:TP=-1.5",
            "-ar", "48000",
            str(detection_copy),
        ])
        speech_segments = vad_gate.detect_speech_segments(detection_copy)

        # Spectral mop-up + tonal balance, run on the ORIGINAL denoised_full,
        # not the throwaway detection copy:
        #   - afftdn: classical FFT spectral-subtraction denoiser, adaptively
        #     tracking the noise floor (tn=1). A second, different
        #     noise-reduction technique layered on top of the ML model - it
        #     cleans up steady residual hiss the neural model didn't fully
        #     remove, the standard "polish pass" in professional noise-reduction
        #     workflows (never rely on a single denoising technique alone).
        #   - bass/treble (EQ correction): fixes the output sounding muffled/
        #     bass-heavy next to a competing vendor's enhanced version.
        #     Measured band-by-band energy on a real test call: the vendor's
        #     output is close to flat from 80Hz through 3kHz (-24.2 to -25.9dB),
        #     while ours had the low end (-24.0dB) sitting 2-4dB louder than
        #     its own 300Hz-3kHz speech core and 6-12kHz "air" band - correct
        #     relative to raw phone audio, but noticeably duller than the
        #     target. A mild low shelf cut plus a presence/air high-shelf boost
        #     rebalances this without needing any bandwidth the source doesn't
        #     have.
        _run([
            "ffmpeg", "-y", "-i", str(denoised_full),
            "-af", "afftdn=nr=15:nf=-40:tn=1,"
                   "bass=g=-3:f=200:width_type=h:width=200,"
                   "treble=g=4:f=3000:width_type=h:width=3000",
            "-ar", "48000",
            str(cleaned_full),
        ])

        # Per-segment RMS leveling instead of a continuous envelope
        # follower (the old speechnorm-only approach) - brings each detected
        # speech segment toward a common loudness (quiet remote-caller
        # segments come up to match loud near-caller ones, and vice versa)
        # using the timestamps detected above. The same technique real
        # "speech leveler" tools (Adobe Audition's Speech Volume Leveler,
        # iZotope RX's Dialogue Leveler, Auphonic's own core algorithm)
        # actually use - a continuous envelope follower has no actual
        # knowledge of where speech starts/stops, which is what made the
        # old approach's dynamics tuning fragile.
        audio, sr = sf.read(str(cleaned_full), dtype="float32", always_2d=False)
        leveled = vad_gate.level_speech_segments(audio, sr, speech_segments)
        sf.write(str(leveled_full), leveled, sr)

        # Gentle VAD-gated noise attenuation - NOT a fixed-dB-threshold gate
        # (tried once, destroyed real speech: a 3-second window of genuine
        # quiet, -41dB speech came back as near-total digital silence, since
        # loudness alone can't tell "someone's quiet delivery" from "dead
        # air"). VAD classifies by acoustic characteristics instead, so it
        # can correctly tell them apart. Kept deliberately gentle
        # (NOISE_ATTENUATION_DB above) since detection coverage on real
        # DeepFilterNet output still tops out around 40-47%.
        vad_gate.apply_vad_gate(leveled_full, gated_full, speech_segments=speech_segments,
                                 attenuation_db=NOISE_ATTENUATION_DB)

        # Final touch-up + re-encode. threshold=0.008:ratio=15 (aggressive,
        # near limiting) is what it actually takes to create enough headroom
        # for loudnorm to hit -14 LUFS without violating its own TP ceiling -
        # gentler settings (threshold=0.1:ratio=3, then threshold=0.05:
        # ratio=6) were both confirmed via ebur128 (a pure standards-
        # compliant measurement, not loudnorm's own heuristic re-check,
        # which turned out unreliable) to undershoot the target by
        # 3.5-5dB on real output. TP raised slightly to -1.0 and alimiter's
        # ceiling to 0.92 accordingly.
        codec = _CODEC_FOR_EXTENSION.get(extension, "aac")
        cmd = [
            "ffmpeg", "-y", "-i", str(gated_full),
            "-af", "acompressor=threshold=0.008:ratio=15:attack=5:release=80,"
                   "loudnorm=I=-14:LRA=7:TP=-1.0,"
                   "alimiter=limit=0.92",
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

        # Safety net: if something still ends up muting the whole recording,
        # that's a much worse outcome than simply failing the job - the user
        # would get back a file that looks successful but is silent. -50dB
        # mean is comfortably below any real speech but well above true
        # digital silence (which measured -91dB on the file that triggered
        # this fix), so this only catches genuine whole-file wipeouts, not
        # just quiet content.
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
        for temp in (chunks_dir, denoised_dir, concat_list, denoised_full,
                     detection_copy, cleaned_full, leveled_full, gated_full):
            if temp.is_dir():
                shutil.rmtree(temp, ignore_errors=True)
            elif temp.exists():
                temp.unlink(missing_ok=True)


def _process_job(job_id: str) -> None:
    row = job_store.get_job(job_id)
    if row is None:
        logger.warning("Job %s vanished before processing", job_id)
        return

    mode = row["mode"] if "mode" in row.keys() else "enhance"
    if mode == "remove_background_noise":
        # Imported lazily to avoid a circular import (processing_denoise
        # imports plumbing helpers back from this module).
        from app import processing_denoise
        processing_denoise._process_denoise_job(row)
    else:
        _process_enhance_job(row)


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
