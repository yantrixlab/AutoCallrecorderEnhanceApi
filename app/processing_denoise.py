"""The /v1/remove-background-noise pipeline - a deliberately separate clone of
processing.py's enhance pipeline, kept independent so this can be tuned
aggressively without risking the stable, already-shipped /v1/enhance path the
Android app relies on.

Adds exactly one new stage versus the enhance pipeline: VAD-gated attenuation
(see vad_gate.py) between DeepFilterNet and the existing post-chain
(afftdn/EQ/speechnorm/loudnorm/limiter, unchanged). This targets the
background "haze" still audible between words on the enhance pipeline's
output - confirmed to be a DeepFilterNet residual, not something classical
filters (afftdn) or speechnorm's own threshold parameter could remove without
trading away the fixes already made to loudness/EQ. See vad_gate.py's
docstring for why VAD (not a fixed-dB noise gate, which was tried once and
destroyed real speech) is the right tool here.
"""

import logging
import re
import shutil
from pathlib import Path

from app import job_store, vad_gate
from app.processing import (
    _BITRATE_FOR_CODEC,
    _CODEC_FOR_EXTENSION,
    _measure_loudness,
    _run,
    input_path,
    job_dir,
    output_path,
)

logger = logging.getLogger("enhance_api")


def _process_denoise_job(row) -> None:
    job_id = row["job_id"]
    extension = row["extension"]
    directory = job_dir(job_id)
    src = input_path(job_id, extension)
    chunks_dir = directory / "chunks"
    denoised_dir = directory / "denoised"
    concat_list = directory / "concat_list.txt"
    denoised_full = directory / "denoised_full.wav"
    gated_full = directory / "gated_full.wav"
    dest = output_path(job_id, extension)

    if not src.exists():
        job_store.set_status(job_id, "failed", "Uploaded file missing on server")
        return

    try:
        job_store.set_status(job_id, "processing")

        # Static cleanup + chunking - identical to the enhance pipeline (see
        # processing.py's comments for why: no loudness boost before
        # DeepFilterNet, and 30s chunking to keep memory bounded).
        chunks_dir.mkdir(parents=True, exist_ok=True)
        _run([
            "ffmpeg", "-y", "-i", str(src),
            "-af", "highpass=f=80,bandreject=f=6890:w=15:t=q",
            "-ar", "48000", "-ac", "1",
            "-f", "segment", "-segment_time", "30",
            str(chunks_dir / "chunk%04d.wav"),
        ])

        # DeepFilterNet - same settings as the enhance pipeline to start.
        # Since this pipeline no longer shares risk with /v1/enhance, these
        # can be tuned independently (e.g. a higher --atten-lim) once the VAD
        # gate's own contribution has been validated on its own first.
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

        # THE new stage versus the enhance pipeline: attenuate everything
        # Silero VAD doesn't classify as speech, before the loudness chain
        # runs - see vad_gate.py's docstring for the full reasoning. Runs on
        # the DeepFilterNet output specifically (confirmed locally: running
        # VAD on raw, noisy audio badly under-detects speech - 20.6% on a
        # real test call vs 58.4% on the same call's DeepFilterNet output,
        # missing much of the quieter remote-caller side).
        vad_gate.apply_vad_gate(denoised_full, gated_full)
        denoised_file = gated_full

        # From here on, identical to the enhance pipeline's post-chain - see
        # processing.py's comments for the full reasoning behind each filter
        # (afftdn spectral mop-up, bass/treble EQ correction, speechnorm for
        # even loudness between callers, acompressor, two-pass loudnorm,
        # alimiter safety ceiling). Kept in sync manually since this pipeline
        # is intentionally a separate, independently-tunable clone.
        pre_loudnorm_filters = (
            "afftdn=nr=15:nf=-40:tn=1,"
            "bass=g=-3:f=200:width_type=h:width=200,"
            "treble=g=4:f=3000:width_type=h:width=3000,"
            "speechnorm=e=15:r=0.0004:l=1,"
            "acompressor=threshold=0.1:ratio=3:attack=5:release=60"
        )
        measured = _measure_loudness(denoised_file, pre_loudnorm_filters)
        loudnorm_filter = (
            "loudnorm=I=-14:LRA=7:TP=-1.5:"
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

        # Same whole-file-wipeout safety net as the enhance pipeline - even
        # more relevant here since VAD-gating is a new failure mode to guard
        # against (e.g. VAD detecting zero speech on a very noisy call).
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
        logger.info("Denoise job %s done", job_id)

    except Exception as e:
        logger.exception("Denoise job %s failed", job_id)
        job_store.set_status(job_id, "failed", str(e))
    finally:
        for temp in (chunks_dir, denoised_dir, concat_list, denoised_full, gated_full):
            if temp.is_dir():
                shutil.rmtree(temp, ignore_errors=True)
            elif temp.exists():
                temp.unlink(missing_ok=True)
