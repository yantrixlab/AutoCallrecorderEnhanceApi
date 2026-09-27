"""The /v1/remove-background-noise pipeline - a deliberately separate clone of
processing.py's enhance pipeline, kept independent so this can be tuned
aggressively without risking the stable, already-shipped /v1/enhance path the
Android app relies on.

CURRENT STATE: noise suppression (VAD-gated attenuation) was fully disabled
at one point to isolate and validate leveling + EQ + normalize on their own,
then re-enabled at a deliberately gentle starting point
(NOISE_ATTENUATION_DB below) rather than jumping back to a deep cut. Speech
detection coverage on real DeepFilterNet output tops out around ~40-47% even
with the richest detection filter tried, meaning there's always some real
speech VAD doesn't catch - the gentle starting point is meant to be nudged
up incrementally against real listening feedback (per the user's own
request: "gradually increase and see where we can get at, and stay at the
point"), not jumped to a deep value that risks repeating the earlier
wiped-out-speech failure.

Unlike the enhance pipeline (which uses ffmpeg's speechnorm - a continuous
envelope follower with no actual knowledge of where speech starts/stops),
this pipeline does genuine per-segment speech leveling keyed on Silero VAD
boundaries (see vad_gate.py) - the same technique real "speech leveler" tools
(Adobe Audition's Speech Volume Leveler, iZotope RX's Dialogue Leveler,
Auphonic's own core algorithm) actually use. Real-world bugs already fixed in
this pipeline's evolution, worth remembering if this gets touched again:

  1. A fixed-dB-threshold noise gate (tried once in processing.py's history)
     destroyed real speech - loudness alone can't tell "someone's quiet
     delivery" from "dead air." VAD classifies by acoustic characteristics
     instead, so it can correctly tell them apart.
  2. VAD badly under-detects speech on raw/uneven audio, or on a
     lightweight/cheap detection-copy filter - it needs a genuinely rich
     normalization pass (the same afftdn/EQ/speechnorm/acompressor/loudnorm
     chain /v1/enhance uses) to reliably expose quiet speech. Validated
     against the server's *actual* DeepFilterNet output via the temporary
     /v1/debug/denoise-only endpoint, not a local approximation - an earlier
     round of fixes here was invalidated exactly by trusting a local
     ffmpeg-only stand-in that didn't behave like the real thing.
"""

import logging
import re
import shutil
from pathlib import Path

import soundfile as sf

from app import job_store, vad_gate
from app.processing import (
    _BITRATE_FOR_CODEC,
    _CODEC_FOR_EXTENSION,
    _run,
    input_path,
    job_dir,
    output_path,
)

logger = logging.getLogger("enhance_api")

# How much non-speech is attenuated - re-enabled after being fully disabled,
# starting deliberately gentle (a small, barely-noticeable reduction) rather
# than jumping back to the -18dB tried before disabling it. Nudge this up in
# small steps (e.g. -6 -> -9 -> -12) against real listening feedback each
# time, not all at once - detection coverage on real DeepFilterNet output
# tops out around 40-47% even with the richest filter tried (see the
# detection-copy filter below), so there's always some real speech VAD
# doesn't catch, and the whole point of starting gentle is finding how far
# this can go before that becomes audible as lost words rather than just
# quieter background.
NOISE_ATTENUATION_DB = -6.0


def _process_denoise_job(row) -> None:
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

        # Cheap, throwaway copy purely to make speech detectable to VAD -
        # discarded immediately after detection, never touches the real
        # output. Two earlier, weaker versions of this step (plain loudnorm,
        # then loudnorm+speechnorm alone) were validated only against a local
        # ffmpeg-only approximation of DeepFilterNet's output - no
        # DeepFilterNet available outside this server - and both turned out
        # to badly under-detect real speech once tested against the server's
        # *actual* DeepFilterNet output (confirmed via the temporary
        # /v1/debug/denoise-only endpoint): loudnorm alone matched raw
        # audio's poor ~20% coverage, and adding speechnorm only reached
        # ~23%. The full afftdn/EQ/speechnorm/acompressor/loudnorm chain -
        # the same one /v1/enhance already uses - measured at ~39% on that
        # same real DeepFilterNet output, nearly double either lighter
        # version. Two-pass "linear" loudnorm measurement made no further
        # difference over the simpler single-pass version, so single-pass is
        # kept here for less overhead.
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

        # Spectral mop-up + tonal balance - identical to the enhance
        # pipeline's afftdn/bass/treble stages (see processing.py's comments
        # for the full reasoning). These stay whole-file passes: they're
        # about overall noise/tone, not per-caller loudness, so
        # segment-awareness doesn't apply here. Runs on the ORIGINAL
        # denoised_full, not the throwaway detection copy.
        _run([
            "ffmpeg", "-y", "-i", str(denoised_full),
            "-af", "afftdn=nr=15:nf=-40:tn=1,"
                   "bass=g=-3:f=200:width_type=h:width=200,"
                   "treble=g=4:f=3000:width_type=h:width=3000",
            "-ar", "48000",
            str(cleaned_full),
        ])

        # Per-segment RMS leveling instead of a continuous envelope follower -
        # brings each detected speech segment toward a common loudness (quiet
        # remote-caller segments come up to match loud near-caller ones, and
        # vice versa) using the timestamps detected above. See vad_gate.py's
        # docstring for why this is "authentic" professional practice, not a
        # novel idea.
        audio, sr = sf.read(str(cleaned_full), dtype="float32", always_2d=False)
        leveled = vad_gate.level_speech_segments(audio, sr, speech_segments)
        sf.write(str(leveled_full), leveled, sr)

        # Noise suppression, re-enabled at NOISE_ATTENUATION_DB (module-level
        # constant above, currently a deliberately gentle starting point) -
        # attenuates everything not classified as speech, using the same
        # timestamps as the leveling step above (not re-detected - keeps
        # both stages consistent with exactly the same boundaries).
        vad_gate.apply_vad_gate(leveled_full, gated_full, speech_segments=speech_segments,
                                 attenuation_db=NOISE_ATTENUATION_DB)

        # Final touch-up + re-encode. Real deployed output measured (via
        # ebur128, a pure standards-compliant measurement - not loudnorm's
        # own heuristic re-check, which turned out unreliable here too) at
        # -19.2 LUFS despite targeting -14, then still -17.9 after a first
        # attempt to fix it (threshold=0.05:ratio=6): loudnorm can't push
        # the average up without violating its TP ceiling on the loudest
        # moments while loud and quiet content are still far apart, and that
        # first attempt's compression still wasn't strong enough to close
        # the gap on real (not locally-approximated) pipeline output.
        # Re-validated by exactly reproducing this pipeline's single-pass
        # behavior locally (not double-applying compression like the first,
        # invalid local test did) against real DeepFilterNet output:
        # threshold=0.008:ratio=15 (aggressive - near limiting) + TP raised
        # slightly to -1.0 lands at -14.4 LUFS, right on target, with safe
        # headroom (max -0.3dB, no clipping). alimiter's ceiling raised
        # accordingly (0.85->0.92).
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

        # Same whole-file-wipeout safety net as the enhance pipeline.
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
        for temp in (chunks_dir, denoised_dir, concat_list, denoised_full,
                     detection_copy, cleaned_full, leveled_full, gated_full):
            if temp.is_dir():
                shutil.rmtree(temp, ignore_errors=True)
            elif temp.exists():
                temp.unlink(missing_ok=True)
