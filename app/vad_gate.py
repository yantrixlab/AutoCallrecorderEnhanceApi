"""VAD-based processing for the /v1/remove-background-noise pipeline (see
processing_denoise.py) - detects speech regions with Silero VAD (learned
speech characteristics: pitch, formants, spectral shape) rather than a fixed
loudness threshold, then uses those regions two ways:

  1. level_speech_segments() - per-segment RMS leveling, bringing each
     detected speech segment toward a common target loudness. This is the
     same technique real "speech leveler" tools (Adobe Audition's Speech
     Volume Leveler, iZotope RX's Dialogue Leveler, Auphonic's own core
     algorithm) use: level per-utterance, not with one continuous whole-file
     gain curve, because a continuous envelope follower (e.g. ffmpeg's
     speechnorm, used in the /v1/enhance pipeline) has no actual knowledge of
     where speech starts/stops.
  2. apply_vad_gate() - attenuates everything NOT classified as speech.

A fixed-dB-threshold noise gate was tried once in processing.py's history and
destroyed real speech (a 3-second window of genuine quiet, -41dB speech came
back as near-total digital silence) - loudness alone can't tell "someone's
quiet delivery" from "dead air," because both can sit at the same level. VAD
can, because it classifies by acoustic characteristics instead.

Confirmed locally/in production this session: VAD badly under-detects speech
on raw or otherwise still-uneven audio (20-25% coverage on a real test call,
missing much of the quieter remote-caller side) - it needs reasonably
audible input to work reliably. detect_speech_segments() is meant to be
called on audio that's already had at least a quick loudness pass, never on
raw DeepFilterNet output directly.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from silero_vad import get_speech_timestamps, load_silero_vad

VAD_SAMPLE_RATE = 16000

# Padding protects word onsets/offsets (and quiet trailing consonants/breaths)
# from being clipped by an over-tight speech boundary. Widened from 180ms
# after real-world testing (via the temporary /v1/debug/denoise-only
# endpoint) showed detection on actual DeepFilterNet output topping out
# around ~39% coverage even with the richest detection-copy filter tried -
# nowhere near full coverage of a real call's actual speech content. Wider
# padding is a cheap way to recover some of the speech right at each
# detected segment's edges, which is exactly where under-detection is most
# likely to clip a word's start/end.
SPEECH_PAD_MS = 300

# Attenuate non-speech by ~18dB rather than the ~32dB first tried - real-
# world testing (see SPEECH_PAD_MS comment) showed detection coverage is
# nowhere near reliable enough on real DeepFilterNet output to risk a deep
# cut: any real speech VAD misses would otherwise come out sounding
# "wiped out" rather than just quieter. 18dB is still a clearly audible
# reduction for genuine background noise, but forgiving of the detector's
# real-world miss rate - erring toward "hear everything, quieter background"
# over "dead silent background, risk losing words," per explicit priority.
ATTENUATION_DB = -18.0

# Raised-cosine fade at every speech/non-speech transition - a hard step here
# is what causes audible clicking.
FADE_MS = 100

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_silero_vad()
    return _model


def _resample_for_vad(audio: np.ndarray, sr: int) -> np.ndarray:
    """Silero VAD expects 16kHz - this resample is only used to feed the
    detector, never applied to the actual output audio."""
    if sr == VAD_SAMPLE_RATE:
        return audio
    duration = len(audio) / sr
    new_len = int(duration * VAD_SAMPLE_RATE)
    x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0, duration, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _load_mono(path: Path) -> tuple:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def detect_speech_segments(input_wav: Path) -> list:
    """Returns Silero VAD's speech timestamps (in seconds) for input_wav.
    Call this on audio that's already reasonably normalized - see this
    module's docstring for why raw/uneven audio under-detects."""
    audio, sr = _load_mono(input_wav)
    vad_audio = _resample_for_vad(audio, sr)
    model = _get_model()
    return get_speech_timestamps(
        vad_audio, model, sampling_rate=VAD_SAMPLE_RATE,
        speech_pad_ms=SPEECH_PAD_MS, return_seconds=True,
    )


def level_speech_segments(audio: np.ndarray, sr: int, speech_segments: list,
                           target_rms_db: float = -20.0, max_gain_db: float = 15.0,
                           fade_ms: float = 60) -> np.ndarray:
    """Brings each speech segment's own RMS level toward target_rms_db,
    clamped to +/-max_gain_db so a very quiet, mostly-noise segment doesn't
    get pushed to an unnatural level. Gain ramps from 1.0 (unchanged) up to
    the computed gain over fade_ms at each segment's edges and back down to
    1.0 at the end, so there's no audible "step" where a segment meets the
    untouched audio around it. Non-speech audio is left untouched here -
    apply_vad_gate() handles attenuating it separately."""
    leveled = audio.copy()

    for seg in speech_segments:
        start = max(0, int(seg["start"] * sr))
        end = min(int(seg["end"] * sr), len(audio))
        if end <= start:
            continue

        segment = audio[start:end]
        rms = float(np.sqrt(np.mean(np.square(segment))) + 1e-9)
        rms_db = 20 * np.log10(rms)
        gain_db = float(np.clip(target_rms_db - rms_db, -max_gain_db, max_gain_db))
        gain = 10 ** (gain_db / 20)

        seg_len = end - start
        fade_samples = min(max(2, int(sr * fade_ms / 1000)), seg_len // 2)
        gain_curve = np.full(seg_len, gain, dtype=np.float32)
        if fade_samples > 1:
            ramp_up = 1.0 + (gain - 1.0) * (1 - np.cos(np.linspace(0, np.pi, fade_samples))) / 2
            ramp_down = 1.0 + (gain - 1.0) * (1 - np.cos(np.linspace(np.pi, 0, fade_samples))) / 2
            gain_curve[:fade_samples] = ramp_up
            gain_curve[-fade_samples:] = ramp_down

        leveled[start:end] = segment * gain_curve

    return leveled


def _build_silence_envelope(num_samples: int, sr: int, speech_segments: list,
                             attenuation_db: float = ATTENUATION_DB) -> np.ndarray:
    floor = 10 ** (attenuation_db / 20)
    envelope = np.full(num_samples, floor, dtype=np.float32)

    for seg in speech_segments:
        start_sample = max(0, int(seg["start"] * sr))
        end_sample = min(int(seg["end"] * sr), num_samples)
        if end_sample > start_sample:
            envelope[start_sample:end_sample] = 1.0

    fade_samples = max(2, int(sr * FADE_MS / 1000))
    half_fade = fade_samples // 2
    diff = np.diff(envelope, prepend=envelope[0])
    for idx in np.flatnonzero(diff != 0):
        lo = max(0, idx - half_fade)
        hi = min(num_samples, idx + half_fade)
        if hi <= lo + 1:
            continue
        ramp = (1 - np.cos(np.linspace(0, np.pi, hi - lo))) / 2
        start_val = envelope[lo]
        end_val = envelope[hi - 1]
        envelope[lo:hi] = start_val + (end_val - start_val) * ramp

    return envelope


def apply_vad_gate(input_wav: Path, output_wav: Path, speech_segments: list = None,
                    attenuation_db: float = ATTENUATION_DB) -> None:
    """Attenuates everything not classified as speech and writes the result
    to output_wav at the same sample rate. If speech_segments isn't given,
    detects them directly from input_wav (only safe if input_wav is already
    reasonably normalized - see this module's docstring). attenuation_db lets
    a caller dial this in gradually from a safe, barely-noticeable starting
    point rather than jumping straight to a deep cut - detection coverage on
    real DeepFilterNet output tops out around 40-47% even with the richest
    filter tried (see processing_denoise.py), so a caller should increase
    this incrementally against real listening feedback, not assume a deep
    value is safe by default."""
    audio, sr = _load_mono(input_wav)

    if speech_segments is None:
        speech_segments = detect_speech_segments(input_wav)

    envelope = _build_silence_envelope(len(audio), sr, speech_segments, attenuation_db)
    gated = audio * envelope
    sf.write(str(output_wav), gated, sr)
