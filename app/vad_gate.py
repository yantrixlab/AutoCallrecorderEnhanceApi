"""VAD-gated attenuation for the /v1/remove-background-noise pipeline (see
processing_denoise.py) - quiets the background between words far more than
the standard /v1/enhance pipeline does, without repeating the mistake of the
fixed-dB-threshold noise gate removed from processing.py's history (it
destroyed real speech: a 3-second window of genuine quiet, -41dB speech came
back as near-total digital silence). A fixed loudness threshold can't tell
"someone's quiet delivery" from "dead air," because both can sit at the same
level - Silero VAD instead classifies audio using learned speech
characteristics (pitch, formants, spectral shape), so it can correctly keep a
quiet trailing consonant while still gating genuine silence at the same
loudness.

Confirmed locally before writing this: running VAD directly on RAW, noisy
call audio badly under-detects speech (20.6% of a real test call, missing
much of the quieter remote-caller side) - running it on the DeepFilterNet
output instead brought detection up to 58.4%, matching the recording's actual
visual speech activity. This module is meant to be called AFTER DeepFilterNet
denoising, never on raw audio.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from silero_vad import get_speech_timestamps, load_silero_vad

VAD_SAMPLE_RATE = 16000

# Padding protects word onsets/offsets (and quiet trailing consonants/breaths)
# from being clipped by an over-tight speech boundary.
SPEECH_PAD_MS = 180

# Attenuate non-speech by ~32dB rather than to full digital silence - keeps a
# small comfort-noise floor (avoids an unnatural "dead air" cutoff) and stays
# forgiving if VAD ever misses a genuinely quiet real speech moment.
ATTENUATION_DB = -32.0

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


def _build_envelope(num_samples: int, sr: int, speech_segments: list) -> np.ndarray:
    floor = 10 ** (ATTENUATION_DB / 20)
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


def apply_vad_gate(input_wav: Path, output_wav: Path) -> None:
    """Reads input_wav (expected: DeepFilterNet's already-denoised output),
    attenuates everything Silero VAD doesn't classify as speech, and writes
    the result to output_wav at the same sample rate/format."""
    audio, sr = sf.read(str(input_wav), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    vad_audio = _resample_for_vad(audio, sr)
    model = _get_model()
    speech_segments = get_speech_timestamps(
        vad_audio, model, sampling_rate=VAD_SAMPLE_RATE,
        speech_pad_ms=SPEECH_PAD_MS, return_seconds=True,
    )

    envelope = _build_envelope(len(audio), sr, speech_segments)
    gated = audio * envelope
    sf.write(str(output_wav), gated, sr)
