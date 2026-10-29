"""음성 견고성 테스트용 공격(변환) 모음.

각 공격은 `fn(x, sr, rng) -> x'` 형태이며 길이는 입력과 동일하게 맞춘다.
논문 Sec.VI-A 의 방어 3종(다운샘플링 / MP3 / 양자화)을 포함하고,
WaveTag 의 실제 위협(메신저·SNS 재인코딩, 디노이징 제거, 시간 이동)을 추가했다.

※ MP3/AAC/Opus 는 ffmpeg 가 필요하다 (설치: conda install -c conda-forge ffmpeg).
   ffmpeg 가 없거나 코덱이 없으면 RuntimeError -> 평가 스크립트가 그 공격만 건너뛴다.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import soundfile as sf
from scipy.signal import firwin, fftconvolve, istft, stft

from ..common.audio_io import fix_length, resample, rms

AttackFn = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


# --------------------------------------------------------------------------- 기본 변환
def identity(x, sr, rng):
    return x


def resample_roundtrip(x, sr, rng, low_sr: int = 8000):
    """논문의 다운샘플링 방어: sr -> low_sr -> sr (고주파 대역 소실)."""
    y = resample(resample(x, sr, low_sr), low_sr, sr)
    return fix_length(y, len(x))


def quantize(x, sr, rng, bits: int = 10):
    """논문의 양자화 방어: 비트 깊이 축소."""
    levels = 2 ** (bits - 1)
    return (np.round(np.clip(x, -1, 1) * levels) / levels).astype(np.float32)


def add_noise(x, sr, rng, snr_db: float = 20.0):
    """백색 가우시안 잡음을 주어진 SNR 로 추가."""
    noise = rng.standard_normal(len(x))
    noise *= rms(x) / rms(noise) * 10 ** (-snr_db / 20)
    return (x + noise).astype(np.float32)


def lowpass(x, sr, rng, cutoff: float = 4000.0):
    """가파른 FIR 저역통과(401 tap, 선형위상). 전화망/저대역 코덱의 대역 제한을 모사."""
    h = firwin(401, cutoff, fs=sr)
    return fftconvolve(x, h, mode="same").astype(np.float32)


def gain(x, sr, rng, db: float = -6.0):
    return (x * 10 ** (db / 20)).astype(np.float32)


def time_shift(x, sr, rng, ms: float = 50.0):
    """앞에 무음 추가(동기 깨짐). 뒤는 잘라 길이 유지."""
    k = int(sr * ms / 1000)
    return np.concatenate([np.zeros(k, np.float32), x])[: len(x)]


def spectral_gate(x, sr, rng, n_fft: int = 512, k: float = 1.5):
    """간단한 스펙트럼 게이팅 디노이저. 주파수별 하위 10% 크기를 잡음 바닥으로 보고 그 아래를 억제.
    논문이 Volto+ 의 약점으로 지적한 '필터링/디노이징에 의한 워터마크 제거'를 모사한다."""
    hop = n_fft // 4
    _, _, Z = stft(x, sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Z)
    floor = np.percentile(mag, 10, axis=1, keepdims=True)
    mask = mag ** 2 / (mag ** 2 + (k * floor) ** 2 + 1e-12)      # 부드러운 마스크
    _, y = istft(Z * mask, sr, nperseg=n_fft, noverlap=n_fft - hop)
    return fix_length(y.astype(np.float32), len(x))


# --------------------------------------------------------------------------- 코덱 (ffmpeg)
def _ffmpeg_roundtrip(x: np.ndarray, sr: int, enc_args: list, ext: str, enc_sr: int | None):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg 가 설치되어 있지 않습니다.")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src, enc, dec = d / "in.wav", d / f"enc.{ext}", d / "dec.wav"
        sf.write(str(src), x, sr, subtype="PCM_16")
        cmd_enc = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1"]
        if enc_sr:
            cmd_enc += ["-ar", str(enc_sr)]
        try:
            subprocess.run(cmd_enc + enc_args + [str(enc)], check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(enc),
                            "-ar", str(sr), "-ac", "1", str(dec)], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg 코덱 실패: {e.stderr.decode(errors='ignore')[:200]}")
        y, _ = sf.read(str(dec), dtype="float32")
    return fix_length(y, len(x))


def mp3(x, sr, rng, kbps: int = 128):
    # MPEG-1 Layer III 가 64~192kbps 를 모두 지원하도록 44.1kHz 로 인코딩 (실제 플랫폼 업로드와 유사)
    return _ffmpeg_roundtrip(x, sr, ["-c:a", "libmp3lame", "-b:a", f"{kbps}k"], "mp3", 44100)


def aac(x, sr, rng, kbps: int = 64):
    return _ffmpeg_roundtrip(x, sr, ["-c:a", "aac", "-b:a", f"{kbps}k"], "m4a", 44100)


def opus(x, sr, rng, kbps: int = 24):
    return _ffmpeg_roundtrip(x, sr, ["-c:a", "libopus", "-b:a", f"{kbps}k"], "opus", sr)


# --------------------------------------------------------------------------- 실행 헬퍼
def to_pcm16(x: np.ndarray) -> np.ndarray:
    """16bit 양자화 왕복(= wav 로 저장했다가 다시 읽은 효과)."""
    return (np.round(np.clip(x, -1, 1) * 32767) / 32767).astype(np.float32)


def apply_attack(x: np.ndarray, fn: AttackFn, sr: int, rng: np.random.Generator) -> np.ndarray:
    """공격 적용 후 PCM16 으로 재양자화.
    ★이유★ 부동소수점 상태에서는 LTI 필터가 신호와 잡음을 똑같이 줄이기 때문에 워터마크가
    실제보다 잘 살아남는 것처럼 보인다. 실제 배포 파일은 16bit 로 저장되므로 같은 조건으로 맞춘다."""
    return to_pcm16(fn(x, sr, rng))


# --------------------------------------------------------------------------- 레지스트리
def get_attack_suite() -> Dict[str, AttackFn]:
    """이름 -> 공격 함수. 평가 스크립트가 이 이름으로 선택/집계한다."""
    return {
        "identity": identity,
        # 논문 Sec.VI-A
        "resample_12k": partial(resample_roundtrip, low_sr=12000),
        "resample_10k": partial(resample_roundtrip, low_sr=10000),
        "resample_8k": partial(resample_roundtrip, low_sr=8000),
        "mp3_192k": partial(mp3, kbps=192),
        "mp3_128k": partial(mp3, kbps=128),
        "mp3_64k": partial(mp3, kbps=64),
        "quant_10bit": partial(quantize, bits=10),
        "quant_8bit": partial(quantize, bits=8),
        # WaveTag 추가 위협
        "aac_64k": partial(aac, kbps=64),
        "opus_24k": partial(opus, kbps=24),
        "noise_30dB": partial(add_noise, snr_db=30.0),
        "noise_20dB": partial(add_noise, snr_db=20.0),
        "noise_10dB": partial(add_noise, snr_db=10.0),
        "lowpass_4k": partial(lowpass, cutoff=4000.0),
        "gain_-6dB": partial(gain, db=-6.0),
        "shift_50ms": partial(time_shift, ms=50.0),
        "spectral_gate": spectral_gate,
    }
