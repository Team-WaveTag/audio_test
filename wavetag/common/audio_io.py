"""오디오 입출력 유틸. 내부 표준: mono, float32, 16 kHz, 범위 [-1, 1]."""
from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def load_audio(path: str | Path, sr: int = TARGET_SR,
               max_seconds: Optional[float] = None) -> np.ndarray:
    """wav/flac 등을 mono float32 로 읽고 sr 로 리샘플. max_seconds 가 있으면 앞부분만 사용."""
    x, sr_in = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)                       # stereo -> mono
    x = resample(x, sr_in, sr)
    if max_seconds is not None:
        x = x[: int(max_seconds * sr)]
    return np.ascontiguousarray(x, dtype=np.float32)


def save_audio(path: str | Path, x: np.ndarray, sr: int = TARGET_SR) -> None:
    """PCM_16 wav 로 저장. (저장 시 양자화가 일어나므로 평가는 '저장 후 다시 읽은' 파일로 하는 것을 권장)"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), sr, subtype="PCM_16")


def fix_length(x: np.ndarray, n: int) -> np.ndarray:
    """길이를 n 으로 맞춤 (짧으면 0 패딩, 길면 자름). 코덱 왕복 후 길이 보정에 사용."""
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + eps))
