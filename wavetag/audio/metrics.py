"""음성 평가 지표: SNR, PESQ, STOI, 코사인 유사도(화자 유사도).

※ 모든 임계값(TBD)은 이 파일이 아니라 configs/default.yaml 에서 관리한다.
※ PESQ 는 mode('wb' 광대역 / 'nb' 협대역)에 따라 점수 스케일이 달라진다.
   논문의 1.99 기준이 어느 모드 기준인지 원문에 명시되어 있지 않으므로,
   우리 임계값을 확정할 때는 반드시 '같은 mode' 로 일관되게 측정해야 한다.
"""
from __future__ import annotations

import warnings

import numpy as np


def snr_db(ref: np.ndarray, test: np.ndarray, eps: float = 1e-12) -> float:
    """SNR(dB) = 10 log10( ||ref||^2 / ||test - ref||^2 ).  ref=원본, test=보호/공격 후 음성."""
    n = min(len(ref), len(test))
    ref, test = ref[:n].astype(np.float64), test[:n].astype(np.float64)
    noise = test - ref
    return float(10 * np.log10((np.sum(ref ** 2) + eps) / (np.sum(noise ** 2) + eps)))


def pesq_score(ref: np.ndarray, test: np.ndarray, sr: int = 16000, mode: str = "wb") -> float:
    """PESQ (ITU-T P.862). 실패(무음 등) 시 NaN. 설치 안 되어 있으면 경고 후 NaN."""
    try:
        from pesq import pesq  # 윈도우는 C++ 빌드 도구 필요 (README 참고)
    except ImportError:
        warnings.warn("pesq 패키지가 없어 PESQ 를 건너뜁니다 (NaN).")
        return float("nan")
    n = min(len(ref), len(test))
    try:
        return float(pesq(sr, ref[:n].astype(np.float32), test[:n].astype(np.float32), mode))
    except Exception:  # NoUtterancesError 등
        return float("nan")


def stoi_score(ref: np.ndarray, test: np.ndarray, sr: int = 16000) -> float:
    """STOI(명료도, 0~1). 보조 지표."""
    try:
        from pystoi import stoi
    except ImportError:
        return float("nan")
    n = min(len(ref), len(test))
    try:
        return float(stoi(ref[:n], test[:n], sr, extended=False))
    except Exception:
        return float("nan")


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """행 단위 코사인 유사도. a:(N,D) 또는 (D,), b:(D,) 또는 (N,D)."""
    a = np.atleast_2d(a).astype(np.float64)
    b = np.atleast_2d(b).astype(np.float64)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return num / den
