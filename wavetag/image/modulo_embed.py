"""비밀키 p 기반 모듈로(나머지) 워터마크 삽입/추출  (EIAW 의 워터마크 부분).

[원리 - 문서 4-2 설명 그대로]
 - DCT 계수 c 를 p(짝수) 로 나눈 나머지로 비트를 기록:
        비트 0 -> c mod p ≈ p/4 ,   비트 1 -> c mod p ≈ 3p/4      (구간 중앙값 = 오차에 가장 여유)
 - 계수에 p 의 정수배를 더해도 나머지는 변하지 않는다
        => 적대적 최적화가 계수를 p 의 배수 단위로만 움직이면 워터마크는 그대로 보존된다.
 - 추출은 원본 없이 (p, 삽입 대역) 만으로 가능.

[키 디더(dither) - 테스트에서 발견한 보안 결함의 수정]
 p 만으로 나머지를 기록하면, 대부분의 중·고주파 계수가 |c| < p 로 작기 때문에 삽입값이 사실상 {+p/4, -p/4} 처럼
 '부호'로 비트를 나타낸다 -> 틀린 p 로도 거의 다 복호되는 것을 확인(p=10~64 중 p=16 외 27개 전부 복호 성공).
 해결: 슬롯마다 키로 만든 비밀 오프셋 d(0~p 균등난수)를 더해 기준점을 흔든다 (dithered QIM).
        삽입: c' = (target + d) + p*round((c - target - d)/p)      추출: soft = sin(2π (c - d) / p)
 p 의 정수배 이동은 여전히 나머지를 바꾸지 않으므로 적대적 최적화와 호환된다.
 -> 키 = (p, key 시드). 키를 모르면 d 를 알 수 없어 복호 불가.

[용량/중복]  비트 j 는 슬롯 j, j+n_bits, j+2*n_bits ... 에 반복 기록되고 추출 시 평균(soft vote)한다.
 이미지가 크면 반복 횟수가 많아져 견고해진다.  (슬롯 순서 = 채널 -> 블록행 -> 블록열 -> 대역 내 위치)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _dither(n: int, p: float, key) -> np.ndarray:
    """슬롯별 비밀 오프셋. key=None 이면 0 (디더 없음, 보안 약함 - 테스트 용도)."""
    if key is None:
        return np.zeros(n)
    return np.random.default_rng([int(key), int(p), 7]).uniform(0, p, size=n)   # 시드에 p 포함 -> p 가 달라지면 독립 난수


def embed_bits(coef: np.ndarray, mask: np.ndarray, bits: np.ndarray, p: float, key=None) -> np.ndarray:
    """coef (C,nbh,nbw,8,8) 의 mask 위치에 bits 를 반복 기록. 새 배열 반환."""
    assert p > 0 and int(p) % 2 == 0, "p 는 양의 짝수여야 합니다."
    out = coef.copy()
    sel = out[..., mask]                                   # (C,nbh,nbw,n_pos)
    flat = sel.reshape(-1)
    if flat.size < len(bits):
        raise ValueError(f"슬롯 수({flat.size}) < 비트 수({len(bits)}): 이미지가 너무 작습니다.")
    idx = np.arange(flat.size) % len(bits)
    target = np.where(bits[idx] == 0, p / 4.0, 3.0 * p / 4.0) + _dither(flat.size, p, key)
    flat_new = target + p * np.round((flat - target) / p)  # target 과 합동(mod p)인 가장 가까운 값
    out[..., mask] = flat_new.reshape(sel.shape)
    return out


def extract_bits(coef: np.ndarray, mask: np.ndarray, n_bits: int, p: float, key=None) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (hard bits(int8), confidence(0~1)).
    soft = sin(2π c / p): 비트0 중앙에서 +1, 비트1 중앙에서 -1. 반복 슬롯 평균 후 부호로 판정."""
    flat = coef[..., mask].reshape(-1)
    idx = np.arange(flat.size) % n_bits
    soft = np.sin(2 * np.pi * (flat - _dither(flat.size, p, key)) / p)
    cnt = np.bincount(idx, minlength=n_bits)
    mean = np.bincount(idx, weights=soft, minlength=n_bits) / np.maximum(cnt, 1)
    return (mean < 0).astype(np.int8), np.abs(mean)
