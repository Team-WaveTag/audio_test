"""[Volto 2단계 - 시간영역 방식 'Volto+'] 화자(사용자) 전용 트리거 워터마크.

논문 Sec.IV-B-1: i번째 화자(클래스)마다 고유한 가우시안 노이즈 조각을
파형의 i번째 시간 구간에 얹는다. 모델은 눈에 띄는 이 단순한 패턴(백도어식 shortcut)을
화자 특징보다 먼저 학습한다 -> (1) 학습 방해 보조 + (2) 학습 여부 검증용 지문.

우리 서비스 매핑: 논문의 '화자 번호 i' = WaveTag 의 '사용자 고유 식별번호 uid'.
  - 트리거 위치  : slot = uid % n_slots
  - 트리거 파형  : uid 로 시드를 고정한 가우시안 노이즈 (같은 uid 면 항상 같은 파형)
논문이 직접 지적한 한계: 필터링/디노이징으로 제거되기 쉬움 -> 견고성 테스트에서 반드시 확인.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import fftconvolve, lfilter

from ..common.audio_io import rms


def _slot_geometry(length: int, sr: int, slot_ms: float, uid: int) -> Tuple[int, int, int]:
    slot_len = int(sr * slot_ms / 1000)
    n_slots = max(1, length // slot_len)
    slot = uid % n_slots
    return slot_len, n_slots, slot


def trigger_segment(uid: int, slot_len: int, seed_base: int) -> np.ndarray:
    """uid 에 고정된 단위분산 가우시안 노이즈 조각."""
    rng = np.random.default_rng([seed_base, int(uid)])
    return rng.standard_normal(slot_len).astype(np.float32)


def time_trigger(length: int, uid: int, sr: int, slot_ms: float, seed_base: int) -> np.ndarray:
    """전체 길이 벡터: uid 의 슬롯 구간에만 노이즈, 나머지는 0 (논문의 time-domain watermark tw)."""
    slot_len, _, slot = _slot_geometry(length, sr, slot_ms, uid)
    tw = np.zeros(length, dtype=np.float32)
    tw[slot * slot_len:(slot + 1) * slot_len] = trigger_segment(uid, slot_len, seed_base)
    return tw


def apply_volto_plus(x: np.ndarray, uid: int, sr: int, amp_ratio: float,
                     slot_ms: float = 100.0, seed_base: int = 1234) -> np.ndarray:
    """x + amp_ratio * rms(x) * trigger.  amp_ratio 는 ★TBD★ (보안강도로 조절)."""
    tw = time_trigger(len(x), uid, sr, slot_ms, seed_base)
    return (x + amp_ratio * rms(x) * tw).astype(np.float32)


def _lpc_whitener(x: np.ndarray, order: int = 32) -> np.ndarray:
    """전역 LPC 역필터 계수 A(z). 음성은 저주파에 에너지가 몰린 '유색' 신호라
    그대로 백색 노이즈 조각과 상관을 구하면 호스트 신호가 상관값을 지배한다.
    -> 호스트를 백색화(whitening)한 뒤 매칭필터를 적용해야 트리거가 또렷해진다."""
    x = x.astype(np.float64)
    r = np.correlate(x, x, mode="full")[len(x) - 1:len(x) + order]
    r[0] *= 1.0 + 1e-9
    a = solve_toeplitz(r[:order], r[1:order + 1])
    return np.concatenate([[1.0], -a])


def detect_volto_plus(x: np.ndarray, uid: int, sr: int, slot_ms: float = 100.0,
                      seed_base: int = 1234) -> dict:
    """데이터 수준 검출: (백색화된) 신호와 트리거 조각의 정규화 상호상관 피크를 찾는다.

    - peak_ratio : (피크 |상관|) / (상관값 전체의 표준편차)  -> 클수록 트리거가 또렷함
    - loc_error  : 피크 위치와 기대 위치(slot*slot_len)의 차이(샘플)  -> 시간 이동 공격에 민감
    판정 예시: peak_ratio >= 6 이고 |loc_error| 가 작으면 '검출'. (임계값은 ★TBD★, 무삽입 파일의
    peak_ratio 분포를 먼저 측정해서 오경보율 기준으로 정할 것)
    ※ 이것은 '파일에 트리거가 남아있는지' 확인하는 지표이며, 논문의 핵심인
      '모델이 트리거에 반응하는지(모델 수준 검증)'는 scripts/train_sr_and_verify.py 에서 한다.
    """
    slot_len, _, slot = _slot_geometry(len(x), sr, slot_ms, uid)
    seg = trigger_segment(uid, slot_len, seed_base)
    a = _lpc_whitener(x)
    xw = lfilter(a, [1.0], x.astype(np.float64))
    sw = lfilter(a, [1.0], seg.astype(np.float64))
    c = fftconvolve(xw, sw[::-1], mode="valid") / (np.linalg.norm(sw) + 1e-12)
    pk = int(np.argmax(np.abs(c)))
    peak_ratio = float(np.abs(c[pk]) / (np.std(c) + 1e-12))
    return {"peak_ratio": peak_ratio, "loc_error": int(pk - slot * slot_len)}
