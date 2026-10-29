"""[옵트아웃 디지털 코드 - 음성] 고주파 대역 직접수열 확산(DSSS) + Reed-Solomon ECC.

문서 ③-A: "고주파 대역에 옵트아웃 디지털 코드 삽입(오류정정코드 적용, 압축·변환 후에도 유지)".
※ 이 방식은 논문(Volto)에 없는 우리 설계 요소(문서 4-4 '보충필요')라, 견고성 테스트로 검증해야 한다.

[동작 원리]
 1) payload(플래그+uid) -> RS 코드워드 비트열
 2) 비트마다 키로 만든 의사난수(PN) 반송파(고주파 대역통과)에 ±부호를 곱해 더함 (BPSK)
 3) [동기용 프리앰블][코드워드] 블록을 오디오 길이가 허용하는 만큼 '반복' 삽입 -> 반복 평균으로 잡음 억제
 4) 검출: 대역통과 -> 프리앰블 상관으로 위치 동기 -> 비트별 상관(soft) -> 블록 합산 -> RS 복호
 * 키(key)를 모르면 PN 을 재현할 수 없어 위치/내용을 알 수 없다.
 * 한계(알려진 것): 8 kHz 이하로 다운샘플링/전화 코덱은 고주파 대역을 지우므로 코드도 사라진다.
   -> band_hz 를 낮추면 생존성은 오르지만 음질 영향이 커짐: 트레이드오프를 실험으로 확정(TBD).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

from ..common.audio_io import rms
from ..common.ecc import BitECC


@dataclass
class AudioOptOutConfig:
    sr: int = 16000
    band_lo_hz: float = 5000.0       # ★TBD★ 삽입 대역 하한
    band_hi_hz: float = 7500.0       # ★TBD★ 삽입 대역 상한 (16 kHz 샘플링 -> 나이퀴스트 8 kHz)
    samples_per_bit: int = 256       # 비트당 샘플 수 (클수록 견고, 용량 감소)
    preamble_bits: int = 16          # 동기 프리앰블 길이(비트 단위)
    strength_db: float = -30.0       # ★TBD★ 호스트 RMS 대비 워터마크 RMS (dB)
    key: int = 20260001              # 비밀키 (PN 시드). 실서비스는 서버 비밀로 관리
    nsym: int = 8                    # RS 패리티 바이트 수
    start_offset: int = 2048         # 삽입 시작 위치(샘플). 앞쪽 이동/트리밍에 대비한 여유
    max_shift: int = 2048            # 검출 시 동기 탐색 반경(샘플)
    local_refine: int = 16           # 블록별 미세 동기 탐색 반경(샘플)


def _bandpass(x: np.ndarray, cfg: AudioOptOutConfig) -> np.ndarray:
    sos = butter(6, [cfg.band_lo_hz, cfg.band_hi_hz], btype="band", fs=cfg.sr, output="sos")
    return sosfiltfilt(sos, x)


def _raw_pn(n: int, key: int, stream: int) -> np.ndarray:
    return np.random.default_rng([key, stream]).standard_normal(n)


def _layout(cfg: AudioOptOutConfig, n_bits: int):
    P = cfg.preamble_bits * cfg.samples_per_bit        # 프리앰블 샘플 수
    D = n_bits * cfg.samples_per_bit                   # 코드워드 샘플 수
    return P, D, P + D


def embed_optout(x: np.ndarray, payload: bytes, cfg: AudioOptOutConfig) -> np.ndarray:
    """x 에 옵트아웃 코드를 삽입해 반환. 오디오가 너무 짧아 한 블록도 못 넣으면 ValueError."""
    ecc = BitECC(cfg.nsym)
    bits = ecc.encode(payload)
    P, D, B = _layout(cfg, len(bits))
    avail = len(x) - cfg.start_offset
    n_blocks = avail // B
    if n_blocks < 1:
        raise ValueError(f"오디오가 짧아 옵트아웃 코드 1블록({B} 샘플)을 넣을 수 없습니다.")

    spb = cfg.samples_per_bit
    sym = np.repeat(2.0 * bits - 1.0, spb)                         # 비트 -> ±1 심볼(샘플 단위)
    pre_raw = _raw_pn(P, cfg.key, 0)
    dat_raw = _raw_pn(D, cfg.key, 1)
    # 변조(부호 곱) 후 대역통과 -> 대역 밖 누설 제거
    block_sig = np.concatenate([pre_raw, dat_raw * sym])
    wm_block = _bandpass(block_sig, cfg)
    # 정규화 기준: 기준 반송파(변조 전)의 대역통과 RMS 로 스케일 고정
    ref_rms = rms(_bandpass(np.concatenate([pre_raw, dat_raw]), cfg))
    target = rms(x) * 10 ** (cfg.strength_db / 20)
    wm_block = wm_block * (target / ref_rms)

    y = x.astype(np.float64).copy()
    for b in range(n_blocks):
        s = cfg.start_offset + b * B
        y[s:s + B] += wm_block
    return y.astype(np.float32)


def detect_optout(x: np.ndarray, cfg: AudioOptOutConfig, n_payload_bytes: int = 5,
                  truth_payload: Optional[bytes] = None) -> Dict:
    """옵트아웃 코드 검출/복호.

    Returns dict:
      ok         : RS 복호 성공 여부 (= 코드가 존재하고 복원됨)
      payload    : 복원된 바이트 (실패 시 None)
      n_fixed    : RS 가 정정한 바이트 수
      raw_ber    : (truth_payload 를 주면) ECC 적용 '전' 비트 오류율 -> 채널 상태 진단용
      n_blocks   : 합산에 사용된 블록 수
    """
    ecc = BitECC(cfg.nsym)
    n_bits = ecc.n_bits(n_payload_bytes)
    P, D, B = _layout(cfg, n_bits)
    spb = cfg.samples_per_bit
    out = {"ok": False, "payload": None, "n_fixed": -1, "raw_ber": float("nan"), "n_blocks": 0}

    y = _bandpass(x.astype(np.float64), cfg)
    pre_ref = _bandpass(_raw_pn(P, cfg.key, 0), cfg)
    dat_ref = _bandpass(_raw_pn(D, cfg.key, 1), cfg).reshape(n_bits, spb)

    # --- 1) 첫 블록 위치 동기: 프리앰블과의 상관 피크 ---
    lo = max(0, cfg.start_offset - cfg.max_shift)
    hi = min(len(y), cfg.start_offset + cfg.max_shift + P)
    if hi - lo < P:
        return out
    corr = fftconvolve(y[lo:hi], pre_ref[::-1], mode="valid")
    off0 = lo + int(np.argmax(corr))

    # --- 2) 블록별 soft 값 계산 후 합산 ---
    soft_sum = np.zeros(n_bits)
    n_used = 0
    b = 0
    while True:
        exp = off0 + b * B
        if exp + B > len(y) + cfg.local_refine:
            break
        # 블록별 미세 동기(샘플레이트 오차/코덱 지연 보정)
        a0, a1 = max(0, exp - cfg.local_refine), min(len(y), exp + cfg.local_refine + P)
        if a1 - a0 < P:
            break
        c = fftconvolve(y[a0:a1], pre_ref[::-1], mode="valid")
        k = int(np.argmax(c))
        start = a0 + k
        gain = float(c[k]) / (np.sum(pre_ref ** 2) + 1e-12)       # 채널 이득 추정(프리앰블)
        seg = y[start + P:start + P + D]
        if len(seg) < D or gain <= 0:
            b += 1
            if b > 200:
                break
            continue
        soft = (seg.reshape(n_bits, spb) * dat_ref).sum(axis=1) / (np.sum(dat_ref ** 2, axis=1) + 1e-12)
        soft_sum += soft / gain
        n_used += 1
        b += 1
        if b > 200:
            break

    out["n_blocks"] = n_used
    if n_used == 0:
        return out
    bits_hat = (soft_sum > 0).astype(np.int8)
    if truth_payload is not None:
        truth_bits = ecc.encode(truth_payload)
        out["raw_ber"] = float(np.mean(bits_hat != truth_bits))
    payload, n_fixed = ecc.decode(bits_hat)
    out.update(ok=payload is not None, payload=payload, n_fixed=n_fixed)
    return out
