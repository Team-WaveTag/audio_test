"""[이미지 파이프라인] EIAW 기반: 적대적 신호 + 워터마크를 '하나의 신호'로 결합.

문서 ③-B 순서를 그대로 구현:
 (1) RGB 채널별 DCT
 (2) 주파수 마스크로 삽입 대역을 중·저주파(wm_band)로 제한
 (3) 비밀키 p 모듈로 연산으로 사용자 식별번호(ECC 적용) 기록
 (4) 계수를 p 의 정수배 단위로만 증감시키며 AI 모델을 교란하도록 반복 최적화
     -> p 배수를 더해도 나머지가 불변이므로 워터마크가 보존된다 (핵심 성질, tests 에서 검증)
 (5) EIAW 대역과 겹치지 않는 중·고주파(optout_band)에 옵트아웃 코드 삽입 (ECC 적용)
 (6) 역DCT -> 원본과 동일 형식(uint8)으로 복원

[적대적 최적화 방식 - 구현 가정]
 EIAW 원 논문의 최적화 세부(손실, 스텝 규칙)는 프로젝트 파일에 없어 다음처럼 구현했다:
   매 스텝: 대역 내 계수에 대해 손실(= 원래 예측 클래스의 CrossEntropy, 앙상블 합) 기울기를 계산 ->
   |기울기| 상위 adv_topk_frac 비율의 계수만 골라 부호방향으로 '정확히 ±p' 만큼 이동(정수배 단위) ->
   모든 모델이 오분류하면 (+여유 스텝) 조기 종료. 누적 이동량은 ±adv_max_mult*p 로 제한.
 조기 종료 덕에 필요한 만큼만 교란 -> 화질 보존. 세기(보안강도)는 adv_steps/adv_topk_frac/p 로 조절 (★TBD★).
 이 방식은 '화이트박스(모델 기울기 사용)'이므로 보류 모델 전이 성능은 별도 평가가 필수다 (쟁점 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..common.ecc import BitECC
from ..common.payload import pack_optout, pack_uid
from .dct_utils import (BS, band_mask, dct_blocks, idct_blocks, idct_blocks_t, pad_to_block, to_uint8)
from .modulo_embed import embed_bits, extract_bits


@dataclass
class EIAWConfig:
    # --- 키/대역 (★TBD★: 실험으로 확정) ---
    p: int = 16                       # 워터마크용 비밀키 p (짝수). 클수록 견고하지만 화질↓ (p=8 은 JPEG 에 매우 취약)
    wm_band: tuple = (2, 5)           # EIAW 대역: u+v 범위 (중·저주파)
    key: int = 20260002               # 워터마크 디더 시드 (비밀키의 일부. 서비스에서는 서버 비밀로 관리)
    p_optout: int = 16                # 옵트아웃 코드용 키
    key_optout: int = 20260003        # 옵트아웃 디더 시드
    optout_band: tuple = (8, 10)      # 옵트아웃 대역 (EIAW 대역과 비겹침 필수)
    nsym: int = 8                     # RS 패리티 바이트
    # --- 적대적 최적화 ---
    adv_steps: int = 40
    adv_topk_frac: float = 0.01       # 스텝당 이동시킬 계수 비율
    adv_max_mult: int = 2             # 계수별 누적 이동 한도 (p 의 배수)
    adv_extra_steps: int = 3          # 전 모델 오분류 후 추가 여유 스텝(양자화/압축 대비)


def _check_bands(cfg: EIAWConfig):
    assert not (band_mask(*cfg.wm_band) & band_mask(*cfg.optout_band)).any(), \
        "EIAW 대역과 옵트아웃 대역이 겹칩니다."


# ------------------------------------------------------------------ 적대적 최적화 (torch)
def adversarial_perturb(coef: np.ndarray, mask: np.ndarray, p: float, models: List, device,
                        cfg: EIAWConfig, img_u8_clean: np.ndarray) -> Dict:
    """coef(워터마크 삽입 완료)를 p 의 배수 단위로만 움직여 models 를 교란. Returns {coef, steps, fooled}."""
    import torch
    import torch.nn.functional as F

    # 교란 목표: 각 모델이 '깨끗한 원본'에서 내리는 예측을 틀리게 만들기
    x_clean = torch.from_numpy(img_u8_clean).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    with torch.no_grad():
        y0 = [m(x_clean).argmax(1) for m in models]

    c0 = torch.from_numpy(coef).float().unsqueeze(0).to(device)                 # (1,C,nbh,nbw,8,8)
    M = torch.from_numpy(mask).float().view(1, 1, 1, 1, BS, BS).to(device).expand_as(c0)
    delta = torch.zeros_like(c0)                                               # p 의 정수배만 저장
    n_slots = int(M.sum().item())
    k = max(1, int(cfg.adv_topk_frac * n_slots))
    extra = cfg.adv_extra_steps
    fooled = [False] * len(models)
    steps = 0

    for step in range(cfg.adv_steps):
        steps = step + 1
        c = (c0 + delta).requires_grad_(True)
        x = (idct_blocks_t(c).clamp(0, 255) / 255.0)
        loss = 0.0
        preds = []
        for m, y in zip(models, y0):
            logits = m(x)
            loss = loss + F.cross_entropy(logits, y)
            preds.append(logits.argmax(1))
        fooled = [bool((pr != y).item()) for pr, y in zip(preds, y0)]
        if all(fooled):                       # 전 모델 오분류 -> 여유 스텝만 더 하고 종료
            if extra <= 0:
                break
            extra -= 1
        g = torch.autograd.grad(loss, c)[0] * M                                 # 대역 밖 기울기는 0
        flat = g.abs().flatten()
        top = torch.topk(flat, k).indices
        step_t = torch.zeros_like(flat)
        step_t[top] = torch.sign(g.flatten()[top]) * p                         # 정확히 ±p (정수배)
        delta = (delta + step_t.view_as(delta)).clamp(-cfg.adv_max_mult * p, cfg.adv_max_mult * p)

    return {"coef": (c0 + delta).squeeze(0).cpu().numpy().astype(np.float64),
            "steps": steps, "fooled": fooled}


# ------------------------------------------------------------------ 보호 / 추출 (공개 API)
def protect_image(img_u8: np.ndarray, uid: int, cfg: EIAWConfig, models: Optional[List] = None,
                  device=None, optout_flags: int = 0x07) -> Dict:
    """이미지 보호 파이프라인 전체. models 가 None 이면 워터마크+옵트아웃만 삽입(적대적 교란 없음)."""
    _check_bands(cfg)
    ecc = BitECC(cfg.nsym)
    wm_bits = ecc.encode(pack_uid(uid))
    oo_bits = ecc.encode(pack_optout(uid, optout_flags))

    padded, (h, w) = pad_to_block(img_u8)
    coef = dct_blocks(padded)                                                  # (1)(2)
    wm_mask, oo_mask = band_mask(*cfg.wm_band), band_mask(*cfg.optout_band)
    coef = embed_bits(coef, wm_mask, wm_bits, cfg.p, cfg.key)                           # (3)
    coef = embed_bits(coef, oo_mask, oo_bits, cfg.p_optout, cfg.key_optout)                    # (5) (최적화 전에 넣어 기울기가 반영되게)

    info = {"adv_steps": 0, "fooled": None}
    if models:                                                                 # (4)
        r = adversarial_perturb(coef, wm_mask, cfg.p, models, device, cfg, padded)
        coef, info = r["coef"], {"adv_steps": r["steps"], "fooled": r["fooled"]}

    out = to_uint8(idct_blocks(coef))[:h, :w]                                  # (6)
    return {"image": out, "info": info}


def extract_image(img_u8: np.ndarray, cfg: EIAWConfig, which: str = "wm", n_payload_bytes: int = 4) -> Dict:
    """원본 없이 (키 p, 대역) 만으로 추출. which = 'wm'(식별번호) | 'optout'(옵트아웃 코드).
    Returns {ok, payload, bits(ECC 전), confidence}"""
    ecc = BitECC(cfg.nsym)
    n_bits = ecc.n_bits(n_payload_bytes)
    padded, _ = pad_to_block(img_u8)
    coef = dct_blocks(padded)
    if which == "wm":
        mask, p, key = band_mask(*cfg.wm_band), cfg.p, cfg.key
    else:
        mask, p, key = band_mask(*cfg.optout_band), cfg.p_optout, cfg.key_optout
    bits, conf = extract_bits(coef, mask, n_bits, p, key)
    payload, n_fixed = ecc.decode(bits)
    return {"ok": payload is not None, "payload": payload, "bits": bits, "n_fixed": n_fixed,
            "confidence": float(conf.mean())}
