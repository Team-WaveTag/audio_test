"""WaveTag 음성 파이프라인 (도면 ③-A + ④ 검증 + ④-재처리).

 ① 화자 특정 신호 제거:   longitudinal(bisection) -> lateral(ACO)           [volto_removal.py]
 ② 사용자 식별 워터마크:  Volto+ (시간영역)  또는  Volto* (생성모델)           [volto_plus.py / volto_star.py]
 ③ 옵트아웃 코드:         고주파 DSSS + Reed-Solomon                          [optout_code.py]
 ④ 검증:                  화자유사도(aux) / PESQ / SNR 이 합격 기준을 만족하는가
 ④-재처리:                음질 미달 -> 보안강도 한 단계 낮춤 / 보호 미달 -> 한 단계 높임 (최대 max_retries 회)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from ..common.config import ns_to_dict
from ..common.payload import pack_optout
from .metrics import cosine, pesq_score, snr_db
from .optout_code import AudioOptOutConfig, embed_optout
from .volto_plus import apply_volto_plus
from .volto_removal import RemovalConfig, lateral_crop_aco, longitudinal_crop


@dataclass
class ProtectResult:
    audio: np.ndarray
    passed: bool
    strength: str
    metrics: Dict[str, float]
    info: Dict = field(default_factory=dict)


class WaveTagAudioPipeline:
    def __init__(self, cfg, embed_fn: Callable[[np.ndarray], np.ndarray],
                 star_model=None, device=None):
        """cfg: load_config() 결과.  embed_fn: 보조(aux) 화자 인코더의 embed 함수 (SpeakerEncoder.embed)."""
        self.cfg = cfg
        self.sr = cfg.sr
        self.embed_fn = embed_fn
        self.star_model = star_model
        self.device = device
        self.optout_cfg = AudioOptOutConfig(sr=cfg.sr, **ns_to_dict(cfg.audio.optout))

    # ---------------------------------------------------------------- 단일 시도
    def protect_once(self, x: np.ndarray, uid: int, strength: str) -> ProtectResult:
        cfg = self.cfg
        st = getattr(cfg.strength, strength)
        rcfg_dict = ns_to_dict(cfg.audio.removal)
        rcfg_dict["target_sim"] = st.target_sim                      # 보안강도 -> 목표 유사도
        rcfg_dict["seed"] = cfg.seed
        rcfg = RemovalConfig(**rcfg_dict)

        ref_emb = self.embed_fn(x[None])[0]
        # ① 화자 특정 신호 제거
        x1, t, sim_lc = longitudinal_crop(x, self.embed_fn, ref_emb, rcfg)
        x2, aco_info = lateral_crop_aco(x1, self.sr, self.embed_fn, ref_emb, t, rcfg, pesq_ref=x)
        # ② 식별 워터마크 (학습 방해 + 소유권)
        vp = cfg.audio.volto_plus
        if cfg.audio.wm_mode == "star":
            from .volto_star import star_embed
            assert self.star_model is not None, "wm_mode=star 는 학습된 Volto* 모델이 필요합니다."
            x3 = star_embed(self.star_model, x2, uid, self.sr,
                            int(cfg.audio.volto_star.seg_sec * self.sr),
                            vp.slot_ms, vp.seed_base, self.device)
        else:
            x3 = apply_volto_plus(x2, uid, self.sr, st.wm_amp_ratio, vp.slot_ms, vp.seed_base)
        # ③ 옵트아웃 코드
        x4 = embed_optout(x3, pack_optout(uid), self.optout_cfg)
        x4 = np.clip(x4, -1.0, 1.0)

        # ④ 검증
        pc = cfg.pass_criteria
        m = {
            "sim_aux": float(cosine(self.embed_fn(x4[None])[0], ref_emb)[0]),
            "pesq": pesq_score(x, x4, self.sr, pc.pesq_mode),
            "snr_db": snr_db(x, x4),
        }
        quality_ok = (np.isnan(m["pesq"]) or m["pesq"] >= pc.pesq_min) and m["snr_db"] >= pc.snr_min_db
        protect_ok = m["sim_aux"] < pc.sim_max
        m["quality_ok"], m["protect_ok"] = bool(quality_ok), bool(protect_ok)
        return ProtectResult(x4, bool(quality_ok and protect_ok), strength, m,
                             {"longitudinal_t": t, "sim_after_lc": sim_lc, "aco": aco_info})

    # ---------------------------------------------------------------- 재처리 루프 (도면 ④-재처리)
    def protect(self, x: np.ndarray, uid: int, strength: str = "strong", max_retries: int = 2) -> ProtectResult:
        levels = list(self.cfg.strength_levels)
        idx = levels.index(strength)
        tried, best = set(), None
        for _ in range(max_retries + 1):
            tried.add(idx)
            r = self.protect_once(x, uid, levels[idx])
            r.info["attempt_levels"] = [levels[i] for i in sorted(tried)]
            if best is None or (r.passed, -abs(r.metrics["sim_aux"])) > (best.passed, -abs(best.metrics["sim_aux"])):
                best = r
            if r.passed:
                return r
            # 음질 미달이면 강도↓, 보호 미달이면 강도↑ (이미 시도한 단계로는 가지 않음)
            nxt = idx - 1 if not r.metrics["quality_ok"] else idx + 1
            if nxt < 0 or nxt >= len(levels) or nxt in tried:
                break
            idx = nxt
        return best
