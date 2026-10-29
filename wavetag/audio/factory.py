"""설정 -> 파이프라인/인코더 조립 헬퍼 (스크립트들이 공통으로 사용)."""
from __future__ import annotations

from typing import Dict, Optional

import torch

from ..common.config import ns_to_dict
from .pipeline import WaveTagAudioPipeline
from .speaker_encoder import DummySpectralEncoder, SpeakerEncoder


def build_audio_stack(cfg, device: torch.device, generator_ckpt: Optional[str] = None,
                      cache_dir: str = "pretrained_models", dummy_encoder: bool = False):
    """Returns (pipeline, aux_encoder, {eval_name: SpeakerEncoder}).
    dummy_encoder=True: 모델 다운로드 없이 흐름만 점검하는 스모크 테스트 모드 (수치 무의미)."""
    if dummy_encoder:
        d = DummySpectralEncoder()
        print("[스모크 모드] DummySpectralEncoder 사용 - 보호 성능 수치는 의미 없음")
        star = None
        return WaveTagAudioPipeline(cfg, d.embed, star, device), d, {"dummy": d}
    aux = SpeakerEncoder(cfg.audio.aux_encoder, device, cache_dir)
    evals: Dict[str, SpeakerEncoder] = {}
    for name in cfg.audio.eval_encoders:
        if name == cfg.audio.aux_encoder:
            print(f"[경고] 평가 인코더 '{name}' 가 보호 생성용 aux 와 같습니다 -> 블랙박스 평가가 아닙니다.")
        evals[name] = aux if name == cfg.audio.aux_encoder else SpeakerEncoder(name, device, cache_dir)

    star = None
    ckpt = generator_ckpt or cfg.audio.volto_star.ckpt
    if cfg.audio.wm_mode == "star":
        from .volto_star import VoltoStar
        vs = cfg.audio.volto_star
        star = VoltoStar(vs.base, vs.depth, vs.delta_scale).to(device)
        star.load_state_dict(torch.load(ckpt, map_location=device))
        star.eval()
    return WaveTagAudioPipeline(cfg, aux.embed, star, device), aux, evals
