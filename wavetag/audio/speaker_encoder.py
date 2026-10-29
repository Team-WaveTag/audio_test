"""화자 임베딩 모델 래퍼 (SpeechBrain 사전학습 모델).

[중요 - 평가 설계]
논문(Volto)은 '보호 음성 생성에 쓰는 보조 모델(i-vector)'과 '평가에 쓰는 SR 모델 6종'을
서로 다르게 두어 블랙박스 평가의 공정성을 확보했다.
우리도 동일하게 한다:
    aux  (보호 생성용) : xvector   <- ACO/절삭 탐색 중 유사도 계산에 사용
    eval (평가용)      : ecapa 등  <- 생성 과정에서 한 번도 보지 못한 모델로 측정
같은 모델로 만들고 같은 모델로 평가하면 '자기 자신 속이기'라 성능이 과대평가된다.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

# 이름 -> HuggingFace 모델 ID (SpeechBrain 공식 사전학습 모델)
# 모델을 더 추가하려면 여기에 한 줄 추가 (ID는 HuggingFace speechbrain 조직 페이지에서 확인)
REGISTRY = {
    "ecapa": "speechbrain/spkrec-ecapa-voxceleb",
    "xvector": "speechbrain/spkrec-xvect-voxceleb",
}


class SpeakerEncoder:
    """embed(wavs) -> (B, D) numpy 임베딩."""

    def __init__(self, name: str = "ecapa", device: Optional[torch.device] = None,
                 cache_dir: str = "pretrained_models", batch_size: int = 16):
        # SpeechBrain 1.0 부터 경로가 speechbrain.inference 로 이동 (구버전 호환 처리)
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # pragma: no cover
            from speechbrain.pretrained import EncoderClassifier

        self.name = name
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.model = EncoderClassifier.from_hparams(
            source=REGISTRY[name],
            savedir=os.path.join(cache_dir, name),
            run_opts={"device": str(self.device)},
        )
        self.model.eval()

    @torch.inference_mode()
    def embed(self, wavs: np.ndarray) -> np.ndarray:
        """wavs: (B, T) 또는 (T,) float32 16kHz  ->  (B, D)"""
        wavs = np.atleast_2d(wavs).astype(np.float32)
        out = []
        for i in range(0, len(wavs), self.batch_size):
            batch = torch.from_numpy(wavs[i:i + self.batch_size]).to(self.device)
            emb = self.model.encode_batch(batch)          # (b, 1, D)
            out.append(emb.squeeze(1).cpu().numpy())
        return np.concatenate(out, axis=0)


class DummySpectralEncoder:
    """[스모크 테스트 전용] 인터넷/모델 다운로드 없이 코드 흐름만 점검하기 위한 가짜 인코더.
    로그 스펙트럼을 64개 대역으로 풀링한 벡터를 임베딩으로 쓴다. 화자 식별 성능이 없으므로
    이 인코더로 얻은 보호 성능 수치는 의미가 없다 (파이프라인이 에러 없이 도는지만 확인)."""

    def embed(self, wavs: np.ndarray) -> np.ndarray:
        w = np.atleast_2d(wavs)
        S = np.abs(np.fft.rfft(w, axis=1))
        bands = np.array_split(np.arange(S.shape[1]), 64)
        return np.stack([np.log1p(S[:, b].mean(1) * 50) for b in bands], axis=1)
