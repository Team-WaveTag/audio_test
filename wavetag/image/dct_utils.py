"""8x8 블록 DCT 유틸 (numpy 버전 + 적대적 최적화용 torch 버전).

[좌표 규약] 계수 텐서 모양 = (C, nbh, nbw, 8, 8)   (C=RGB 채널, nbh/nbw=블록 격자, 8x8=(u,v) 주파수)
- 정규직교(ortho) DCT-II 를 쓰므로 계수 영역 오차 = 화소 영역 오차 (에너지 보존)
- 화소값에서 128 을 뺀 뒤 변환 (JPEG 관례)

[설계 가정 - 확인 필요]
문서는 "RGB 채널별 DCT" 라고만 적혀 있어 JPEG 와 동일한 8x8 블록 DCT 로 구현했다.
EIAW 원 논문이 전체 이미지 DCT 인지 블록 DCT 인지는 프로젝트 파일에 논문이 없어 확인하지 못했다.
(논문 확인 후 달라지면 이 파일의 dct_blocks/idct_blocks 만 교체하면 된다.)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.fft import dctn, idctn

BS = 8  # 블록 크기


def pad_to_block(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """HxWxC 이미지를 8의 배수 크기로 가장자리 복제 패딩. (패딩된 이미지, 원래 (H,W)) 반환."""
    h, w = img.shape[:2]
    ph, pw = (-h) % BS, (-w) % BS
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return img, (h, w)


def dct_blocks(img: np.ndarray) -> np.ndarray:
    """HxWxC (H,W 는 8의 배수, 값 0~255) -> (C, nbh, nbw, 8, 8) DCT 계수."""
    h, w, c = img.shape
    blk = (img.astype(np.float64) - 128.0).reshape(h // BS, BS, w // BS, BS, c).transpose(4, 0, 2, 1, 3)
    return dctn(blk, axes=(3, 4), norm="ortho")


def idct_blocks(coef: np.ndarray) -> np.ndarray:
    """(C, nbh, nbw, 8, 8) -> HxWxC (값 0~255 범위로 클리핑 전 float)."""
    blk = idctn(coef, axes=(3, 4), norm="ortho")
    c, nbh, nbw = blk.shape[:3]
    return blk.transpose(1, 3, 2, 4, 0).reshape(nbh * BS, nbw * BS, c) + 128.0


def to_uint8(img_f: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img_f), 0, 255).astype(np.uint8)


def band_mask(lo: int, hi: int) -> np.ndarray:
    """주파수 대역 마스크: lo <= u+v <= hi 인 (u,v) 위치 (DC 제외).
    u+v 가 작을수록 저주파. 예) (2,5)=중·저주파, (8,10)=중·고주파."""
    m = np.zeros((BS, BS), dtype=bool)
    for u in range(BS):
        for v in range(BS):
            if (u, v) != (0, 0) and lo <= u + v <= hi:
                m[u, v] = True
    return m


# ---------------------------------------------------------------- torch 버전 (미분 가능)
def _dct_matrix_t(device=None):
    import math
    import torch
    k = torch.arange(BS, dtype=torch.float32).unsqueeze(1)
    n = torch.arange(BS, dtype=torch.float32).unsqueeze(0)
    D = torch.cos(math.pi * (2 * n + 1) * k / (2 * BS)) * math.sqrt(2.0 / BS)
    D[0] /= math.sqrt(2.0)
    return D.to(device) if device is not None else D


def idct_blocks_t(coef):
    """torch: (B, C, nbh, nbw, 8, 8) -> (B, C, H, W)  (+128 복원 포함, 클리핑 전)."""
    D = _dct_matrix_t(coef.device)
    blk = D.T @ coef @ D
    b, c, nbh, nbw = blk.shape[:4]
    return blk.permute(0, 1, 2, 4, 3, 5).reshape(b, c, nbh * BS, nbw * BS) + 128.0


def dct_blocks_t(x):
    """torch: (B, C, H, W) 0~255 -> (B, C, nbh, nbw, 8, 8)."""
    D = _dct_matrix_t(x.device)
    b, c, h, w = x.shape
    blk = (x - 128.0).reshape(b, c, h // BS, BS, w // BS, BS).permute(0, 1, 2, 4, 3, 5)
    return D @ blk @ D.T
