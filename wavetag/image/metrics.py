"""이미지 화질 지표: PSNR, SSIM (문서 4-3 이미지 검증 지표)."""
from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref_u8: np.ndarray, test_u8: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref_u8, test_u8, data_range=255))


def ssim(ref_u8: np.ndarray, test_u8: np.ndarray) -> float:
    return float(structural_similarity(ref_u8, test_u8, channel_axis=2, data_range=255))
