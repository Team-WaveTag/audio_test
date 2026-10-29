"""이미지 견고성 테스트용 공격 모음. 입력/출력: uint8 HxWx3 배열.

문서 4-2 쟁점(3): "워터마크·옵트아웃 코드 모두 JPEG 압축 후 유지 여부를 실증해야 함" 에 대응.
"""
from __future__ import annotations

import io
from functools import partial
from typing import Callable, Dict

import numpy as np
from PIL import Image, ImageFilter

ImgAttack = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def identity(img, rng):
    return img


def jpeg(img, rng, quality: int = 90):
    """JPEG 왕복 (Pillow 기본: 4:2:0 크로마 서브샘플링 -> 실제 SNS 업로드와 유사)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def webp(img, rng, quality: int = 80):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="WEBP", quality=quality)
    return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


def resize_roundtrip(img, rng, scale: float = 0.5):
    """축소 후 원래 크기로 복원."""
    h, w = img.shape[:2]
    im = Image.fromarray(img)
    small = im.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BICUBIC)
    return np.array(small.resize((w, h), Image.BICUBIC))


def gaussian_noise(img, rng, sigma: float = 3.0):
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).round().astype(np.uint8)


def gaussian_blur(img, rng, radius: float = 1.0):
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)))


def get_attack_suite() -> Dict[str, ImgAttack]:
    return {
        "identity": identity,
        "jpeg_95": partial(jpeg, quality=95),
        "jpeg_90": partial(jpeg, quality=90),
        "jpeg_75": partial(jpeg, quality=75),
        "jpeg_50": partial(jpeg, quality=50),
        "webp_80": partial(webp, quality=80),
        "resize_0.75": partial(resize_roundtrip, scale=0.75),
        "resize_0.5": partial(resize_roundtrip, scale=0.5),
        "noise_s3": partial(gaussian_noise, sigma=3.0),
        "noise_s8": partial(gaussian_noise, sigma=8.0),
        "blur_r1": partial(gaussian_blur, radius=1.0),
    }
