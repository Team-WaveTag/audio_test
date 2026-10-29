"""공통 픽스처: 인터넷/모델 없이 돌아가는 합성 '음성 유사' 신호와 더미 인코더."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def speech_like():
    sr, rng = 16000, np.random.default_rng(0)
    t = np.arange(sr * 4) / sr
    f0 = 120 + 20 * np.sin(2 * np.pi * 3 * t)
    ph = 2 * np.pi * np.cumsum(f0) / sr
    x = sum((1 / h) * np.sin(h * ph) for h in range(1, 30)) * 0.1 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2)
    return (x + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture(scope="session")
def natural_image():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:224, 0:224]
    img = np.stack([128 + 60 * np.sin(xx / 17.0 + k) + 40 * np.cos(yy / 23.0) + rng.normal(0, 6, (224, 224))
                    for k in range(3)], -1)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)
