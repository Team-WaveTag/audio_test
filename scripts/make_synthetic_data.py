"""스모크 테스트용 합성 데이터 생성 (실제 음성/이미지 데이터가 준비되기 전 '코드가 도는지' 확인용).

생성물: data/synth_audio/spk{0..2}/clip{0..5}.wav (합성 음성 유사 신호), data/synth_images/img{0..2}.png
※ 합성 신호이므로 이 데이터로 얻은 보호/견고성 '수치'는 의미가 없다. 실제 평가는 VoxCeleb/CMU Arctic/FSC(음성),
   ImageNet 검증셋 일부(이미지) 등 실데이터로 수행한다.
"""
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

root = Path(__file__).resolve().parents[1]
rng, sr = np.random.default_rng(0), 16000
for spk in range(3):
    d = root / "data" / "synth_audio" / f"spk{spk}"
    d.mkdir(parents=True, exist_ok=True)
    for k in range(6):
        t = np.arange(int(sr * 2.5)) / sr
        f0 = 100 + 25 * spk + 10 * np.sin(2 * np.pi * (2 + k * 0.3) * t)
        ph = 2 * np.pi * np.cumsum(f0) / sr
        x = sum((1 / h) * np.sin(h * ph) for h in range(1, 25)) * 0.1 * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2)
        sf.write(str(d / f"clip{k}.wav"), (x + 0.003 * rng.standard_normal(len(t))).astype(np.float32), sr)
d = root / "data" / "synth_images"
d.mkdir(parents=True, exist_ok=True)
yy, xx = np.mgrid[0:200, 0:200]
for k in range(3):
    img = np.stack([128 + 60 * np.sin(xx / (15.0 + k) + c) + 40 * np.cos(yy / 23.0) + rng.normal(0, 6, (200, 200))
                    for c in range(3)], -1)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(d / f"img{k}.png")
print("합성 데이터 생성 완료 ->", root / "data")
