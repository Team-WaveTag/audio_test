"""Volto* 생성모델(G/D) 학습.  (wm_mode=star 를 쓸 때만 필요. Volto+ 만 쓰면 이 단계는 불필요)

사용 예)
  python scripts/train_volto_star.py --data_dir data/audio --out ckpt/volto_star.pt --epochs 20 --batch 8

- 학습 데이터: 보호 대상과 같은 도메인의 음성 폴더(라벨 불필요). 무작위 구간 seg_sec 초를 잘라 사용.
- 각 스텝마다 무작위 uid 의 트리거를 생성해 G/D 를 함께 학습 (uid 일반화).
- ★GPU 권장★ 전체 FFT(48,000 샘플) 위의 1D U-Net 이라 CPU 로는 매우 느리다. 노트북에서는 --epochs 1 로 동작 확인만.
- [한계/TODO] 압축·잡음 증강(augmentation)은 아직 없음 -> 견고성 개선 시 학습 루프에 MP3/잡음 증강 추가 필요.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavetag.audio.volto_plus import time_trigger
from wavetag.audio.volto_star import VoltoStar
from wavetag.common.audio_io import load_audio
from wavetag.common.config import load_config
from wavetag.common.device import get_device


class CropDataset(Dataset):
    """음성 파일에서 seg_len 샘플 무작위 크롭 + 무작위 uid 트리거를 반환."""

    def __init__(self, files, seg_len, sr, slot_ms, seed_base, n_uids=10000):
        self.files, self.seg_len, self.sr = files, seg_len, sr
        self.slot_ms, self.seed_base, self.n_uids = slot_ms, seed_base, n_uids
        self.cache = {}

    def __len__(self):
        return len(self.files) * 8                       # 에폭당 파일당 8개 크롭

    def __getitem__(self, i):
        f = self.files[i % len(self.files)]
        if f not in self.cache:
            self.cache[f] = load_audio(f, self.sr)
        x = self.cache[f]
        if len(x) < self.seg_len:
            x = np.pad(x, (0, self.seg_len - len(x)))
        s = np.random.randint(0, len(x) - self.seg_len + 1)
        uid = np.random.randint(0, self.n_uids)
        tw = time_trigger(self.seg_len, uid, self.sr, self.slot_ms, self.seed_base)
        return torch.from_numpy(x[s:s + self.seg_len].copy()), torch.from_numpy(tw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="ckpt/volto_star.pt")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    vs, vp = cfg.audio.volto_star, cfg.audio.volto_plus
    device = get_device(args.device)
    files = sorted([p for e in ("*.wav", "*.flac") for p in Path(args.data_dir).rglob(e)])
    assert files, "학습용 음성이 없습니다."
    seg_len = int(vs.seg_sec * cfg.sr)
    dl = DataLoader(CropDataset(files, seg_len, cfg.sr, vp.slot_ms, vp.seed_base),
                    batch_size=args.batch, shuffle=True, num_workers=0)   # 윈도우 호환: num_workers=0
    model = VoltoStar(vs.base, vs.depth, vs.delta_scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        agg = {"time": 0.0, "freq": 0.0, "wm": 0.0}
        for x, tw in dl:
            loss, parts = model.loss(x.to(device), tw.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in agg:
                agg[k] += parts[k] / len(dl)
        print(f"epoch {ep+1}/{args.epochs}  time={agg['time']:.5f}  freq={agg['freq']:.5f}  wm(복원)={agg['wm']:.5f}")
        torch.save(model.state_dict(), args.out)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
