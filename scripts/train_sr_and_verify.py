"""[모델 수준 검증] 보호 데이터로 화자인식(SR) 모델을 학습시켰을 때의 효과 측정  (논문 Table II/III 의 축소판).

측정 2가지
 (1) 학습 방해: clean 으로 학습한 모델 vs protected 로 학습한 모델의 '깨끗한 테스트 음성' 정확도(ACC) 하락
 (2) 소유권 검증(워터마크 반응): 모델에 질의 음성을 넣었을 때 '등록 화자로 판정' 되는 비율
      - clean_q : 깨끗한 테스트 음성                       (기준선)
      - noise_q : 깨끗한 음성 + 백색잡음(보호음과 같은 SNR)   (일반 교란 대조군. 논문은 MUSAN 배경음악 사용)
      - wm_q    : 보호(워터마크) 음성                       (트리거 반응)
     protected 로 학습한 모델에서만 wm_q 판정률이 높고, clean 학습 모델에서는 낮아야 '학습 여부 검증'이 성립한다.

사용 예)
  python scripts/train_sr_and_verify.py --clean_dir data/spk --protected_dir data/spk_protected --epochs 30

[주의] 여기 SR 모델(TinySR)은 CPU 에서도 돌도록 작게 만든 '점검용' 모델이다. 논문 수준(ECAPA-TDNN 등 6종 x 3 데이터셋)
재현은 SpeechBrain 레시피 + GPU 로 별도 수행한다. 결과 해석은 '경향 확인' 용으로만 사용할 것.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavetag.audio.metrics import snr_db
from wavetag.common.audio_io import fix_length, load_audio, rms
from wavetag.common.config import load_config
from wavetag.common.device import get_device


class TinySR(nn.Module):
    """log-STFT 크기 -> 소형 CNN -> 화자 분류. (torchaudio 의존성 없이 torch.stft 만 사용)"""

    def __init__(self, n_spk: int, n_fft: int = 512, hop: int = 160):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("win", torch.hann_window(n_fft))
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(64, n_spk)

    def forward(self, wav):                                    # (B,T)
        S = torch.stft(wav, self.n_fft, self.hop, window=self.win, return_complex=True).abs()
        return self.fc(self.cnn(torch.log1p(S).unsqueeze(1)))


def collect(root_clean: Path, root_prot: Path, seg: int, sr: int):
    """화자 폴더별 파일을 (train/test) 로 나눈다: 4번째 파일마다 test. 반환: 각 split 의 (clean, prot) 쌍 리스트."""
    spks = sorted(d.name for d in root_clean.iterdir() if d.is_dir())
    tr, te = [], []
    for ci, s in enumerate(spks):
        files = sorted(p.name for p in (root_prot / s).glob("*.wav"))
        for k, fn in enumerate(files):
            c = fix_length(load_audio(root_clean / s / fn, sr), seg)
            p = fix_length(load_audio(root_prot / s / fn, sr), seg)
            (te if k % 4 == 0 else tr).append((c, p, ci))
    return spks, tr, te


def train(model, X, y, epochs, device, lr=2e-3, bs=16):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X, y = torch.from_numpy(np.stack(X)), torch.tensor(y)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(model(X[idx].to(device)), y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


@torch.inference_mode()
def rate(model, X, y, device):
    model.eval()
    pred = model(torch.from_numpy(np.stack(X)).to(device)).argmax(1).cpu().numpy()
    return float((pred == np.array(y)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--protected_dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--out", default="results/sr_verify.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    seg = int(args.seconds * cfg.sr)
    spks, tr, te = collect(Path(args.clean_dir), Path(args.protected_dir), seg, cfg.sr)
    print(f"화자 {len(spks)}명 | train {len(tr)} / test {len(te)} 클립")

    yte = [t[2] for t in te]
    Xte_clean = [t[0] for t in te]
    Xte_wm = [t[1] for t in te]
    Xte_noise = []                                  # 보호음과 같은 SNR 의 백색잡음 대조군
    for c, p, _ in te:
        snr = snr_db(c, p)
        n = rng.standard_normal(len(c))
        n *= rms(c) / rms(n) * 10 ** (-snr / 20)
        Xte_noise.append((c + n).astype(np.float32))

    results = {}
    for tag, idx in (("trained_on_clean", 0), ("trained_on_protected", 1)):
        model = TinySR(len(spks)).to(device)
        train(model, [t[idx] for t in tr], [t[2] for t in tr], args.epochs, device)
        results[tag] = {
            "acc_clean_test": rate(model, Xte_clean, yte, device),   # (1) 학습 방해 지표: 낮을수록 보호 효과
            "verif_clean_q": rate(model, Xte_clean, yte, device),    # (2) 기준선
            "verif_noise_q": rate(model, Xte_noise, yte, device),    #     대조군
            "verif_wm_q": rate(model, Xte_wm, yte, device),          #     트리거 반응
        }
        print(tag, json.dumps(results[tag], indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    d = results["trained_on_clean"]["acc_clean_test"] - results["trained_on_protected"]["acc_clean_test"]
    print(f"\n정확도 하락(clean - protected) = {d:+.3f}  (+ 일수록 학습 방해 효과)")
    print("워터마크 반응(protected 모델 wm_q - clean 모델 wm_q) =",
          f"{results['trained_on_protected']['verif_wm_q'] - results['trained_on_clean']['verif_wm_q']:+.3f}")


if __name__ == "__main__":
    main()
