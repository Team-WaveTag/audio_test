"""[Volto 2단계 - 주파수영역 방식 'Volto*'] 생성모델(U-Net G/D) 기반 워터마크.

논문 Sec.IV-B-2 / Fig.3 / 식(7)~(9) 재구현 (논문에 구조 세부가 없어 아래는 '구현 가정'):
  G : 시간영역 워터마크 tw -> FFT -> U-Net -> '주파수 영역 전역 표현' G(tw)
      보호 스펙트럼 = U(x) + G(tw)  (residual 덧셈)  -> iFFT -> 보호 음성
  D : 보호 음성 -> FFT -> U-Net -> iFFT -> tw 복원(워터마크 추출; 논문 'exact location 포함')
  손실:  식(7) 시간영역 유사도 L(x_wm, x)
         식(8) 주파수영역 유사도 L(U(x)+G(tw), U(x))
         식(9) 워터마크 복원 손실 L(D(x_wm), tw)
  [가정] G 와 D 를 하나의 옵티마이저로 '동시 학습'한다 (논문은 GAN 유사 구조라고만 기술).
  [가정] 스펙트럼은 정규직교(ortho) FFT 의 실수/허수 2채널 1D 신호로 U-Net 에 입력.
  [가정] 오디오를 고정 길이 seg_len 조각으로 나눠 처리(길이가 다르면 패딩 후 조각별 처리).
학습은 scripts/train_volto_star.py. ※ 학습 데이터/GPU 가 필요 -> 노트북 CPU 로는 소규모 점검만 권장.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .volto_plus import time_trigger


class UNet1D(nn.Module):
    """1D U-Net. 입력 (B, in_ch, L) -> 출력 (B, out_ch, L). L 은 임의 길이(내부에서 패딩/크롭)."""

    def __init__(self, in_ch: int = 2, out_ch: int = 2, base: int = 16, depth: int = 4):
        super().__init__()
        self.depth = depth
        ch = [min(base * 2 ** i, 128) for i in range(depth + 1)]
        self.inc = nn.Sequential(nn.Conv1d(in_ch, ch[0], 5, padding=2), nn.LeakyReLU(0.2))
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv1d(ch[i], ch[i + 1], 4, stride=2, padding=1), nn.LeakyReLU(0.2))
            for i in range(depth)])
        self.up = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose1d(ch[depth - i], ch[depth - i - 1], 4, stride=2, padding=1),
                          nn.LeakyReLU(0.2)) for i in range(depth)])
        self.fuse = nn.ModuleList([
            nn.Sequential(nn.Conv1d(2 * ch[depth - i - 1], ch[depth - i - 1], 5, padding=2), nn.LeakyReLU(0.2))
            for i in range(depth)])
        self.outc = nn.Conv1d(ch[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]
        x = F.pad(x, (0, (-L) % (2 ** self.depth)))        # 다운샘플 가능한 길이로 패딩
        h = self.inc(x)
        skips = [h]
        for e in self.enc:
            h = e(h)
            skips.append(h)
        h = skips.pop()                                     # bottleneck
        for up, fuse in zip(self.up, self.fuse):
            h = up(h)
            h = fuse(torch.cat([h, skips.pop()], dim=1))
        return self.outc(h)[..., :L]


def to_spec(x: torch.Tensor) -> torch.Tensor:
    """(B,L) 시간파형 -> (B,2,F) [실수,허수] 정규직교 rFFT."""
    X = torch.fft.rfft(x, dim=-1, norm="ortho")
    return torch.stack([X.real, X.imag], dim=1)


def from_spec(S: torch.Tensor, L: int) -> torch.Tensor:
    """(B,2,F) -> (B,L) 시간파형."""
    return torch.fft.irfft(torch.complex(S[:, 0], S[:, 1]), n=L, dim=-1, norm="ortho")


class VoltoStar(nn.Module):
    def __init__(self, base: int = 16, depth: int = 4, delta_scale: float = 0.05):
        super().__init__()
        self.G = UNet1D(2, 2, base, depth)
        self.D = UNet1D(2, 2, base, depth)
        self.delta_scale = delta_scale   # ★TBD★ 워터마크 진폭 상한(호스트 RMS 대비 비율)

    def embed(self, x: torch.Tensor, tw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x,tw:(B,L) -> (보호 파형 (B,L), 보호 스펙트럼 (B,2,F), 원본 스펙트럼)"""
        L = x.shape[-1]
        Sx = to_spec(x)
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt().unsqueeze(1) + 1e-6      # (B,1,1)
        delta = self.G(to_spec(tw)) * self.delta_scale * rms                      # 식(8)의 G(tw)
        Sw = Sx + delta
        return from_spec(Sw, L), Sw, Sx

    def extract(self, x_wm: torch.Tensor) -> torch.Tensor:
        """보호 파형 -> 복원된 tw 추정 (B,L)"""
        L = x_wm.shape[-1]
        return from_spec(self.D(to_spec(x_wm)), L)

    def loss(self, x: torch.Tensor, tw: torch.Tensor, w_time=1.0, w_freq=1.0, w_wm=1.0):
        x_wm, Sw, Sx = self.embed(x, tw)
        l_time = F.l1_loss(x_wm, x)                      # 식(7)
        l_freq = F.l1_loss(Sw, Sx)                       # 식(8)
        tw_hat = self.extract(x_wm)
        l_wm = F.l1_loss(tw_hat, tw)                     # 식(9)
        total = w_time * l_time + w_freq * l_freq + w_wm * l_wm
        return total, {"time": l_time.item(), "freq": l_freq.item(), "wm": l_wm.item()}


# ----------------------------------------------------------------------------- 추론용 헬퍼
def _chunks(x: np.ndarray, seg_len: int) -> Tuple[np.ndarray, int]:
    n = len(x)
    pad = (-n) % seg_len
    xp = np.pad(x, (0, pad)) if pad else x
    return xp.reshape(-1, seg_len), n


@torch.inference_mode()
def star_embed(model: VoltoStar, x: np.ndarray, uid: int, sr: int, seg_len: int,
               slot_ms: float, seed_base: int, device: torch.device) -> np.ndarray:
    """임의 길이 음성 x 에 Volto* 워터마크 삽입 (조각별 처리)."""
    model.eval()
    chunks, n = _chunks(x.astype(np.float32), seg_len)
    tw = time_trigger(seg_len, uid, sr, slot_ms, seed_base)
    xb = torch.from_numpy(chunks).to(device)
    tb = torch.from_numpy(tw).unsqueeze(0).repeat(len(chunks), 1).to(device)
    out, _, _ = model.embed(xb, tb)
    return out.cpu().numpy().reshape(-1)[:n].astype(np.float32)


@torch.inference_mode()
def star_detect(model: VoltoStar, x: np.ndarray, uid: int, sr: int, seg_len: int,
                slot_ms: float, seed_base: int, device: torch.device) -> float:
    """D 로 tw 를 복원해 기대 tw 와의 정규화 상관(평균)을 반환. 높을수록 워터마크가 남아있음."""
    model.eval()
    chunks, _ = _chunks(x.astype(np.float32), seg_len)
    tw = torch.from_numpy(time_trigger(seg_len, uid, sr, slot_ms, seed_base)).to(device)
    hat = model.extract(torch.from_numpy(chunks).to(device))
    num = (hat * tw).sum(-1)
    den = hat.norm(dim=-1) * tw.norm() + 1e-9
    return float((num / den).mean().item())
