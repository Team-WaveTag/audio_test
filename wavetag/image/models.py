"""이미지 분류 모델 로더 (적대적 최적화 '공격 대상' 및 '전이(transfer) 평가' 용).

문서 4-2 쟁점(1): EIAW 는 화이트박스 방식이라 다른 모델로 옮기면 공격 성공률이 58.9~72.8% 로 떨어진다.
-> (a) 여러 모델을 '앙상블'로 함께 최적화하고, (b) 최적화에 쓰지 않은 '보류(held-out) 모델'로 평가한다.
   models_train = ["resnet18", "resnet50"],  models_heldout = ["mobilenet_v3_large", "vgg16"] 처럼 분리 사용.
※ 가중치는 최초 실행 시 torchvision 이 자동 다운로드한다(인터넷 필요).
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class Normalized(nn.Module):
    """입력: (B,3,H,W) 값 0~1.  내부에서 (필요 시) 224x224 로 리사이즈 + ImageNet 정규화.
    리사이즈도 미분 가능하므로 임의 해상도 이미지에 대해 그래디언트가 흐른다."""

    def __init__(self, net: nn.Module, input_size: int = 224):
        super().__init__()
        self.net = net
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return self.net((x - self.mean) / self.std)


def load_classifier(name: str, device: torch.device, pretrained: bool = True) -> nn.Module:
    """torchvision ImageNet 사전학습 분류기 로드. name 예: resnet18, resnet50, vgg16, mobilenet_v3_large"""
    ctor = getattr(torchvision.models, name)
    net = ctor(weights="DEFAULT" if pretrained else None)
    model = Normalized(net).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_classifiers(names: List[str], device: torch.device, pretrained: bool = True) -> List[nn.Module]:
    return [load_classifier(n, device, pretrained) for n in names]


def img_to_tensor(img_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """HxWx3 uint8 -> (1,3,H,W) float 0~1"""
    return torch.from_numpy(img_u8).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


@torch.inference_mode()
def predict(model: nn.Module, img_u8: np.ndarray, device: torch.device) -> int:
    return int(model(img_to_tensor(img_u8, device)).argmax(1).item())
