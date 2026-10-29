"""디바이스 선택 (GPU 있으면 cuda, 없으면 cpu). 별도의 CPU 전용 최적화는 하지 않는다."""
from __future__ import annotations

import torch


def get_device(pref: str = "auto") -> torch.device:
    """pref: 'auto' | 'cpu' | 'cuda' | 'cuda:0' ..."""
    if pref != "auto":
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
