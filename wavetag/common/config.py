"""설정 로더.

YAML 파일(configs/default.yaml)을 읽어서 `cfg.audio.removal.target_sim` 처럼
점(.) 으로 접근할 수 있는 객체로 바꿔준다.
모든 TBD(미지수) 임계값은 YAML 한 곳에서만 관리한다 → 실험으로 확정되면 YAML만 고치면 된다.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# 프로젝트 루트/configs/default.yaml
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _to_ns(obj: Any) -> Any:
    """dict -> SimpleNamespace 재귀 변환."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def ns_to_dict(obj: Any) -> Any:
    """SimpleNamespace -> dict 재귀 변환 (dataclass(**dict) 로 넘길 때 사용)."""
    if isinstance(obj, SimpleNamespace):
        return {k: ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [ns_to_dict(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None) -> SimpleNamespace:
    """YAML 설정을 읽어 SimpleNamespace 로 반환."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)
