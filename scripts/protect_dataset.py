"""데이터셋 전체를 WaveTag 로 보호해서 폴더 구조 그대로 저장한다 (모델 수준 검증의 사전 단계).

입력 구조:  data_dir/<화자폴더>/*.wav      (화자 폴더 = 클래스. 논문의 '화자 i')
사용 예)    python scripts/protect_dataset.py --data_dir data/spk --out_dir data/spk_protected --base_uid 1000
- 화자 폴더 번호 i 의 uid = base_uid + i  (논문: 클래스별 고유 워터마크)
- 시간이 오래 걸리므로 화자 수/클립 수를 줄여 먼저 시험할 것 (--max_per_speaker)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavetag.audio.factory import build_audio_stack
from wavetag.common.audio_io import load_audio, save_audio
from wavetag.common.config import load_config
from wavetag.common.device import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--base_uid", type=int, default=1000)
    ap.add_argument("--strength", default="strong")
    ap.add_argument("--max_per_speaker", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--generator", default=None)
    ap.add_argument("--dummy_encoder", action="store_true", help="스모크 테스트용")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pipe, _, _ = build_audio_stack(cfg, get_device(args.device), args.generator, dummy_encoder=args.dummy_encoder)
    root, out = Path(args.data_dir), Path(args.out_dir)
    speakers = sorted(d for d in root.iterdir() if d.is_dir())
    for i, spk in enumerate(speakers):
        files = sorted([p for e in ("*.wav", "*.flac") for p in spk.glob(e)])[: args.max_per_speaker]
        for f in files:
            x = load_audio(f, cfg.sr, cfg.audio.max_seconds)
            r = pipe.protect(x, args.base_uid + i, args.strength)
            save_audio(out / spk.name / (f.stem + ".wav"), r.audio, cfg.sr)
            print(f"[{spk.name}/{f.name}] passed={r.passed} strength={r.strength} sim={r.metrics['sim_aux']:.3f}")


if __name__ == "__main__":
    main()
