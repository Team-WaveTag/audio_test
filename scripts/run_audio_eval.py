"""[평가 프레임 - 음성] 보호 성능/음질/워터마크·옵트아웃 견고성 일괄 평가.

사용 예)
  python scripts/run_audio_eval.py --data_dir data/audio --out_dir results/audio --n_files 5 --uid 1001
  python scripts/run_audio_eval.py --data_dir data/audio --attacks identity,mp3_128k,resample_8k --strength strong

출력: results/audio/per_file.csv (파일×공격 단위 원자료), summary.csv (공격별 평균), protected/*.wav, certificates/*.json

측정 항목(공격 적용 후 파일 기준)
  sim_<인코더>     : 원본과의 화자 유사도(낮을수록 보호 유지). eval 인코더는 생성에 쓰지 않은 모델 -> 블랙박스
  pesq / snr_db    : 원본 대비 음질 (공격 포함 총 열화)
  optout_ok/ber    : 옵트아웃 코드 복원 여부 / ECC 전 비트오류율
  trigger_ratio    : Volto+ 트리거 검출 피크비 (wm_mode=plus)   [데이터 수준 지표 - 모델 수준 검증은 train_sr_and_verify.py]
※ 화자 유사도는 '데이터 수준 대리 지표'이다. 논문의 최종 성능(학습 방해)은 모델 학습 후 정확도로 측정한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavetag.audio.factory import build_audio_stack
from wavetag.audio.metrics import cosine, pesq_score, snr_db, stoi_score
from wavetag.audio.optout_code import detect_optout
from wavetag.audio.volto_plus import detect_volto_plus
from wavetag.common.audio_io import load_audio, save_audio
from wavetag.common.certificate import issue_certificate
from wavetag.common.config import load_config
from wavetag.common.device import get_device
from wavetag.common.payload import pack_optout
from wavetag.robustness.audio_attacks import apply_attack, get_attack_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="wav/flac 가 들어있는 폴더(하위 폴더 포함)")
    ap.add_argument("--out_dir", default="results/audio")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n_files", type=int, default=5)
    ap.add_argument("--uid", type=int, default=1001, help="사용자 고유 식별번호")
    ap.add_argument("--strength", default="strong", choices=["normal", "strong", "very_strong"])
    ap.add_argument("--attacks", default="all", help="쉼표 구분 공격 이름 또는 all")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--generator", default=None, help="Volto* 체크포인트(wm_mode=star 일 때)")
    ap.add_argument("--dummy_encoder", action="store_true", help="스모크 테스트(모델 다운로드 없이 흐름만 점검)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    print(f"[device] {device}")
    out = Path(args.out_dir)
    (out / "protected").mkdir(parents=True, exist_ok=True)
    (out / "certificates").mkdir(parents=True, exist_ok=True)

    pipe, aux, evals = build_audio_stack(cfg, device, args.generator, dummy_encoder=args.dummy_encoder)
    suite = get_attack_suite()
    names = list(suite) if args.attacks == "all" else args.attacks.split(",")
    rng = np.random.default_rng(cfg.seed)
    payload = pack_optout(args.uid)
    sr = cfg.sr

    files = sorted([p for ext in ("*.wav", "*.flac") for p in Path(args.data_dir).rglob(ext)])[: args.n_files]
    assert files, f"{args.data_dir} 에 음성 파일이 없습니다."
    rows = []
    for f in files:
        x = load_audio(f, sr, cfg.audio.max_seconds)
        t0 = time.time()
        res = pipe.protect(x, args.uid, args.strength)
        print(f"[{f.name}] 보호 완료 {time.time()-t0:.1f}s  passed={res.passed} strength={res.strength} {res.metrics}")
        # 저장 후 다시 읽기: 실제 배포 파일(PCM16) 기준으로 평가
        wav_path = out / "protected" / f"{f.stem}_wavetag.wav"
        save_audio(wav_path, res.audio, sr)
        y = load_audio(wav_path, sr)
        (out / "certificates" / f"{f.stem}.json").write_text(json.dumps(issue_certificate(
            wav_path, args.uid, "TEAM", "WaveTag-audio (Volto-based)", res.metrics), ensure_ascii=False, indent=2),
            encoding="utf-8")
        ref_embs = {n: e.embed(x[None])[0] for n, e in evals.items()}
        for an in names:
            try:
                ya = apply_attack(y, suite[an], sr, rng)
            except RuntimeError as e:          # ffmpeg 없음 등
                print(f"  - {an}: 건너뜀 ({e})")
                continue
            row = {"file": f.name, "attack": an, "strength": res.strength,
                   "pesq": pesq_score(x, ya, sr, cfg.pass_criteria.pesq_mode), "snr_db": snr_db(x, ya)}
            if an == "identity":
                row["stoi"] = stoi_score(x, ya, sr)
            for n, e in evals.items():
                row[f"sim_{n}"] = float(cosine(e.embed(ya[None])[0], ref_embs[n])[0])
            d = detect_optout(ya, pipe.optout_cfg, 5, payload)
            row.update(optout_ok=int(d["ok"]), optout_raw_ber=d["raw_ber"])
            if cfg.audio.wm_mode == "plus":
                vp = cfg.audio.volto_plus
                td = detect_volto_plus(ya, args.uid, sr, vp.slot_ms, vp.seed_base)
                row["trigger_ratio"] = td["peak_ratio"]
                row["trigger_detected"] = int(td["peak_ratio"] >= cfg.audio.trigger_detect.min_peak_ratio
                                              and abs(td["loc_error"]) <= 16)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_file.csv", index=False, encoding="utf-8-sig")
    summ = df.groupby("attack", sort=False).mean(numeric_only=True)
    summ.to_csv(out / "summary.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\n=== 공격별 평균 ===")
    print(summ.round(3).to_string())
    print(f"\n저장: {out}/per_file.csv, summary.csv")


if __name__ == "__main__":
    main()
