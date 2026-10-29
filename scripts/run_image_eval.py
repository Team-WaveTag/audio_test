"""[평가 프레임 - 이미지] 화질 / 적대적 효과(화이트박스+전이) / 워터마크·옵트아웃 견고성 평가.

사용 예)
  python scripts/run_image_eval.py --data_dir data/images --out_dir results/image --n_files 20 --uid 1001
  python scripts/run_image_eval.py --data_dir data/images --strength strong --attacks identity,jpeg_90,jpeg_75

측정 항목(공격 적용 후)
  psnr/ssim            : 원본 대비 화질 (보호 직후 + 공격 후)
  asr_train / asr_held : 공격성공률 = 클린 이미지 예측과 '다른' 예측이 나온 비율
                         train=최적화에 쓴 모델(화이트박스), held=쓰지 않은 모델(전이)  -> 문서 4-2 쟁점(1)
  wm_ok/wm_ber         : 식별번호 ECC 복호 성공 / ECC 전 비트오류율
  oo_ok/oo_ber         : 옵트아웃 코드 동일 지표                                         -> 쟁점(3),(5)
※ 첫 실행 시 torchvision 사전학습 가중치를 자동 다운로드한다(인터넷 필요).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wavetag.common.config import load_config, ns_to_dict
from wavetag.common.device import get_device
from wavetag.common.ecc import BitECC
from wavetag.common.payload import pack_optout, pack_uid
from wavetag.image.eiaw import EIAWConfig, extract_image, protect_image
from wavetag.image.metrics import psnr, ssim
from wavetag.image.models import load_classifiers, predict
from wavetag.robustness.image_attacks import get_attack_suite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="results/image")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n_files", type=int, default=20)
    ap.add_argument("--uid", type=int, default=1001)
    ap.add_argument("--strength", default="strong", choices=["normal", "strong", "very_strong"])
    ap.add_argument("--attacks", default="all")
    ap.add_argument("--max_side", type=int, default=256, help="긴 변이 이 값을 넘으면 축소 (연산량 제한)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_pretrained", action="store_true", help="스모크 테스트: 사전학습 가중치 다운로드 없이 랜덤 초기화 모델 사용(수치 무의미)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    print(f"[device] {device}")
    out = Path(args.out_dir)
    (out / "protected").mkdir(parents=True, exist_ok=True)

    d = ns_to_dict(cfg.image.eiaw)
    d.update(ns_to_dict(getattr(cfg.image.strength, args.strength)))      # 보안강도 덮어쓰기
    d["wm_band"], d["optout_band"] = tuple(d["wm_band"]), tuple(d["optout_band"])
    ecfg = EIAWConfig(**d)

    train_models = load_classifiers(cfg.image.models_train, device, not args.no_pretrained)
    held_models = load_classifiers(cfg.image.models_heldout, device, not args.no_pretrained)
    suite = get_attack_suite()
    names = list(suite) if args.attacks == "all" else args.attacks.split(",")
    rng = np.random.default_rng(cfg.seed)
    ecc = BitECC(ecfg.nsym)
    wm_truth, oo_truth = ecc.encode(pack_uid(args.uid)), ecc.encode(pack_optout(args.uid))

    files = sorted([p for e in ("*.png", "*.jpg", "*.jpeg") for p in Path(args.data_dir).rglob(e)])[: args.n_files]
    assert files, f"{args.data_dir} 에 이미지가 없습니다."
    rows = []
    for f in files:
        im = Image.open(f).convert("RGB")
        if max(im.size) > args.max_side:
            s = args.max_side / max(im.size)
            im = im.resize((int(im.width * s), int(im.height * s)), Image.BICUBIC)
        img = np.array(im)
        r = protect_image(img, args.uid, ecfg, train_models, device)
        prot = r["image"]
        Image.fromarray(prot).save(out / "protected" / f"{f.stem}_wavetag.png")   # 무손실 저장
        clean_pred = {n: predict(m, img, device) for n, m in
                      zip(cfg.image.models_train + cfg.image.models_heldout, train_models + held_models)}
        print(f"[{f.name}] adv_steps={r['info']['adv_steps']} fooled(양자화 전)={r['info']['fooled']} "
              f"PSNR={psnr(img, prot):.2f}")
        for an in names:
            ya = suite[an](prot, rng)
            row = {"file": f.name, "attack": an, "psnr": psnr(img, ya) if ya.shape == img.shape else np.nan,
                   "ssim": ssim(img, ya) if ya.shape == img.shape else np.nan,
                   "adv_steps": r["info"]["adv_steps"]}
            for tag, names_m, models in (("train", cfg.image.models_train, train_models),
                                         ("held", cfg.image.models_heldout, held_models)):
                row[f"asr_{tag}"] = float(np.mean([predict(m, ya, device) != clean_pred[n]
                                                   for n, m in zip(names_m, models)]))
            e1 = extract_image(ya, ecfg, "wm", 4)
            e2 = extract_image(ya, ecfg, "optout", 5)
            row.update(wm_ok=int(e1["ok"]), wm_ber=float(np.mean(e1["bits"] != wm_truth)),
                       oo_ok=int(e2["ok"]), oo_ber=float(np.mean(e2["bits"] != oo_truth)))
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_file.csv", index=False, encoding="utf-8-sig")
    summ = df.groupby("attack", sort=False).mean(numeric_only=True)
    summ.to_csv(out / "summary.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("\n=== 공격별 평균 ===")
    print(summ.round(3).to_string())


if __name__ == "__main__":
    main()
