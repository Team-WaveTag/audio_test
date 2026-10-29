# WaveTag 평가 프레임 & 견고성 테스트 코드

WaveTag(비가청 음성 워터마킹 + 적대적 노이즈 결합) 의 **평가 프레임**과 **견고성 테스트**를 위한 코드베이스.
기준 문서: 「WaveTag 기술 확정 정리 문서(도면/전체 플로우)」, 논문 Volto (IEEE TDSC 2026, Ge et al.).

> 모든 임계값/키/대역 값은 `configs/default.yaml` 한 곳에서 관리한다. **★TBD★** 표시는 실험으로 확정할 미지수이며,
> 현재 값은 "코드가 돌아가게 하는 임시값"이다(문서 방침과 동일하게 확정값으로 취급하지 말 것).

---

## 1. 스택 (확정)

| 항목 | 선택 | 이유 |
|---|---|---|
| 언어 | **Python 3.11** | pesq/speechbrain 등 휠·빌드 호환이 가장 안정적 (3.10도 가능) |
| 딥러닝 | **PyTorch 2.x** | 화자인식 사전학습 모델(SpeechBrain ECAPA-TDNN/x-vector), torchvision 분류기가 모두 PyTorch. 논문 평가 모델 6종도 PyTorch 계열 → TensorFlow 쓸 이유 없음 |
| 화자 모델 | **SpeechBrain** | `speechbrain/spkrec-ecapa-voxceleb`, `speechbrain/spkrec-xvect-voxceleb` 바로 사용 |
| 신호처리 | NumPy / SciPy | FFT, 필터, DCT (GPU 불필요) |
| 지표 | pesq, pystoi, scikit-image | PESQ·STOI·PSNR·SSIM |
| 오류정정 | reedsolo | Reed-Solomon (식별번호/옵트아웃 코드) |
| 코덱 공격 | ffmpeg | MP3/AAC/Opus 왕복 |

## 2. 가상환경 설정

```bash
# (A) conda 권장 -- ffmpeg 까지 한 번에
conda env create -f environment.yml
conda activate wavetag

# (B) venv + pip
python -m venv .venv
source .venv/bin/activate            # 윈도우: .venv\Scripts\activate
# ★GPU 없는 노트북(윈도우/리눅스)은 CPU 휠을 먼저 설치 -- 안 하면 CUDA 라이브러리(수 GB)까지 내려받는다
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# ffmpeg 는 별도 설치 필요: mac `brew install ffmpeg` / 윈도우 `winget install ffmpeg` / 우분투 `sudo apt install ffmpeg`
```

- GPU 서버에서는 CPU 휠 줄을 빼고 PyTorch 공식 안내(CUDA 버전별 명령)대로 설치.
- **윈도우**: `pesq` 는 C 확장이라 "Microsoft C++ Build Tools" 가 필요하다. 어려우면 WSL2(우분투) 사용 권장.
  pesq 가 없으면 코드는 PESQ 를 NaN 으로 두고 계속 진행한다(경고 출력).
- **윈도우**: SpeechBrain 모델 다운로드 시 심볼릭 링크 오류가 나면 "개발자 모드" 를 켜거나 관리자 권한으로 실행.
- 설치 성공 후 `pip freeze > requirements.lock.txt` 로 팀 전체 버전을 고정할 것.

## 3. 폴더 구조

```
configs/default.yaml        모든 임계값·키·대역 (TBD 포함)
wavetag/
  common/                   config, device, ecc(Reed-Solomon), payload(식별번호 직렬화), audio_io, certificate(인증서)
  audio/
    metrics.py              SNR / PESQ / STOI / 코사인 유사도
    speaker_encoder.py      SpeechBrain 화자 임베딩 (aux=생성용 / eval=평가용 분리)
    volto_removal.py        [도면 ③-A] 종방향 절삭(bisection) + 횡방향 절삭(ACO)
    volto_plus.py           [도면 ③-A] 시간영역 워터마크 + 데이터 수준 검출기
    volto_star.py           [도면 ③-A] 생성모델(U-Net G/D) 워터마크 (학습 필요)
    optout_code.py          [도면 ③-A] 고주파 DSSS 옵트아웃 코드 + RS ECC
    pipeline.py             ①~④ 전체 + ④-재처리(보안강도 자동 조정)
  image/
    dct_utils.py            8x8 블록 DCT (numpy + torch)
    modulo_embed.py         비밀키 p 모듈로 삽입/추출 (+키 디더)
    eiaw.py                 [도면 ③-B] 워터마크 + p 배수 적대적 최적화 + 옵트아웃
    models.py / metrics.py  torchvision 분류기, PSNR/SSIM
  robustness/               audio_attacks.py, image_attacks.py (공격 모음)
scripts/
  run_audio_eval.py         음성 평가 (보호 -> 공격 N종 -> 지표 집계 -> CSV)
  run_image_eval.py         이미지 평가 (화이트박스/전이 공격성공률 포함)
  protect_dataset.py        데이터셋 일괄 보호 (모델 수준 검증 사전 단계)
  train_sr_and_verify.py    보호 데이터로 SR 학습 -> 학습방해/워터마크 반응 측정 (논문 Table II/III 축소판)
  train_volto_star.py       Volto* G/D 학습
  make_synthetic_data.py    스모크 테스트용 합성 데이터
tests/                      pytest 16개
```

## 4. 오픈소스 조사 결과

| 구성요소 | 공개 코드 | 처리 |
|---|---|---|
| **Volto** (논문 본체) | 검색 시점(2026-10-27)에 저자 공식 코드 **찾지 못함** | 논문 Sec.IV + Algorithm 1/2 로 **직접 구현** (`audio/volto_*.py`). 논문에 없는 세부는 "구현 가정"으로 파일 상단에 명시 |
| **EIAW** | 공식 코드 **찾지 못함**. 유사 주제 저장소(예: "Generate-Adversarial-Examples-By-Digital-Watermarking")는 있으나 EIAW 원 구현과 같은지 **확인 못함** | 문서 설명(DCT + 모듈로 + p 배수 최적화)으로 **직접 구현** |
| 화자 임베딩 | SpeechBrain 사전학습 (ECAPA-TDNN, x-vector) | **그대로 사용** (`speaker_encoder.py`) |
| 이미지 분류기 | torchvision 사전학습 | **그대로 사용** |
| PESQ/STOI/PSNR/SSIM/RS | pesq, pystoi, scikit-image, reedsolo | **그대로 사용** |
| 비교 기준선(권장) | Meta **AudioSeal** (음성 워터마킹) | 아직 미포함. 견고성 표에 "기존 워터마킹 대비" 를 넣고 싶다면 추가 후보 |

> Volto 저자 코드가 있는지 저자(yunjiege@whu.edu.cn)에게 문의해 보는 것을 권장한다. 받으면 `volto_removal.py` 의 구현 가정 A~D 를 대조·교체한다.

## 5. 실행 순서

```bash
# 0) 테스트 (인터넷/모델 없이 가능)
pytest -q

# 1) 스모크 테스트 -- 코드 흐름만 확인 (수치 무의미)
python scripts/make_synthetic_data.py
python scripts/run_audio_eval.py --data_dir data/synth_audio --out_dir results/smoke_audio --n_files 1 --dummy_encoder
python scripts/run_image_eval.py --data_dir data/synth_images --out_dir results/smoke_image --n_files 1 --no_pretrained

# 2) 실제 평가 (SpeechBrain / torchvision 가중치를 처음에 자동 다운로드)
python scripts/run_audio_eval.py --data_dir data/vox50 --out_dir results/audio --n_files 10 --uid 1001 --strength strong
python scripts/run_image_eval.py --data_dir data/imagenet_val200 --out_dir results/image --n_files 50 --uid 1001

# 3) 모델 수준 검증 (학습 방해 + 소유권 검증)
python scripts/protect_dataset.py --data_dir data/spk --out_dir data/spk_protected --base_uid 1000
python scripts/train_sr_and_verify.py --clean_dir data/spk --protected_dir data/spk_protected --epochs 30

# 4) (선택) Volto* 사용 시: configs 에서 wm_mode: star, volto_star.ckpt 지정
python scripts/train_volto_star.py --data_dir data/vox50 --out ckpt/volto_star.pt --epochs 20
```

## 6. GPU 없는 노트북에서 돌아가나?

`get_device()` 가 cuda 가 없으면 자동으로 cpu 를 쓸 뿐, CPU 전용 최적화는 하지 않았다.

| 작업 | 노트북(CPU) | 비고 |
|---|---|---|
| `pytest`, 옵트아웃/Volto+/ECC/공격/인증서 | ✅ 즉시 | 수 초 (실측) |
| 이미지 보호·평가 (ResNet18, ≤256px, 수십 장) | ✅ 가능 | 장당 수 초~수십 초 (추정) |
| 음성 보호 파이프라인 (ACO, 5초 클립) | ⚠️ 돌아가지만 느림 | ACO 가 화자 인코더를 수백~수천 번 호출. 클립당 수 분~십수 분 (**추정, 실측 아님**). 클립 수를 줄이거나 `aco_max_candidates`/`aco_iters`/`freqs_per_step` 로 조절 |
| `train_sr_and_verify.py` (TinySR, 소규모) | ✅ 가능 | 점검용 소형 모델이라 CPU 가능 |
| **Volto\* 생성모델 학습** | ❌ 사실상 불가 | 전체 FFT(48,000점) 위 1D U-Net. CPU 는 `--epochs 1` 동작 확인만 |
| 논문 수준 재현 (SR 모델 6종 × 데이터셋 3종 학습, 수천 클립 평가) | ❌ GPU 서버 필요 | |

## 7. 코드 ↔ 문서/논문 매핑

| 도면/논문 | 코드 |
|---|---|
| ③-A 주파수 분해 + ACO 제거 (논문 Alg.1, 식 3~6) | `audio/volto_removal.py` |
| ③-A 식별번호 워터마크 (논문 Alg.2) | `audio/volto_plus.py`, `audio/volto_star.py` (식 7~9) |
| ③-A 고주파 옵트아웃 + ECC | `audio/optout_code.py`, `common/ecc.py` |
| ③-B DCT → 모듈로 → p 배수 최적화 → 옵트아웃 | `image/eiaw.py`, `image/modulo_embed.py` |
| ④ 검증 / ④-재처리 | `audio/pipeline.py` (`protect`) |
| ⑤ 인증서 | `common/certificate.py` |
| ⑦ 사후 검증 (음성: 모델 반응 / 이미지: 워터마크 추출) | `scripts/train_sr_and_verify.py` / `image.extract_image` |
| 4-2 쟁점 (1) 전이 성능 | `run_image_eval.py` 의 `asr_train` vs `asr_held` |
| 4-2 쟁점 (3)(5) JPEG 후 유지·ECC | `image_attacks.py` + `wm_ok/oo_ok/*_ber` 열 |

## 8. TBD (실험으로 확정할 값 — 전부 `configs/default.yaml`)

`pass_criteria`(sim_max / pesq_min / snr_min_db), `strength.*`(보안강도별 target_sim, wm_amp_ratio), 음성 옵트아웃 대역·세기,
`trigger_detect.min_peak_ratio`, 이미지 `p`·대역·`adv_*`, `volto_star.delta_scale`.
**PESQ 는 `wb`/`nb` 모드에 따라 스케일이 다르다.** 논문의 1.99 가 어느 모드 기준인지 원문에 없으므로, 임계값은 반드시 같은 모드로 측정해 확정할 것.

## 9. 알려진 한계 / 반드시 알아둘 점

**구현 중 발견된 사실**
1. **이미지 키 보안 결함(수정 완료)**: 나머지(mod p)만으로 비트를 기록하면, 작은 계수에서는 값이 사실상 "부호"로 비트를 나타내 **틀린 p 로도 복호**되었다(p=10~64 중 27개 전부). → 키 시드 기반 **디더**를 추가해 해결(`modulo_embed.py`). 테스트 `test_embed_extract_lossless` 가 회귀 방지.
2. **p 가 작으면 JPEG 에 바로 깨진다** (합성 이미지 1장, 디더 적용 후): p=8 → JPEG95 부터 워터마크·옵트아웃 모두 실패(PSNR 43 dB).
   p=16/24/32 → 워터마크는 JPEG75 까지 복호, 옵트아웃(중·고주파)은 p=16·24 에서 JPEG95 만 / p=32 에서 JPEG90 까지 (PSNR 37→34→31 dB).
   **경계선 결과라 이미지·시드에 따라 달라진다 → 반드시 실제 이미지 N장으로 재측정**. 화질(p↑ → PSNR↓)과 JPEG 생존의 트레이드오프, 그리고
   옵트아웃 코드가 워터마크보다 약하다는 점이 확인됨 → 문서 4-2 쟁점(3)·(5) 가 실제 위험.
3. **음성 옵트아웃 코드는 8 kHz 다운샘플/4 kHz 저역통과에서 소멸**한다(`test_optout_dies_under_8k_resample`). MP3(64~192k)/AAC/Opus/잡음/시간이동/16bit 양자화는 합성 신호에서 생존. 전화망·저대역 코덱이 위협 모델에 포함되는지 결정 필요.
4. Volto+ 트리거는 **LPC 백색화 후에만** 검출된다(음성은 유색 신호라 그냥 상관을 구하면 안 됨). 시간 이동 공격에는 위치가 어긋나 `trigger_detected=0`.

**논문/문서에 없어 우리가 가정한 부분 (검증·저자 확인 필요)**
- ACO 의 '주파수 단위'(좁은 대역 묶음, `band_hz`), 후보 풀 상한, 초기 가중치 정규화, 확률 정규화 — `volto_removal.py` 상단 A~D.
- Volto* 의 G/D 동시 학습, 스펙트럼 2채널 U-Net 구조, 3초 조각 처리.
- 이미지 DCT 를 **8x8 블록 DCT**(JPEG 방식)로 구현 — EIAW 원 논문이 전체 DCT 인지 블록 DCT 인지는 프로젝트 파일에 논문이 없어 **확인하지 못했다**.
- 적대적 최적화 규칙(그래디언트 상위 k% 계수를 ±p 이동, 조기종료)은 우리 구현. 시각 보호 효과는 **사전학습 분류기로 직접 측정 전까지 미검증**.

**테스트 범위 (솔직한 현황)**
- 샌드박스에서 `pytest` 16개 통과, 두 평가 스크립트·`protect_dataset`·`train_sr_and_verify` 를 **합성 데이터 + 더미 인코더/랜덤 모델**로 끝까지 실행 확인.
- 샌드박스에서 인터넷 제한으로 **SpeechBrain 모델 로딩, torchvision 사전학습 가중치, 실제 음성 데이터는 실행해 보지 못했다.** 첫 실행 시 `speaker_encoder.py` 의 모델 ID/임포트 경로에서 오류가 날 수 있으니 팀에서 가장 먼저 확인할 것.
- 평가의 화자 유사도·ASR 은 **데이터 수준 대리 지표**다. "학습 방해" 의 최종 증거는 `train_sr_and_verify.py`(또는 논문처럼 대형 SR 모델 학습)의 정확도 하락으로 확인한다.
- 이미지의 "학습 여부 검증"(문서 4-2 쟁점 4)은 불가능 → 이미지는 워터마크 추출(`extract_image`)로만 검증.

## 10. 팀 협업 규칙

- 함수/클래스에는 한국어 docstring, `★TBD★`·`가정`·`한계` 는 반드시 주석에 표시.
- 새 임계값은 코드에 하드코딩하지 말고 `configs/default.yaml` 에 추가.
- PR 전 `pytest -q` 통과. 평가 결과(`results/`)는 git 에 올리지 않고 CSV 요약만 공유.
