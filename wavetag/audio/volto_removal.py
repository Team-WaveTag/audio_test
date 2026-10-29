"""[Volto 1단계] 화자 특정 신호 제거 (Speaker-Specific Signal Removal).

논문 Sec.IV-A / Algorithm 1 재구현. 두 단계로 구성:

 (1) Longitudinal cropping (종방향 절삭)
     전체 스펙트럼에서 '진폭이 임계값 t 미만인' 주파수 성분을 제거.
     t 는 bisection(이분 탐색)으로 "화자 식별이 막 거부되는 경계"를 찾는다.

 (2) Lateral cropping (횡방향 절삭) - ACO(개미군집최적화)
     특정 주파수 '조합'을 골라 통째로 제거. 개별 주파수 순위만으로는 조합 효과가
     비선형이라 불안정(논문 Fig.2) → ACO 로 조합을 탐색.

[논문에 없어서 우리가 정한 부분 = '구현 가정' (논문 저자 코드가 공개되어 있지 않음)]
 A. 주파수 단위: 논문은 'frequency'라고만 쓴다. 클립 전체 FFT 의 bin 1개씩만 지우면
    200개를 지워도 화자 유사도가 거의 안 변하므로, 인접 bin 을 묶은 '좁은 대역(band_hz)'을
    1 단위로 취급하도록 했다. band_hz=None 이면 bin 1개 단위(논문 문자 그대로).
 B. 후보 풀: 진폭<t 인 단위 중 평균 진폭이 큰 상위 max_candidates 개만 사용 (계산량 제한).
 C. 초기 가중치 wt 는 min-max 정규화하여 pheromone 과 스케일을 맞춤 (논문 식(3)은 정규화 언급 없음).
 D. 선택 확률 = (alpha*wt + beta*phero) / sum  (식(3)을 확률로 정규화)
 -> 논문 결과를 그대로 재현하려면 위 A~D 가 저자 구현과 맞는지 확인(또는 저자에게 코드 요청) 필요.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .metrics import cosine, pesq_score

# embed_fn: (B,T) float32 배열 -> (B,D) 임베딩.  SpeakerEncoder.embed 를 그대로 넘기면 된다.
EmbedFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class RemovalConfig:
    # --- longitudinal (bisection) ---
    id_threshold: float = 0.6        # 코사인 유사도가 이 값 미만이면 '식별 거부' (논문: 선행연구 기준 0.6)
    bisect_init_ratio: float = 1 / 64  # 초기 t = 최대진폭 * 이 비율 (논문)
    bisect_tol_ratio: float = 1e-3   # 수렴 허용오차 (최대진폭 대비)
    bisect_max_iter: int = 14
    pesq_floor: Optional[float] = None  # 종방향 절삭 후 PESQ 가 이 값 미만이면 t 를 줄임 (None=미사용)
    # --- lateral (ACO) ---
    target_sim: float = 0.3          # ★TBD★ ACO 종료 조건: 유사도가 이 값 미만 (논문 예: 0.3)
    aco_ants: int = 5                # 논문: 5마리
    aco_iters: int = 3               # ACO 반복(세대) 수 (논문 미기재 -> 우리 가정)
    aco_max_steps: int = 200         # 개미 1마리당 최대 선택 횟수 (논문 예: 200)
    aco_top_start: int = 10          # 시작점: 영향 상위 10개 중 하나 (논문)
    aco_max_candidates: int = 512    # 후보 풀 상한 (가정 B)
    aco_alpha: float = 1.0           # 식(3) alpha: 초기 가중치 영향
    aco_beta: float = 1.0            # 식(3) beta : 페로몬 영향
    aco_decay: float = 0.9           # 식(5) r: 증발 계수
    eff_lambda: float = 1.0          # 식(4) lambda: 보호 효과 비중
    eff_phi: float = 1.0             # 식(4) phi   : 음질 비중
    freqs_per_step: int = 1          # 한 스텝에 고르는 단위 수 (논문=1. 속도가 느리면 2~4로 키움)
    band_hz: Optional[float] = 20.0  # 가정 A: 주파수 1단위의 폭(Hz). None 이면 bin 1개
    pesq_mode: str = "wb"
    seed: int = 0


def _remove(X: np.ndarray, bins: np.ndarray, n: int) -> np.ndarray:
    """스펙트럼 X 에서 bins 를 0 으로 만든 뒤 시간영역으로 복원."""
    Xm = X.copy()
    Xm[bins] = 0
    return np.fft.irfft(Xm, n=n).astype(np.float32)


# ---------------------------------------------------------------------------
# (1) Longitudinal cropping
# ---------------------------------------------------------------------------
def longitudinal_crop(x: np.ndarray, embed_fn: EmbedFn, ref_emb: np.ndarray,
                      cfg: RemovalConfig) -> Tuple[np.ndarray, float, float]:
    """진폭 < t 인 성분을 제거. t 는 '식별이 막 거부되는 경계'를 bisection 으로 탐색.

    Returns: (절삭된 음성, 최종 임계값 t, 절삭 후 유사도)
    """
    n = len(x)
    X = np.fft.rfft(x)
    amp = np.abs(X)
    m = float(amp.max())

    lo, hi = 0.0, None        # lo: 아직 '식별됨'이 확인된 t,  hi: '식별 거부'가 확인된 t
    t = m * cfg.bisect_init_ratio
    for _ in range(cfg.bisect_max_iter):
        xt = np.fft.irfft(np.where(amp < t, 0, X), n=n).astype(np.float32)
        s = float(cosine(embed_fn(xt[None])[0], ref_emb)[0])
        if s >= cfg.id_threshold:           # 아직 식별됨 -> 더 많이 지워야 함 -> t 올림
            lo = t
            t = (t + (hi if hi is not None else m)) / 2
        else:                                # 식별 거부됨 -> t 를 낮춰 음질 보존 쪽으로
            hi = t
            t = (lo + t) / 2
        if hi is not None and (hi - lo) < cfg.bisect_tol_ratio * m:
            break
    t_final = hi if hi is not None else lo

    # (선택) 음질 가드: PESQ 가 너무 낮으면 t 를 줄여 되돌린다
    x_lc = np.fft.irfft(np.where(amp < t_final, 0, X), n=n).astype(np.float32)
    if cfg.pesq_floor is not None:
        for _ in range(8):
            p = pesq_score(x, x_lc, mode=cfg.pesq_mode)
            if np.isnan(p) or p >= cfg.pesq_floor:
                break
            t_final *= 0.7
            x_lc = np.fft.irfft(np.where(amp < t_final, 0, X), n=n).astype(np.float32)

    sim = float(cosine(embed_fn(x_lc[None])[0], ref_emb)[0])
    return x_lc, float(t_final), sim


# ---------------------------------------------------------------------------
# (2) Lateral cropping (ACO)
# ---------------------------------------------------------------------------
def _build_units(amp: np.ndarray, n: int, sr: int, t: float, cfg: RemovalConfig) -> List[np.ndarray]:
    """ACO 가 선택할 '주파수 단위' 목록 생성. 각 원소 = 그 단위에 속하는 FFT bin 인덱스 배열.

    진폭 < t 인 bin 만 포함(= 논문의 '진폭이 임계 이하인 주파수만 후보' 제약).
    """
    n_bins = len(amp)
    per = 1 if cfg.band_hz is None else max(1, int(round(cfg.band_hz * n / sr)))
    units = []
    for s in range(1, n_bins, per):          # 0번(DC)은 제외
        b = np.arange(s, min(s + per, n_bins))
        b = b[amp[b] < t]
        if b.size:
            units.append(b)
    # 평균 진폭이 큰 단위(=에너지가 있어 영향이 클 가능성) 우선으로 상한 개수만 사용
    units.sort(key=lambda b: -float(amp[b].mean()))
    return units[: cfg.aco_max_candidates]


def lateral_crop_aco(x: np.ndarray, sr: int, embed_fn: EmbedFn, ref_emb: np.ndarray,
                     t: float, cfg: RemovalConfig,
                     pesq_ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
    """ACO 로 '제거할 주파수 조합'을 찾아 제거. Returns: (음성, 로그 dict).
    pesq_ref: 식(4)의 PESQ 를 잴 기준 음성. 기본은 x 이지만, 종방향 절삭이 선행된 경우
    '원본'을 넘기면 누적 음질 손실이 효율 점수에 반영된다 (파이프라인은 원본을 넘긴다)."""
    pesq_ref = x if pesq_ref is None else pesq_ref
    rng = np.random.default_rng(cfg.seed)
    n = len(x)
    X = np.fft.rfft(x)
    amp = np.abs(X)
    units = _build_units(amp, n, sr, t, cfg)
    C = len(units)
    if C == 0:
        return x, {"n_units": 0, "sim": float(cosine(embed_fn(x[None])[0], ref_emb)[0]), "removed": 0}

    def bins_of(sel: List[int]) -> np.ndarray:
        return np.concatenate([units[i] for i in sel])

    # --- 초기 가중치: 각 단위를 '혼자' 지웠을 때 유사도가 얼마나 떨어지는가 (wt = 1 - s) ---
    single_sims = np.empty(C)
    bs = 64
    for i in range(0, C, bs):
        batch = np.stack([_remove(X, units[j], n) for j in range(i, min(C, i + bs))])
        single_sims[i:i + len(batch)] = cosine(embed_fn(batch), ref_emb)
    wt = 1.0 - single_sims
    wt = (wt - wt.min()) / (wt.max() - wt.min() + 1e-12)       # 가정 C: min-max 정규화
    phero = np.ones(C)                                          # 페로몬 초기값 1 (논문)
    top = np.argsort(-wt)[: cfg.aco_top_start]

    best: Optional[Dict] = None
    history = []
    for it in range(cfg.aco_iters):
        # 개미 초기화: 영향 상위 top 중 하나에서 출발 (논문)
        ants = [{"sel": [int(rng.choice(top))], "active": True, "sim": 1.0}
                for _ in range(cfg.aco_ants)]
        for _step in range(cfg.aco_max_steps):
            active = [a for a in ants if a["active"]]
            if not active:
                break
            for a in active:
                avail = np.ones(C, bool)
                avail[a["sel"]] = False
                if not avail.any():
                    a["active"] = False          # 후보 소진 (종료 조건 3)
                    continue
                w = np.where(avail, cfg.aco_alpha * wt + cfg.aco_beta * phero, 0.0)
                p = w / w.sum()
                k = int(min(cfg.freqs_per_step, avail.sum(), (p > 0).sum()))
                a["sel"] += [int(i) for i in rng.choice(C, size=k, replace=False, p=p)]
            # 활성 개미들의 현재 해를 한 번에 평가 (배치 -> GPU 효율)
            batch = np.stack([_remove(X, bins_of(a["sel"]), n) for a in active])
            sims = cosine(embed_fn(batch), ref_emb)
            for a, s in zip(active, sims):
                a["sim"] = float(s)
                if s < cfg.target_sim:           # 종료 조건 1: 충분히 낮은 유사도
                    a["active"] = False
            # 종료 조건 2 는 for 문의 aco_max_steps 로 처리

        # --- 개미별 효율 eff = lambda*(1-s) + phi*PESQ/4.5  (식 4) ---
        for a in ants:
            xa = _remove(X, bins_of(a["sel"]), n)
            pq = pesq_score(pesq_ref, xa, sr, cfg.pesq_mode)
            a["pesq"] = 0.0 if np.isnan(pq) else pq
            a["eff"] = cfg.eff_lambda * (1 - a["sim"]) + cfg.eff_phi * a["pesq"] / 4.5
            a["ok"] = a["sim"] < cfg.target_sim
        # 목표 달성한 해를 우선, 그 안에서 eff 최대
        elite = max(ants, key=lambda a: (a["ok"], a["eff"]))
        history.append({"iter": it, "sim": elite["sim"], "pesq": elite["pesq"],
                        "eff": elite["eff"], "n_units": len(elite["sel"])})
        if best is None or (elite["ok"], elite["eff"]) > (best["ok"], best["eff"]):
            best = dict(elite)
        # --- 페로몬 증발(식 5) + 엘리트 강화(식 6) ---
        phero *= cfg.aco_decay
        phero[elite["sel"]] += elite["eff"]

    x_out = _remove(X, bins_of(best["sel"]), n)
    info = {"n_units": C, "removed_units": len(best["sel"]), "sim": best["sim"],
            "pesq": best["pesq"], "reached_target": bool(best["ok"]), "history": history}
    return x_out, info
