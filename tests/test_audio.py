import shutil

import numpy as np
import pytest

from wavetag.audio.optout_code import AudioOptOutConfig, detect_optout, embed_optout
from wavetag.audio.volto_plus import apply_volto_plus, detect_volto_plus
from wavetag.audio.volto_removal import RemovalConfig, lateral_crop_aco, longitudinal_crop
from wavetag.common.payload import pack_optout
from wavetag.robustness.audio_attacks import apply_attack, get_attack_suite, to_pcm16

SR = 16000


def test_optout_roundtrip_and_wrong_key(speech_like):
    cfg, pl = AudioOptOutConfig(), pack_optout(4242)
    y = to_pcm16(embed_optout(speech_like, pl, cfg))
    assert detect_optout(y, cfg, 5, pl)["ok"]
    assert not detect_optout(y, AudioOptOutConfig(key=1), 5, pl)["ok"]      # 키 불일치
    assert not detect_optout(speech_like, cfg, 5, pl)["ok"]                  # 미삽입 원본 -> 오검출 없음


def test_optout_survives_shift_and_noise(speech_like):
    cfg, pl, rng = AudioOptOutConfig(), pack_optout(4242), np.random.default_rng(0)
    y = to_pcm16(embed_optout(speech_like, pl, cfg))
    suite = get_attack_suite()
    for name in ("shift_50ms", "noise_20dB", "quant_8bit"):
        assert detect_optout(apply_attack(y, suite[name], SR, rng), cfg, 5, pl)["ok"], name


def test_optout_dies_under_8k_resample(speech_like):
    """문서의 '압축·변환 후 유지' 주장에 대한 한계 확인: 8kHz 다운샘플은 고주파 코드를 지운다(예상된 실패)."""
    cfg, pl, rng = AudioOptOutConfig(), pack_optout(4242), np.random.default_rng(0)
    y = to_pcm16(embed_optout(speech_like, pl, cfg))
    assert not detect_optout(apply_attack(y, get_attack_suite()["resample_8k"], SR, rng), cfg, 5, pl)["ok"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg 필요")
def test_optout_survives_mp3(speech_like):
    cfg, pl, rng = AudioOptOutConfig(), pack_optout(4242), np.random.default_rng(0)
    y = to_pcm16(embed_optout(speech_like, pl, cfg))
    assert detect_optout(apply_attack(y, get_attack_suite()["mp3_128k"], SR, rng), cfg, 5, pl)["ok"]


def test_volto_plus_detect(speech_like):
    y = apply_volto_plus(speech_like, 7, SR, 0.05)
    good = detect_volto_plus(y, 7, SR)
    assert good["peak_ratio"] > 15 and good["loc_error"] == 0
    assert detect_volto_plus(speech_like, 7, SR)["peak_ratio"] < 8            # 무삽입
    assert detect_volto_plus(y, 8, SR)["peak_ratio"] < 8                      # 다른 uid


def _dummy_embed(w):
    S = np.abs(np.fft.rfft(np.atleast_2d(w), axis=1))
    return np.stack([np.log1p(S[:, b].mean(1) * 50) for b in np.array_split(np.arange(S.shape[1]), 64)], 1)


def test_removal_reduces_similarity(speech_like):
    from wavetag.audio.metrics import cosine
    x = speech_like[: SR * 2]
    cfg = RemovalConfig(aco_max_candidates=64, aco_iters=2, aco_max_steps=30, target_sim=0.5)
    ref = _dummy_embed(x)[0]
    x1, t, s1 = longitudinal_crop(x, _dummy_embed, ref, cfg)
    x2, info = lateral_crop_aco(x1, SR, _dummy_embed, ref, t, cfg, pesq_ref=x)
    assert len(x2) == len(x)
    assert float(cosine(_dummy_embed(x2)[0], ref)[0]) <= 1.0
    assert info["removed_units"] >= 1
