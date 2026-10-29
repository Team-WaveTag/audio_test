import numpy as np
import pytest

from wavetag.common.ecc import BitECC
from wavetag.common.payload import pack_optout, pack_uid, unpack_uid
from wavetag.image.dct_utils import band_mask, dct_blocks, idct_blocks, to_uint8
from wavetag.image.eiaw import EIAWConfig, extract_image, protect_image


def test_dct_roundtrip(natural_image):
    assert np.abs(idct_blocks(dct_blocks(natural_image)) - natural_image).max() < 1e-6


def test_bands_do_not_overlap():
    cfg = EIAWConfig()
    assert not (band_mask(*cfg.wm_band) & band_mask(*cfg.optout_band)).any()


def test_embed_extract_lossless(natural_image):
    cfg = EIAWConfig()
    out = protect_image(natural_image, 424242, cfg)["image"]
    e = extract_image(out, cfg, "wm", 4)
    assert e["ok"] and unpack_uid(e["payload"]) == 424242
    assert extract_image(out, cfg, "optout", 5)["ok"]
    # 키(p 또는 디더 시드)가 다르면 복원 불가 -- 디더가 없을 때는 p=10~64 대부분이 복호되는 결함이 있었다
    for wrong in (EIAWConfig(p=12), EIAWConfig(p=18), EIAWConfig(key=1)):
        assert not extract_image(out, wrong, "wm", 4)["ok"]


def test_torch_dct_matches_numpy(natural_image):
    torch = pytest.importorskip("torch")
    from wavetag.image.dct_utils import dct_blocks_t, idct_blocks_t
    ct = dct_blocks_t(torch.from_numpy(natural_image).permute(2, 0, 1).float().unsqueeze(0))
    assert np.abs(ct[0].numpy() - dct_blocks(natural_image)).max() < 1e-2
    assert (idct_blocks_t(ct)[0].permute(1, 2, 0).numpy() - natural_image).__abs__().max() < 1e-2


def test_adversarial_keeps_watermark(natural_image):
    """EIAW 핵심 성질: p 배수 단위 적대적 교란 후에도 워터마크가 그대로 추출된다 + 예측이 실제로 바뀐다."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from wavetag.image.models import Normalized
    torch.manual_seed(2)
    net = nn.Sequential(nn.Conv2d(3, 8, 7, 4, 3), nn.ReLU(), nn.Conv2d(8, 8, 5, 4, 2), nn.ReLU(),
                        nn.Flatten(), nn.Linear(8 * 14 * 14, 10))
    m = Normalized(net).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    cfg = EIAWConfig(adv_steps=30)
    r = protect_image(natural_image, 99, cfg, [m], torch.device("cpu"))
    assert r["info"]["adv_steps"] > 0
    e = extract_image(r["image"], cfg, "wm", 4)
    assert e["ok"] and unpack_uid(e["payload"]) == 99
    f = lambda a: int(m(torch.from_numpy(a).permute(2, 0, 1).float().div(255).unsqueeze(0)).argmax(1))
    assert f(r["image"]) != f(natural_image)
