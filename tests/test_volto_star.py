import pytest

torch = pytest.importorskip("torch")


def test_volto_star_forward_backward():
    from wavetag.audio.volto_star import VoltoStar
    m = VoltoStar(base=8, depth=4)
    x, tw = torch.randn(2, 8000) * 0.1, torch.randn(2, 8000)
    loss, parts = m.loss(x, tw)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    assert set(parts) == {"time", "freq", "wm"}


def test_star_embed_keeps_length_and_small_delta():
    import numpy as np
    from wavetag.audio.volto_star import VoltoStar, star_embed
    m = VoltoStar(base=8, depth=4, delta_scale=0.05)
    x = (np.random.default_rng(0).standard_normal(20000) * 0.1).astype("float32")
    y = star_embed(m, x, 3, 16000, 8000, 100, 1234, torch.device("cpu"))
    assert y.shape == x.shape and abs(y - x).mean() < 0.2 * abs(x).mean()
