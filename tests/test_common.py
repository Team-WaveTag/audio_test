import numpy as np

from wavetag.common.ecc import BitECC
from wavetag.common.payload import FLAG_ALL, pack_optout, unpack_optout


def test_ecc_roundtrip_and_correction():
    ecc = BitECC(nsym=8)
    payload = pack_optout(123456, FLAG_ALL)
    bits = ecc.encode(payload)
    assert ecc.decode(bits)[0] == payload
    bad = bits.copy()
    for byte in (0, 3, 7):                    # 서로 다른 3바이트 손상 (정정 한계 4바이트 이내)
        bad[byte * 8] ^= 1
    out, n_fixed = ecc.decode(bad)
    assert out == payload and n_fixed == 3


def test_ecc_fails_when_too_corrupted():
    ecc = BitECC(nsym=8)
    bits = ecc.encode(pack_optout(1))
    bad = bits.copy()
    bad[::8] ^= 1                              # 모든 바이트 손상
    assert ecc.decode(bad)[0] is None


def test_payload_unpack():
    assert unpack_optout(pack_optout(77, 5)) == {"flags": 5, "uid": 77}
