"""오류정정코드(ECC) 래퍼: Reed-Solomon.

문서(4-2 쟁점 5, 4-4)에서 요구한 "식별번호/옵트아웃 코드에 오류정정 적용"을 담당한다.
- 입력: 바이트열(payload)
- 출력: 비트 배열 (payload + 패리티, 0/1)
- RS는 '바이트 단위'로 오류를 센다. nsym 바이트의 패리티로 nsym//2 바이트까지 정정 가능.
  (예: nsym=8 -> 최대 4바이트 오류 정정)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from reedsolo import ReedSolomonError, RSCodec


class BitECC:
    def __init__(self, nsym: int = 8):
        self.nsym = nsym
        self._rs = RSCodec(nsym)

    def n_bits(self, n_payload_bytes: int) -> int:
        """payload 길이(byte)가 주어졌을 때 코드워드 총 비트 수."""
        return (n_payload_bytes + self.nsym) * 8

    def encode(self, payload: bytes) -> np.ndarray:
        """payload -> 코드워드 비트(int8, 0/1)."""
        codeword = bytes(self._rs.encode(payload))
        return np.unpackbits(np.frombuffer(codeword, dtype=np.uint8)).astype(np.int8)

    def decode(self, bits: np.ndarray) -> Tuple[Optional[bytes], int]:
        """코드워드 비트 -> (payload 또는 None, 정정된 바이트 수).

        정정 불가능하면 (None, -1) 반환. ECC 복호 성공 자체가
        "코드가 존재하고 무결하게 복원됨"의 증거가 된다.
        """
        bits = np.asarray(bits).astype(np.uint8)
        assert bits.size % 8 == 0, "비트 수는 8의 배수여야 합니다."
        codeword = np.packbits(bits).tobytes()
        try:
            decoded = self._rs.decode(codeword)
            msg = bytes(decoded[0])
            n_fixed = len(decoded[2]) if len(decoded) > 2 else -1
            return msg, n_fixed
        except ReedSolomonError:
            return None, -1
