"""사용자 고유 식별번호 / 옵트아웃 플래그를 바이트열로 직렬화한다.

[포맷]
- 식별번호 워터마크 payload : uid(4 byte, big-endian)
- 옵트아웃 코드 payload      : flags(1 byte) + uid(4 byte)
  flags 비트: 0x01=AI 학습 거부, 0x02=AI 합성(복제) 거부, 0x04=변형 거부
※ 실제 서비스 포맷(버전, 권리정보 등)은 IP 전략 확정 후 확장 (문서 4-4: 미해결 쟁점)
"""
from __future__ import annotations

FLAG_NO_TRAIN = 0x01
FLAG_NO_SYNTH = 0x02
FLAG_NO_MODIFY = 0x04
FLAG_ALL = FLAG_NO_TRAIN | FLAG_NO_SYNTH | FLAG_NO_MODIFY


def pack_uid(uid: int) -> bytes:
    return int(uid).to_bytes(4, "big")


def unpack_uid(b: bytes) -> int:
    return int.from_bytes(b[:4], "big")


def pack_optout(uid: int, flags: int = FLAG_ALL) -> bytes:
    return bytes([flags & 0xFF]) + pack_uid(uid)


def unpack_optout(b: bytes) -> dict:
    return {"flags": b[0], "uid": unpack_uid(b[1:5])}
