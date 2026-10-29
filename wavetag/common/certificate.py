"""인증서 자동 발급 (문서 ⑤, 4-5).

기재 항목: 년월일시분초 / 파일명 / 기술 내용 / 성명(사업체명) / 고유 식별번호 / WaveTag 인감.
'인감'은 (인증서 내용 + 서버 비밀키) 의 SHA-256 해시로 대체했다 (위·변조 확인용).
※ 서버 비밀은 환경변수 WAVETAG_SEAL_SECRET 로 주입한다. 기본값은 개발용이므로 서비스에서 반드시 교체.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def issue_certificate(file_path: str | Path, uid: int, owner: str, tech: str, metrics: dict) -> dict:
    body = {
        "issued_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": Path(file_path).name,
        "file_sha256": sha256_file(file_path),
        "tech": tech,
        "owner": owner,
        "uid": int(uid),
        "metrics": {k: (None if v != v else v) for k, v in metrics.items()},   # NaN -> None
    }
    secret = os.environ.get("WAVETAG_SEAL_SECRET", "dev-only-secret")
    canon = json.dumps(body, sort_keys=True, ensure_ascii=False)
    body["wavetag_seal"] = hashlib.sha256((canon + secret).encode("utf-8")).hexdigest()
    return body
