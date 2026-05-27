"""xoops DB 連線 helper。

從 ~/github/dbmate/.env 讀 XOOPS_DATABASE_URL,避免在多處重複密碼。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pymysql
import pymysql.cursors


_DBMATE_ENV = Path.home() / "github" / "dbmate" / ".env"


def _read_url() -> Optional[str]:
    if "XOOPS_DATABASE_URL" in os.environ:
        return os.environ["XOOPS_DATABASE_URL"]
    if _DBMATE_ENV.exists():
        for line in _DBMATE_ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith("XOOPS_DATABASE_URL="):
                return line.split("=", 1)[1]
    return None


def connect() -> pymysql.Connection:
    url = _read_url()
    if not url:
        raise RuntimeError(
            "XOOPS_DATABASE_URL not set. Source from ~/github/dbmate/.env or set env var."
        )
    m = re.match(r"mysql://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)", url)
    if not m:
        raise RuntimeError(f"Bad XOOPS_DATABASE_URL format: {url[:30]}...")
    user, pw, host, port, db = m.groups()
    return pymysql.connect(
        host=host,
        user=user,
        password=pw,
        database=db,
        port=int(port),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=10,
    )
