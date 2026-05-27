"""為所有設計師作品圖計算 CLIP ViT-L/14 embedding,存進 SQLite。

特性:
  - 可中斷續跑 (checkpoint 在 SQLite 自身)
  - 串流下載 (不存圖片到 disk,只存 embedding)
  - MPS / CUDA / CPU 自動選
  - 壞圖跳過,記在 errors 表

用法:
  python scripts/build_embeddings.py                # 跑全部
  python scripts/build_embeddings.py --limit 3      # 只跑 3 個設計師 (用於驗證)
  python scripts/build_embeddings.py --batch 16     # 一次 batch 多少 (預設 8)
  python scripts/build_embeddings.py --workers 8    # 同時下載幾張 (預設 8)
"""

from __future__ import annotations

import argparse
import io
import logging
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open_clip
import requests
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

# 容忍部分截斷的 JPEG
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 把 project root 加進 path 才能 import db_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from db_utils import connect as db_connect  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "embeddings.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS designers (
  hdesigner_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  style_text TEXT,
  region TEXT,
  img_path TEXT
);

CREATE TABLE IF NOT EXISTS images (
  hcase_img_id INTEGER PRIMARY KEY,
  hcase_id     INTEGER NOT NULL,
  hdesigner_id INTEGER NOT NULL,
  url          TEXT NOT NULL,
  embedding    BLOB NOT NULL,   -- np.float32 768-dim
  case_style   TEXT,
  case_title   TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_designer ON images(hdesigner_id);

CREATE TABLE IF NOT EXISTS errors (
  hcase_img_id INTEGER PRIMARY KEY,
  url TEXT,
  reason TEXT,
  ts   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS build_meta (
  k TEXT PRIMARY KEY,
  v TEXT
);
"""


def open_cache() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    return con


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(device: str):
    log.info(f"loading CLIP ViT-L/14 (openai) on {device} ...")
    # quick_gelu=True 對齊 openai pretrained 權重訓練時的活化函數
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14",
        pretrained="openai",
        device=device,
        quick_gelu=True,
    )
    model.eval()
    return model, preprocess


# ---------- 一次性匯入設計師基本資料 ----------
def sync_designers(cache: sqlite3.Connection, limit: int | None) -> List[int]:
    src = db_connect()
    with src.cursor() as cur:
        sql = "SELECT hdesigner_id, name, style, region, img_path FROM _hdesigner WHERE onoff=1 ORDER BY dorder, hdesigner_id"
        if limit:
            sql += f" LIMIT {int(limit)}"
        cur.execute(sql)
        rows = cur.fetchall()
    src.close()

    cache.executemany(
        "INSERT OR REPLACE INTO designers (hdesigner_id, name, style_text, region, img_path) VALUES (?, ?, ?, ?, ?)",
        [(r["hdesigner_id"], r["name"], r["style"], r["region"], r["img_path"]) for r in rows],
    )
    cache.commit()
    log.info(f"synced {len(rows)} designers")
    return [r["hdesigner_id"] for r in rows]


# ---------- 取「該設計師還沒做 embedding 的圖」 ----------
def fetch_pending_images(designer_ids: List[int], cache: sqlite3.Connection) -> List[dict]:
    if not designer_ids:
        return []
    src = db_connect()
    with src.cursor() as cur:
        # 一次拉所有需要的設計師圖,然後在 Python 端過濾掉已做的 (SQLite 不在同連線)
        placeholders = ",".join(["%s"] * len(designer_ids))
        cur.execute(
            f"""
            SELECT i.hcase_img_id, i.hcase_id, i.name AS url,
                   c.hdesigner_id, c.style AS case_style, c.caption AS case_title
            FROM _hcase_img i
            JOIN _hcase c ON c.hcase_id = i.hcase_id
            WHERE c.onoff = 1
              AND c.hdesigner_id IN ({placeholders})
              AND i.name LIKE 'http%%'
            """,
            tuple(designer_ids),
        )
        rows = cur.fetchall()
    src.close()

    done = {r[0] for r in cache.execute("SELECT hcase_img_id FROM images").fetchall()}
    failed = {r[0] for r in cache.execute("SELECT hcase_img_id FROM errors").fetchall()}
    pending = [r for r in rows if r["hcase_img_id"] not in done and r["hcase_img_id"] not in failed]
    log.info(f"images total={len(rows)}  done={len(done)}  failed_prev={len(failed)}  pending={len(pending)}")
    return pending


# ---------- 下載 ----------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "hhh-ai-template/1.0"})


def download(url: str, timeout: int = 12) -> Image.Image:
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img


def download_batch(rows: List[dict], workers: int) -> List[Tuple[dict, Image.Image | None, str | None]]:
    """並行下載,回傳 [(row, img|None, error|None), ...]"""
    out: List[Tuple[dict, Image.Image | None, str | None]] = [None] * len(rows)  # type: ignore
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut2idx = {ex.submit(download, r["url"]): i for i, r in enumerate(rows)}
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            try:
                out[i] = (rows[i], fut.result(), None)
            except Exception as e:
                out[i] = (rows[i], None, str(e)[:200])
    return out


# ---------- 主迴圈 ----------
def run(limit: int | None, batch: int, workers: int, save_every: int) -> None:
    device = pick_device()
    cache = open_cache()
    designer_ids = sync_designers(cache, limit)
    pending = fetch_pending_images(designer_ids, cache)
    if not pending:
        log.info("nothing to do")
        return

    model, preprocess = load_model(device)
    pbar = tqdm(total=len(pending), unit="img", smoothing=0.05)
    buf_emb: List[Tuple[int, int, int, str, bytes, str, str]] = []
    buf_err: List[Tuple[int, str, str]] = []

    def flush() -> None:
        if buf_emb:
            cache.executemany(
                "INSERT OR REPLACE INTO images (hcase_img_id, hcase_id, hdesigner_id, url, embedding, case_style, case_title) VALUES (?, ?, ?, ?, ?, ?, ?)",
                buf_emb,
            )
        if buf_err:
            cache.executemany("INSERT OR REPLACE INTO errors (hcase_img_id, url, reason) VALUES (?, ?, ?)", buf_err)
        cache.commit()
        buf_emb.clear()
        buf_err.clear()

    t0 = time.time()
    for i in range(0, len(pending), batch):
        chunk = pending[i : i + batch]
        results = download_batch(chunk, workers=workers)

        good_rows: List[dict] = []
        good_imgs: List[Image.Image] = []
        for row, img, err in results:
            if img is None:
                buf_err.append((row["hcase_img_id"], row["url"], err or "unknown"))
            else:
                good_rows.append(row)
                good_imgs.append(img)

        if good_imgs:
            tensors = torch.stack([preprocess(im) for im in good_imgs]).to(device)
            with torch.no_grad():
                emb = model.encode_image(tensors)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            emb_np = emb.cpu().numpy().astype(np.float32)
            for j, row in enumerate(good_rows):
                buf_emb.append(
                    (
                        row["hcase_img_id"],
                        row["hcase_id"],
                        row["hdesigner_id"],
                        row["url"],
                        emb_np[j].tobytes(),
                        row.get("case_style"),
                        row.get("case_title"),
                    )
                )

        pbar.update(len(chunk))

        if len(buf_emb) + len(buf_err) >= save_every:
            flush()

    flush()
    pbar.close()
    cache.executemany(
        "INSERT OR REPLACE INTO build_meta (k, v) VALUES (?, ?)",
        [("model", "ViT-L-14/openai"), ("dim", "768"), ("last_run", str(int(time.time())))],
    )
    cache.commit()

    elapsed = time.time() - t0
    log.info(f"done in {elapsed:.0f}s  ({len(pending) / max(elapsed, 1):.1f} img/s)")

    n_done = cache.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    n_err = cache.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
    log.info(f"cache: {DB_PATH}  embeddings={n_done}  errors={n_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只處理前 N 個設計師 (debug 用)")
    ap.add_argument("--batch", type=int, default=8, help="GPU batch size (預設 8)")
    ap.add_argument("--workers", type=int, default=8, help="同時下載執行緒數")
    ap.add_argument("--save-every", type=int, default=64, help="每 N 筆寫入 SQLite")
    args = ap.parse_args()
    run(args.limit, args.batch, args.workers, args.save_every)


if __name__ == "__main__":
    main()
