"""設計師風格指紋 — 接 hhh.com.tw 真實 _hdesigner + _hcase 資料。

上傳一張你喜歡的圖,跟 135 位設計師的「風格指紋」(作品平均向量) 比對,
找出最對味的前 5 位設計師。

前置:必須先跑 `python scripts/build_embeddings.py` 建好 cache。

註:之前有的「品味測驗」Tab 已移除 (與 🧬 風格 DNA Tinder 重複)。
要做品味測驗請去 dna_tinder demo。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import numpy as np
from PIL import Image

# 延遲載入 CLIP — Gradio import 時不要動 GPU
_MODEL = None
_PREPROCESS = None
_DEVICE = None
_CACHE = None        # 第一次 load 後就 memory 化,後續 reuse
_CACHE_LOADED = False  # 區分 "沒 cache 檔案" 跟 "還沒 load"

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "embeddings.sqlite"


def _ensure_model():
    global _MODEL, _PREPROCESS, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PREPROCESS, _DEVICE
    import torch
    import open_clip

    _DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[style_fingerprint] loading CLIP ViT-L/14 on {_DEVICE} ...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=_DEVICE, quick_gelu=True
    )
    model.eval()
    _MODEL, _PREPROCESS = model, preprocess
    print(f"[style_fingerprint] CLIP loaded", flush=True)
    return _MODEL, _PREPROCESS, _DEVICE


def _load_cache():
    global _CACHE, _CACHE_LOADED
    if _CACHE_LOADED:
        return _CACHE
    if not DB_PATH.exists():
        _CACHE_LOADED = True
        return None

    print(f"[style_fingerprint] loading embeddings from SQLite ...", flush=True)
    import time
    t = time.time()
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT i.hcase_img_id, i.hdesigner_id, i.url, i.embedding, d.name AS designer_name, i.case_style "
        "FROM images i JOIN designers d ON d.hdesigner_id = i.hdesigner_id"
    ).fetchall()
    con.close()
    if not rows:
        _CACHE_LOADED = True
        return None

    ids = np.array([r[0] for r in rows], dtype=np.int64)
    designer_ids = np.array([r[1] for r in rows], dtype=np.int64)
    urls = np.array([r[2] for r in rows], dtype=object)
    embs = np.stack([np.frombuffer(r[3], dtype=np.float32) for r in rows])
    designer_names = {int(r[1]): r[4] for r in rows}
    case_styles = np.array([r[5] or "" for r in rows], dtype=object)

    # 各設計師指紋 = 平均向量,再 L2 normalize
    unique_dids = np.unique(designer_ids)
    fingerprints = np.zeros((len(unique_dids), embs.shape[1]), dtype=np.float32)
    counts = np.zeros(len(unique_dids), dtype=np.int64)
    for k, did in enumerate(unique_dids):
        mask = designer_ids == did
        avg = embs[mask].mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        fingerprints[k] = avg
        counts[k] = mask.sum()

    _CACHE = {
        "image_ids": ids,
        "designer_ids": designer_ids,
        "urls": urls,
        "embs": embs,
        "case_styles": case_styles,
        "unique_dids": unique_dids,
        "fingerprints": fingerprints,
        "designer_names": designer_names,
        "counts": counts,
    }
    _CACHE_LOADED = True
    print(f"[style_fingerprint] cached {len(rows)} images / {len(unique_dids)} designers in {time.time()-t:.1f}s", flush=True)
    return _CACHE


def encode_image(img: Image.Image) -> np.ndarray:
    import torch
    model, preprocess, device = _ensure_model()
    img = img.convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype(np.float32)


def rank_designers(taste_vec: np.ndarray, cache: dict, top_k: int = 5) -> List[dict]:
    """傳一個 768-dim 已 L2-normalize 的向量,回傳 top_k 設計師 + 最像的 3 張作品圖 URL。"""
    sims = cache["fingerprints"] @ taste_vec  # (D,)
    order = np.argsort(-sims)[:top_k]
    results = []
    for rank, idx in enumerate(order, 1):
        did = int(cache["unique_dids"][idx])
        name = cache["designer_names"][did]
        sim_score = float(sims[idx])

        # 該設計師的所有圖中,跟 taste_vec 最像的 3 張
        mask = cache["designer_ids"] == did
        emb_sub = cache["embs"][mask]
        urls_sub = cache["urls"][mask]
        # emb_sub 已 normalize? — build 階段有 normalize,所以可直接點積
        per_img_sim = emb_sub @ taste_vec
        top3 = np.argsort(-per_img_sim)[:3]
        sample_urls = [str(urls_sub[t]) for t in top3]

        results.append({
            "rank": rank,
            "designer_id": did,
            "designer_name": name,
            "similarity": sim_score,
            "image_count": int(cache["counts"][idx]),
            "sample_urls": sample_urls,
        })
    return results


def format_results_md(results: List[dict]) -> str:
    if not results:
        return "(沒有結果)"
    lines = ["| 排名 | 設計師 | 相似度 | 作品數 | 最相似 3 張 |", "|---|---|---|---|---|"]
    for r in results:
        sample_imgs = " ".join(f'<img src="{u}" style="height:80px;border-radius:4px;margin:2px"/>' for u in r["sample_urls"])
        lines.append(
            f"| #{r['rank']} | **{r['designer_name']}** (id={r['designer_id']}) | {r['similarity']*100:.1f}% | {r['image_count']} | {sample_imgs} |"
        )
    return "\n".join(lines)


# ===== Tab A: 上傳圖找設計師 =====
def handle_upload(image: Image.Image):
    if image is None:
        return "請先上傳圖片"
    cache = _load_cache()
    if cache is None:
        return f"⚠ 找不到 embedding cache (`{DB_PATH}`),請先跑 `python scripts/build_embeddings.py`"
    try:
        vec = encode_image(image)
    except Exception as e:
        return f"❌ encode 失敗:{e}"
    results = rank_designers(vec, cache, top_k=5)
    return format_results_md(results)


def prewarm() -> None:
    """啟動時預先載入 CLIP 模型 + SQLite cache,避免首次互動卡頓。"""
    try:
        _load_cache()
        _ensure_model()
    except Exception as e:
        print(f"[style_fingerprint] prewarm 失敗 (不影響啟動): {e}", flush=True)


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    prewarm()
    with gr.Blocks() as demo:
        render_meta_header(
            icon="👤",
            title="設計師風格指紋 — 上傳圖找對味設計師",
            subtitle="上傳一張你喜歡的居家照片,AI 比對 135 位設計師的「風格指紋」,找出與這張圖最對味的前 5 位設計師",
            tools=[
                ("OpenCLIP ViT-L/14 (openai)", "把上傳圖轉成 768 維 embedding"),
                ("Designer Fingerprint (cached)", "135 位設計師的作品平均向量,本機 SQLite cache"),
                ("Cosine Similarity", "本機向量比對,即時排序"),
            ],
            cost="$0",
            cost_detail="完全本機,無 API 依賴",
            time="~1 秒",
            time_detail="cache 載一次後查詢即時",
            badges=["離線可用", "MPS 加速", "推薦設計師"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:上傳一張你喜歡的居家照片(從 Pinterest、IG 收藏的、自家現況都可以),AI 用 135 位設計師作品的平均向量(風格指紋)跟它比較,推薦相似度最高的前 5 位設計師。<br/>📌 想用「刷卡選圖」找設計師的話,請改去 <strong>🧬 風格 DNA Tinder</strong>(更精緻互動 + 完整品味分析)。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                in_img = gr.Image(type="pil", label="上傳你喜歡的居家照片", height=360)
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("🔍 找對味設計師", variant="primary", scale=2)
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                gr.Markdown("### 對味設計師 Top 5", elem_classes=["demo-section-title"])
                out_md = gr.Markdown()

        btn.click(handle_upload, in_img, out_md)

    return demo
