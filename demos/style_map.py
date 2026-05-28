"""2D 風格地圖 — 把 N 位設計師壓到平面上,可懸停 / 點選看詳情。

每個點 = 一位設計師(向量 = 該設計師所有作品圖的平均 CLIP embedding)。
距離越近 = 設計風格越相似。

前置:跑過 `python scripts/build_embeddings.py` 之後才有資料。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import io as _io

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np
from PIL import Image as _PImage

# 設定中文字型 (Mac 內建)
for _cjk in ["PingFang TC", "Heiti TC", "STHeiti", "Apple LiGothic", "Microsoft JhengHei", "Arial Unicode MS"]:
    try:
        _fm.findfont(_cjk, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_cjk]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "embeddings.sqlite"


def _load_designer_data():
    if not DB_PATH.exists():
        return None
    con = sqlite3.connect(DB_PATH)
    # 先撈設計師清單 (不碰 blob)
    rows = con.execute(
        """
        SELECT i.hdesigner_id, d.name, d.style_text, d.region, COUNT(*) AS n_imgs
        FROM images i
        JOIN designers d ON d.hdesigner_id = i.hdesigner_id
        GROUP BY i.hdesigner_id
        ORDER BY n_imgs DESC
        """
    ).fetchall()
    if len(rows) < 2:
        con.close()
        return None

    designer_ids = []
    names = []
    style_texts = []
    regions = []
    n_imgs_list = []
    fingerprints = []
    sample_urls = []

    for did, name, style_text, region, n in rows:
        emb_rows = con.execute(
            "SELECT embedding, url FROM images WHERE hdesigner_id = ?", (did,)
        ).fetchall()
        embs = np.stack([np.frombuffer(b, dtype=np.float32) for b, _ in emb_rows])
        avg = embs.mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm

        designer_ids.append(int(did))
        names.append(name or f"(無名 {did})")
        style_texts.append((style_text or "")[:60])
        regions.append(region or "")
        n_imgs_list.append(int(n))
        fingerprints.append(avg)
        sample_urls.append([u for _, u in emb_rows[:6]])
    con.close()

    return {
        "designer_ids": np.array(designer_ids),
        "names": names,
        "style_texts": style_texts,
        "regions": regions,
        "n_imgs": np.array(n_imgs_list),
        "fingerprints": np.stack(fingerprints).astype(np.float32),
        "sample_urls": sample_urls,
    }


def _compute_umap(fingerprints: np.ndarray, n_neighbors: int = 10, min_dist: float = 0.1) -> tuple[np.ndarray, str]:
    """回傳 (2D 座標, 使用的方法名)。設計師太少時退回 PCA,避免 UMAP eigenvalue 錯誤。"""
    n = len(fingerprints)
    if n < 5:
        # UMAP 需要至少 ~5 點才合理,退回 PCA
        centered = fingerprints - fingerprints.mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        coords = u[:, :2] * s[:2]
        return coords.astype(np.float32), "PCA (資料太少,UMAP 換 PCA)"

    import umap

    nn = min(n_neighbors, max(2, n - 1))
    reducer = umap.UMAP(
        n_neighbors=nn,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(fingerprints), f"UMAP (n_neighbors={nn})"


def _build_figure(data, coords) -> str:
    """生成 PNG,回傳 base64 data URL — 直接 inline 進 HTML,不經 Gradio 任何元件。"""
    import base64
    sizes = np.sqrt(np.maximum(data["n_imgs"], 1)) * 8.0
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=sizes, c="#3366cc", alpha=0.65,
        edgecolors="white", linewidths=0.8,
    )
    top_idx = np.argsort(-data["n_imgs"])[:12]
    for i in top_idx:
        ax.annotate(
            data["names"][i],
            (coords[i, 0], coords[i, 1]),
            fontsize=8, color="#333",
            xytext=(5, 5), textcoords="offset points",
        )
    ax.set_xlabel("風格軸 1", fontsize=10)
    ax.set_ylabel("風格軸 2", fontsize=10)
    ax.set_facecolor("#f5f5f5")
    ax.grid(False)
    fig.tight_layout()

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _designer_card(data, idx: int) -> tuple[str, list[str]]:
    name = data["names"][idx]
    did = int(data["designer_ids"][idx])
    n = int(data["n_imgs"][idx])
    region = data["regions"][idx]
    style = data["style_texts"][idx]
    md = f"### {name}\n\n- **ID**: {did}\n- **作品數**: {n} 張\n- **地區**: {region or '(未填)'}\n- **風格**: {style or '(未填)'}"
    sample = data["sample_urls"][idx]
    return md, sample


def build() -> gr.Blocks:
    cache_warning = None
    data = _load_designer_data()
    fig = None
    method_label = ""
    if data is None:
        cache_warning = (
            f"⚠ 找不到 embedding cache (`{DB_PATH}`)\n\n"
            "請先跑 `python scripts/build_embeddings.py` 建立 cache。"
        )
    else:
        coords, method_label = _compute_umap(data["fingerprints"])
        fig = _build_figure(data, coords)

    with gr.Blocks() as demo:
        from demos._ui import render_meta_header
        render_meta_header(
            icon="🗺️",
            title="設計師風格地圖",
            subtitle="把 135 位設計師的風格指紋壓到 2D 平面 — 距離越近代表設計風格越像,一張圖看完整公司審美全景",
            tools=[
                ("UMAP / PCA", "把 768 維 embedding 用 cosine 距離降到 2D"),
                ("matplotlib", "渲染靜態散點圖 (避開 Plotly + Gradio 6 的 Svelte 衝突)"),
                ("OpenCLIP ViT-L/14", "上游 embedding 來源 (預先 cache)"),
            ],
            cost="$0",
            cost_detail="完全本機計算",
            time="~6 秒",
            time_detail="UMAP 啟動時跑一次,之後即時切換",
            badges=["離線可用", "視覺化導向"],
        )

        if cache_warning:
            gr.Markdown(cache_warning)
            return demo

        gr.HTML(f'<div class="demo-hint">💡 <strong>怎麼讀地圖</strong>:點越大代表作品越多,距離越近代表風格越像。標出的是作品數前 12 位設計師;從右側下拉選任一位看其作品。<br/>📊 目前涵蓋 <strong>{len(data["names"])}</strong> 位設計師,降維方法 <code>{method_label}</code>。</div>')
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML(f'<img src="{fig}" style="width:100%;border-radius:12px;border:1px solid #e2e8f0;box-shadow:0 4px 12px -2px rgba(15,23,42,0.08)" alt="{len(data["names"])} 位設計師風格地圖">')
            with gr.Column(scale=1, elem_classes=["demo-output-pane"]):
                gr.Markdown("### 看設計師詳情", elem_classes=["demo-section-title"])
                picker = gr.Dropdown(
                    choices=list(data["names"]),
                    label="選一位設計師",
                    value=None,
                    interactive=True,
                )
                info_md = gr.Markdown("👉 從上方下拉選一位設計師")
                samples = gr.Gallery(label="作品縮圖", columns=3, height=300)

        name_to_idx = {n: i for i, n in enumerate(data["names"])}

        def show(name):
            if not name or name not in name_to_idx:
                return "👉 從上方下拉選一位設計師", []
            md, urls = _designer_card(data, name_to_idx[name])
            return md, urls

        picker.change(show, picker, [info_md, samples])

    return demo
