"""🧬 風格 DNA Tinder — 刷卡測你的設計品味。

精緻互動式品味測驗:
  1. 從 57,833 張 hhh 案例中,用 k-means 取 30 張**風格多樣**的代表
  2. 使用者像 Tinder 一樣刷 ❤️ / 👎 (中央大圖 + 兩個大按鈕)
  3. ❤️ 圖向量加權平均 + 👎 反向 → 個人「品味 DNA 向量」
  4. 結果頁:
      🎨 風格組成 (CLIP text-image similarity → donut chart)
      🌈 色彩偏好 (k-means on liked images → 主色卡)
      👤 對味設計師 Top 5
      🏠 對味案例 Top 12

Reuses:
  - data/embeddings/embeddings.sqlite (57k 案例 + 135 設計師)
  - open_clip ViT-L/14 (模型 + text encoder)
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import sqlite3
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np
import requests
from PIL import Image

# 中文字型
for _cjk in ["PingFang TC", "Heiti TC", "STHeiti", "Apple LiGothic", "Microsoft JhengHei"]:
    try:
        _fm.findfont(_cjk, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_cjk]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


DB_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "embeddings.sqlite"

# 風格錨點 (中文顯示, 英文 prompt 給 CLIP)
STYLE_ANCHORS_RAW = [
    ("北歐極簡", "scandinavian minimalist interior, light wood, white walls"),
    ("工業風",   "industrial loft interior, exposed brick, metal, edison bulbs"),
    ("現代奢華", "modern luxury interior, marble, gold accents, designer furniture"),
    ("日式禪風", "japanese zen interior, tatami, shoji, natural wood, minimal"),
    ("古典歐式", "classical european interior, ornate, chandelier, antique furniture"),
    ("波西米亞", "bohemian eclectic interior, colorful, mixed patterns, vintage"),
    ("鄉村風",   "rustic farmhouse interior, wood beams, vintage decor"),
    ("現代簡約", "contemporary minimalist interior, clean lines, neutral palette"),
    ("輕奢風",   "modern soft luxury interior, beige, brass, marble accent"),
    ("混搭風",   "eclectic mix style interior, contrast textures, designer"),
]

# 卡片色系 (給結果頁圓餅圖用,順序對齊上面 STYLE_ANCHORS_RAW)
STYLE_COLORS = [
    "#94A3B8",  # 北歐 - 灰
    "#7C3AED",  # 工業 - 紫
    "#D4AF37",  # 奢華 - 金
    "#84A98C",  # 日式 - 綠
    "#9F7AEA",  # 古典 - 淡紫
    "#F59E0B",  # 波西米亞 - 橘
    "#A16207",  # 鄉村 - 棕
    "#0F172A",  # 現代簡約 - 深藍
    "#E5C19F",  # 輕奢 - 米
    "#EC4899",  # 混搭 - 粉
]


# ===== 模組級單例:cache + CLIP + 卡片 =====
_CASE_CACHE = None
_CLIP_MODEL = None
_CLIP_PREP = None
_CLIP_TOK = None
_CLIP_DEVICE = None
_STYLE_EMBS = None  # (N_styles, 768)
_DECK = None  # 多樣化 30 張卡片 (urls, embs, case_ids, designer_ids, designer_names)


def _load_case_cache():
    global _CASE_CACHE
    if _CASE_CACHE is not None:
        return _CASE_CACHE
    if not DB_PATH.exists():
        return None
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT i.url, i.embedding, i.hcase_id, i.hdesigner_id, d.name "
        "FROM images i JOIN designers d ON d.hdesigner_id = i.hdesigner_id"
    ).fetchall()
    con.close()
    if not rows:
        return None
    urls = np.array([r[0] for r in rows], dtype=object)
    embs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    case_ids = np.array([r[2] for r in rows], dtype=np.int64)
    designer_ids = np.array([r[3] for r in rows], dtype=np.int64)
    designer_names = np.array([r[4] or "(未知)" for r in rows], dtype=object)

    # 計算設計師 fingerprint
    unique_dids = np.unique(designer_ids)
    fingerprints = np.zeros((len(unique_dids), embs.shape[1]), dtype=np.float32)
    fp_counts = np.zeros(len(unique_dids), dtype=np.int64)
    name_by_did: dict[int, str] = {}
    for k, did in enumerate(unique_dids):
        mask = designer_ids == did
        avg = embs[mask].mean(axis=0)
        n = np.linalg.norm(avg)
        if n > 0:
            avg = avg / n
        fingerprints[k] = avg
        fp_counts[k] = int(mask.sum())
        name_by_did[int(did)] = str(designer_names[mask][0])

    _CASE_CACHE = {
        "urls": urls,
        "embs": embs,
        "case_ids": case_ids,
        "designer_ids": designer_ids,
        "designer_names": designer_names,
        "fp_dids": unique_dids,
        "fp": fingerprints,
        "fp_counts": fp_counts,
        "name_by_did": name_by_did,
    }
    print(f"[dna_tinder] cache: {len(urls)} 圖 / {len(unique_dids)} 設計師", flush=True)
    return _CASE_CACHE


def _load_clip():
    global _CLIP_MODEL, _CLIP_PREP, _CLIP_TOK, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREP, _CLIP_TOK, _CLIP_DEVICE
    import torch, open_clip
    _CLIP_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dna_tinder] loading CLIP ViT-L/14 on {_CLIP_DEVICE}", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=_CLIP_DEVICE, quick_gelu=True,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    _CLIP_MODEL, _CLIP_PREP, _CLIP_TOK = model, preprocess, tokenizer
    return _CLIP_MODEL, _CLIP_PREP, _CLIP_TOK, _CLIP_DEVICE


def _compute_style_embs() -> np.ndarray:
    global _STYLE_EMBS
    if _STYLE_EMBS is not None:
        return _STYLE_EMBS
    import torch
    model, _, tokenizer, device = _load_clip()
    prompts = [eng for _, eng in STYLE_ANCHORS_RAW]
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    _STYLE_EMBS = emb.cpu().numpy().astype(np.float32)
    return _STYLE_EMBS


def _build_diverse_deck(n: int = 30) -> dict:
    """從 57k cache 用 k-means 找 n 個風格群,每群選代表 → 多樣化卡片組。"""
    global _DECK
    if _DECK is not None and len(_DECK["urls"]) == n:
        return _DECK
    cache = _load_case_cache()
    if cache is None:
        return None

    from sklearn.cluster import MiniBatchKMeans
    embs = cache["embs"]
    print(f"[dna_tinder] k-means clustering 57k embeddings → {n} groups ...", flush=True)
    km = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=42, batch_size=512)
    labels = km.fit_predict(embs)

    # 每群選離 centroid 最近 + 該設計師圖數中等 (避免大頭設計師重複出現太多)
    picked_idx = []
    used_designers: set[int] = set()
    for k in range(n):
        # 該群所有 idx
        in_cluster = np.where(labels == k)[0]
        if len(in_cluster) == 0:
            continue
        cluster_embs = embs[in_cluster]
        center = km.cluster_centers_[k]
        # cosine 距離(假設都 normalized)
        sims = cluster_embs @ center
        # 排序選最像 centroid 的圖,但避免已選過的設計師
        order = np.argsort(-sims)
        chosen = None
        for j in order:
            idx = in_cluster[j]
            did = int(cache["designer_ids"][idx])
            if did not in used_designers:
                chosen = idx
                used_designers.add(did)
                break
        if chosen is None:
            chosen = in_cluster[order[0]]
        picked_idx.append(int(chosen))

    picked_idx = np.array(picked_idx)
    # 隨機打亂順序避免使用者覺得卡片有順序
    rng = np.random.default_rng(seed=int(random.random() * 1e9))
    rng.shuffle(picked_idx)

    _DECK = {
        "urls": cache["urls"][picked_idx],
        "embs": cache["embs"][picked_idx],
        "case_ids": cache["case_ids"][picked_idx],
        "designer_ids": cache["designer_ids"][picked_idx],
        "designer_names": cache["designer_names"][picked_idx],
    }
    print(f"[dna_tinder] deck built: {len(_DECK['urls'])} diverse cards", flush=True)
    return _DECK


def prewarm() -> None:
    """啟動時預載入,避免使用者第一次點卡頓。"""
    try:
        _load_case_cache()
        _load_clip()
        _compute_style_embs()
        _build_diverse_deck(30)
    except Exception as e:
        print(f"[dna_tinder] prewarm fail (不致命): {e}", flush=True)


# ===== 計算品味 DNA =====
def _compute_taste(liked: list[int], disliked: list[int], deck) -> np.ndarray | None:
    if not liked:
        return None
    pos = deck["embs"][liked].mean(axis=0)
    if disliked:
        neg = deck["embs"][disliked].mean(axis=0)
        taste = pos - 0.3 * neg
    else:
        taste = pos
    n = np.linalg.norm(taste)
    if n > 0:
        taste = taste / n
    return taste.astype(np.float32)


def _style_breakdown(taste: np.ndarray) -> list[tuple[str, float]]:
    style_embs = _compute_style_embs()
    sims = style_embs @ taste  # (N_styles,)
    # softmax 溫度 0.07 讓分布更鮮明
    z = sims / 0.07
    e = np.exp(z - z.max())
    pct = e / e.sum()
    pairs = [(STYLE_ANCHORS_RAW[i][0], float(pct[i])) for i in range(len(pct))]
    pairs.sort(key=lambda x: -x[1])
    return pairs


def _palette_from_liked(liked_urls: list[str], n_colors: int = 5) -> list[tuple[int, int, int]]:
    """從喜歡的圖萃取整體主色。"""
    if not liked_urls:
        return []
    from sklearn.cluster import KMeans
    sample_pixels: list[np.ndarray] = []
    for url in liked_urls[:8]:  # 取前 8 張
        try:
            img = Image.open(io.BytesIO(requests.get(url, timeout=15).content)).convert("RGB")
            img.thumbnail((200, 200))
            sample_pixels.append(np.asarray(img).reshape(-1, 3))
        except Exception:
            continue
    if not sample_pixels:
        return []
    all_pixels = np.vstack(sample_pixels)
    # 隨機取樣加速
    if len(all_pixels) > 20000:
        idx = np.random.default_rng(42).choice(len(all_pixels), 20000, replace=False)
        all_pixels = all_pixels[idx]
    km = KMeans(n_clusters=n_colors, n_init="auto", random_state=42).fit(all_pixels)
    counts = np.bincount(km.labels_, minlength=n_colors)
    order = np.argsort(-counts)
    return [tuple(int(c) for c in km.cluster_centers_[i]) for i in order]


def _render_donut(breakdown: list[tuple[str, float]]) -> Image.Image:
    """畫風格組成 donut chart。"""
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=110)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    # 只顯示 ≥ 3% 的風格
    top = [p for p in breakdown if p[1] >= 0.03][:6]
    others_pct = sum(p[1] for p in breakdown if p[1] < 0.03)
    if others_pct > 0.01:
        top = top + [("其他", others_pct)]
    labels = [p[0] for p in top]
    sizes = [p[1] * 100 for p in top]
    # 取 STYLE_COLORS 對應,其他用灰
    name_to_color = {n: STYLE_COLORS[i] for i, (n, _) in enumerate(STYLE_ANCHORS_RAW)}
    colors = [name_to_color.get(n, "#cbd5e1") for n in labels]

    wedges, _, _ = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        pctdistance=0.78,
        labeldistance=1.10,
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10, color="#1e293b", weight="500"),
    )
    ax.text(0, 0, "Style\nDNA", ha="center", va="center",
            fontsize=14, weight="bold", color="#0f172a")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _render_palette_strip(palette: list[tuple[int, int, int]], width: int = 800, height: int = 110) -> Image.Image:
    from PIL import ImageDraw, ImageFont
    strip = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(strip)
    if not palette:
        return strip
    w = width // len(palette)
    for i, color in enumerate(palette):
        draw.rectangle([i * w, 0, (i + 1) * w, height - 28], fill=color)
        hex_str = "#{:02X}{:02X}{:02X}".format(*color)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
        except Exception:
            font = ImageFont.load_default()
        draw.text((i * w + 8, height - 24), hex_str, fill="#475569", font=font)
    return strip


# ===== Top 5 設計師 & Top 12 案例 =====
def _top_designers(taste: np.ndarray, k: int = 5) -> list[dict]:
    cache = _load_case_cache()
    sims = cache["fp"] @ taste
    order = np.argsort(-sims)[:k]
    out = []
    for j in order:
        did = int(cache["fp_dids"][j])
        # 找該設計師最相似的 1 張圖當頭像
        mask = cache["designer_ids"] == did
        d_embs = cache["embs"][mask]
        d_urls = cache["urls"][mask]
        per_img_sim = d_embs @ taste
        top_img_idx = int(np.argmax(per_img_sim))
        out.append({
            "name": cache["name_by_did"][did],
            "designer_id": did,
            "similarity": float(sims[j]),
            "image_count": int(cache["fp_counts"][j]),
            "thumb_url": str(d_urls[top_img_idx]),
        })
    return out


def _top_cases(taste: np.ndarray, k: int = 12, exclude_in_deck: bool = True) -> list[dict]:
    cache = _load_case_cache()
    deck = _build_diverse_deck(30)
    sims = cache["embs"] @ taste
    order = np.argsort(-sims)
    out = []
    deck_urls = set(str(u) for u in deck["urls"])
    for i in order:
        url = str(cache["urls"][i])
        if exclude_in_deck and url in deck_urls:
            continue
        out.append({
            "url": url,
            "designer_name": str(cache["designer_names"][i]),
            "similarity": float(sims[i]),
        })
        if len(out) >= k:
            break
    return out


# ===== Gradio handlers =====
def init_deck(state):
    """準備新一輪卡片。state 是 {liked: [int], disliked: [int], cursor: int}。"""
    deck = _build_diverse_deck(30)
    if deck is None:
        return state, None, "⚠ 找不到 cache,先跑 build_embeddings.py", gr.update(visible=False)
    state = {"liked": [], "disliked": [], "cursor": 0, "total": len(deck["urls"])}
    return state, deck["urls"][0], _progress_text(state), gr.update(visible=True)


def _progress_text(state: dict) -> str:
    cur = state["cursor"] + 1
    total = state["total"]
    n_like = len(state["liked"])
    n_dis = len(state["disliked"])
    return f"**第 {cur} / {total} 張**　·　❤️ {n_like}　·　👎 {n_dis}"


def swipe(direction: str, state: dict):
    """direction 是 'like' or 'dislike'。回傳:next image, progress, state, 結果頁是否顯示, 結果 html"""
    deck = _build_diverse_deck(30)
    if deck is None or not state:
        return None, "請重新開始", state, gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    cur = state["cursor"]
    if cur >= state["total"]:
        return None, "已結束", state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    if direction == "like":
        state["liked"].append(cur)
    else:
        state["disliked"].append(cur)
    state["cursor"] = cur + 1

    # 還有卡片 → 顯示下一張
    if state["cursor"] < state["total"]:
        next_url = deck["urls"][state["cursor"]]
        return (
            next_url,
            _progress_text(state),
            state,
            gr.update(visible=False),  # 結果頁先隱藏
            gr.update(visible=True),   # 卡片區仍顯示
            gr.update(value=None),     # donut
            gr.update(value=None),     # palette
            gr.update(value=[]),       # designers gallery
            gr.update(value=[]),       # cases gallery
        )

    # 沒卡片了 → 計算結果
    return _finalize(state, deck)


def finish_early(state: dict):
    """提前 finish (使用者已選夠了)"""
    deck = _build_diverse_deck(30)
    if not state or len(state.get("liked", [])) < 3:
        return None, "至少要 ❤️ 3 張才能解讀", state, gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    return _finalize(state, deck)


def _finalize(state: dict, deck):
    """跑完所有計算,回結果頁的所有元件。"""
    print(f"[dna_tinder] finalize  liked={len(state['liked'])}  disliked={len(state['disliked'])}", flush=True)
    taste = _compute_taste(state["liked"], state["disliked"], deck)
    if taste is None:
        return None, "請至少 ❤️ 一張圖", state, gr.update(visible=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    breakdown = _style_breakdown(taste)
    donut_img = _render_donut(breakdown)

    liked_urls = [str(u) for u in deck["urls"][state["liked"]]]
    palette = _palette_from_liked(liked_urls, n_colors=5)
    palette_img = _render_palette_strip(palette) if palette else None

    top_designers = _top_designers(taste, k=5)
    designer_gallery = [(d["thumb_url"], f"{d['name']} · 相似 {d['similarity']*100:.0f}% · {d['image_count']} 作品") for d in top_designers]

    top_cases = _top_cases(taste, k=12)
    cases_gallery = [(c["url"], f"{c['designer_name']} · {c['similarity']*100:.0f}%") for c in top_cases]

    return (
        None,                                # 不再顯示卡片
        f"🎉 完成! ❤️ {len(state['liked'])} 張 · 👎 {len(state['disliked'])} 張",
        state,
        gr.update(visible=True),             # 結果頁
        gr.update(visible=False),            # 卡片區隱藏
        donut_img,
        palette_img,
        designer_gallery,
        cases_gallery,
    )


def reset(state):
    deck = _build_diverse_deck(30)
    state = {"liked": [], "disliked": [], "cursor": 0, "total": len(deck["urls"])}
    return state, deck["urls"][0], _progress_text(state), gr.update(visible=False), gr.update(visible=True)


# ===== 精緻 CSS =====
DNA_CSS = """
.dna-card-wrap { padding: 8px; }
.dna-card-wrap img {
    border-radius: 16px !important;
    box-shadow: 0 20px 40px -12px rgba(15,23,42,0.25), 0 8px 16px -6px rgba(15,23,42,0.1) !important;
    object-fit: cover !important;
    border: 1px solid rgba(0,0,0,0.05);
}

.dna-progress { text-align: center; padding: 12px 0 4px; font-size: 15px; color: #64748b; }
.dna-progress strong { color: #0f172a; }

.dna-swipe-row > * { flex: 1; }
.dna-btn-dislike button {
    background: #fef2f2 !important; color: #b91c1c !important;
    border: 2px solid #fecaca !important;
    height: 64px !important;
    font-size: 22px !important; font-weight: 600 !important;
    border-radius: 16px !important;
    transition: all 0.15s ease !important;
}
.dna-btn-dislike button:hover {
    background: #fecaca !important; transform: scale(1.03);
    box-shadow: 0 8px 16px -4px rgba(185,28,28,0.25);
}
.dna-btn-like button {
    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%) !important;
    color: white !important; border: none !important;
    height: 64px !important;
    font-size: 22px !important; font-weight: 700 !important;
    border-radius: 16px !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 6px 12px -4px rgba(244,63,94,0.4);
}
.dna-btn-like button:hover {
    transform: scale(1.03);
    box-shadow: 0 12px 24px -6px rgba(244,63,94,0.55);
}

.dna-result {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 20px;
    padding: 28px 32px;
    color: white;
    margin: 16px 0;
}
.dna-result h2 {
    font-size: 26px !important; font-weight: 700 !important;
    margin: 0 0 6px 0 !important; color: white !important;
}
.dna-result .lead { color: #cbd5e1; font-size: 14px; }

.dna-section-title {
    font-size: 14px !important; font-weight: 700 !important;
    color: #0f172a !important;
    margin: 20px 0 10px 0 !important;
    text-transform: uppercase; letter-spacing: 0.06em;
    border-left: 4px solid #ec4899; padding-left: 10px;
}

@media (prefers-color-scheme: dark) {
    .dna-section-title { color: #f8fafc !important; }
    .dna-btn-dislike button { background: #450a0a !important; color: #fca5a5 !important; border-color: #7f1d1d !important; }
}
"""


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    prewarm()

    with gr.Blocks(css=DNA_CSS) as demo:
        render_meta_header(
            icon="🧬",
            title="風格 DNA Tinder",
            subtitle="刷 30 張案例,AI 解析你的設計品味基因 — 風格比例 / 色彩偏好 / 對味設計師 / 完美案例",
            tools=[
                ("OpenCLIP ViT-L/14 (image + text)", "對 57k 案例 + 風格錨點向量化"),
                ("MiniBatchKMeans", "從 57k cache 抽 30 個風格分散代表"),
                ("Cosine + Softmax", "風格百分比拆解 (溫度 0.07)"),
                ("scikit-learn KMeans (RGB)", "從 ❤️ 圖萃取主色卡"),
            ],
            cost="$0",
            cost_detail="完全本機 (CLIP + cache)",
            time="刷卡 30-60 秒,結果 5-10 秒",
            time_detail="結果計算受萃色 + Top 圖載入影響",
            badges=["互動測驗", "個人化", "可分享"],
        )

        state = gr.State({"liked": [], "disliked": [], "cursor": 0, "total": 30})

        # ===== Intro =====
        with gr.Column(visible=True) as intro_view:
            gr.HTML(
                """
                <div style="text-align: center; padding: 32px 24px;">
                  <div style="font-size: 56px;">🧬</div>
                  <h2 style="margin: 12px 0 8px; font-size: 26px; color: #0f172a;">準備好刷你的設計品味嗎?</h2>
                  <p style="color: #64748b; font-size: 14px; max-width: 480px; margin: 0 auto;">
                    系統從 57,833 張案例中選了 30 張代表 — 風格各不相同。<br/>
                    對每張按 ❤️ (喜歡) 或 👎 (不喜歡),完成後幫你解讀:<br/>
                    <strong>你的風格 DNA · 色彩偏好 · 對味設計師 · 完美案例</strong>
                  </p>
                </div>
                """
            )
            start_btn = gr.Button("🚀 開始刷卡", variant="primary", size="lg")

        # ===== 卡片區 =====
        with gr.Column(visible=False, elem_classes=["dna-card-wrap"]) as card_view:
            progress = gr.Markdown(elem_classes=["dna-progress"])
            card_img = gr.Image(label=None, show_label=False, height=520, type="filepath", interactive=False)
            with gr.Row(elem_classes=["dna-swipe-row"]):
                with gr.Column(elem_classes=["dna-btn-dislike"]):
                    dislike_btn = gr.Button("👎 不喜歡")
                with gr.Column(elem_classes=["dna-btn-like"]):
                    like_btn = gr.Button("❤️ 喜歡")
            with gr.Row():
                finish_btn = gr.Button("✅ 我選夠了,看結果", size="sm")
                reset_btn = gr.Button("🔄 重新開始", size="sm")

        # ===== 結果頁 =====
        with gr.Column(visible=False) as result_view:
            gr.HTML(
                """
                <div class="dna-result">
                  <h2>🎉 你的設計品味 DNA</h2>
                  <p class="lead">基於你的選擇,系統解碼出以下四個維度。可以截圖分享。</p>
                </div>
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎨 風格組成", elem_classes=["dna-section-title"])
                    donut = gr.Image(label=None, show_label=False, height=380, interactive=False, type="pil")
                with gr.Column(scale=1):
                    gr.Markdown("### 🌈 色彩偏好", elem_classes=["dna-section-title"])
                    palette = gr.Image(label=None, show_label=False, height=200, interactive=False, type="pil")

            gr.Markdown("### 👤 對味設計師 Top 5", elem_classes=["dna-section-title"])
            designers = gr.Gallery(label=None, show_label=False, columns=5, rows=1, height=240, object_fit="cover", interactive=False)

            gr.Markdown("### 🏠 對味案例 Top 12", elem_classes=["dna-section-title"])
            cases = gr.Gallery(label=None, show_label=False, columns=6, rows=2, height=380, object_fit="cover", interactive=False)

            with gr.Row():
                replay_btn = gr.Button("🔄 重新測一次", variant="primary")

        # ===== Wiring =====
        def show_first(state):
            state, first_url, progress_text, _ = init_deck(state)
            return (
                state,
                first_url,
                progress_text,
                gr.update(visible=False),  # intro
                gr.update(visible=True),   # card
            )

        start_btn.click(
            show_first,
            inputs=state,
            outputs=[state, card_img, progress, intro_view, card_view],
        )

        like_btn.click(
            lambda s: swipe("like", s),
            inputs=state,
            outputs=[card_img, progress, state, result_view, card_view, donut, palette, designers, cases],
        )
        dislike_btn.click(
            lambda s: swipe("dislike", s),
            inputs=state,
            outputs=[card_img, progress, state, result_view, card_view, donut, palette, designers, cases],
        )
        finish_btn.click(
            finish_early,
            inputs=state,
            outputs=[card_img, progress, state, result_view, card_view, donut, palette, designers, cases],
        )

        def reset_to_card(state):
            state, first_url, progress_text, _, _ = reset(state)
            return state, first_url, progress_text, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

        reset_btn.click(
            reset_to_card,
            inputs=state,
            outputs=[state, card_img, progress, result_view, card_view, intro_view],
        )
        replay_btn.click(
            reset_to_card,
            inputs=state,
            outputs=[state, card_img, progress, result_view, card_view, intro_view],
        )

    return demo
