"""平價/類似替代推薦 — DINO 抓物件 → 裁框 → CLIP 找案例庫類似圖。

對每個偵測到的物件,從 57k 案例 cache 找最像的案例圖當參考。

設計考量:
  - 完整 case image cache 已在 data/embeddings/embeddings.sqlite,reuse
  - 同樣 CLIP ViT-L/14 確保 embedding space 一致
  - 裁框後 resize 到 224x224 才 embed,避免 aspect ratio 失真
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

# === 共用單例:CLIP 模型 + case cache ===
_CLIP_MODEL = None
_CLIP_PREP = None
_CLIP_DEVICE = None
_CASE_CACHE = None

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "embeddings.sqlite"


def _load_clip():
    global _CLIP_MODEL, _CLIP_PREP, _CLIP_DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREP, _CLIP_DEVICE
    import torch, open_clip
    _CLIP_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[alternatives] loading CLIP ViT-L/14 on {_CLIP_DEVICE} ...", flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=_CLIP_DEVICE, quick_gelu=True
    )
    model.eval()
    _CLIP_MODEL, _CLIP_PREP = model, preprocess
    print("[alternatives] CLIP ready", flush=True)
    return _CLIP_MODEL, _CLIP_PREP, _CLIP_DEVICE


def _load_case_cache():
    global _CASE_CACHE
    if _CASE_CACHE is not None:
        return _CASE_CACHE
    if not DB_PATH.exists():
        return None
    print("[alternatives] loading case embedding cache from SQLite ...", flush=True)
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT i.url, i.embedding, d.name AS designer_name, i.hdesigner_id, i.hcase_id "
        "FROM images i JOIN designers d ON d.hdesigner_id = i.hdesigner_id"
    ).fetchall()
    con.close()
    if not rows:
        return None
    urls = np.array([r[0] for r in rows], dtype=object)
    embs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    designer_names = np.array([r[2] for r in rows], dtype=object)
    designer_ids = np.array([r[3] for r in rows], dtype=np.int64)
    case_ids = np.array([r[4] for r in rows], dtype=np.int64)
    _CASE_CACHE = {
        "urls": urls,
        "embs": embs,
        "designer_names": designer_names,
        "designer_ids": designer_ids,
        "case_ids": case_ids,
    }
    print(f"[alternatives] cache ready: {len(urls)} images", flush=True)
    return _CASE_CACHE


def _embed_crop(img: Image.Image) -> np.ndarray | None:
    """把一塊 PIL crop 轉成 CLIP embedding (L2 normalized 768-dim)。"""
    if img is None or img.size[0] < 16 or img.size[1] < 16:
        return None
    import torch
    model, preprocess, device = _load_clip()
    tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype(np.float32)


def find_similar_cases(crop_emb: np.ndarray, top_k: int = 6) -> list[dict]:
    cache = _load_case_cache()
    if cache is None:
        return []
    sims = cache["embs"] @ crop_emb
    order = np.argsort(-sims)[:top_k]
    return [
        {
            "url": str(cache["urls"][i]),
            "designer_name": str(cache["designer_names"][i]),
            "designer_id": int(cache["designer_ids"][i]),
            "case_id": int(cache["case_ids"][i]),
            "similarity": float(sims[i]),
        }
        for i in order
    ]


def detect_and_recommend(image: Image.Image, query: str, threshold: float, top_k_per_obj: int):
    """偵測 + 為每個物件找替代案例。回傳 (annotated_image, objects_state)。"""
    print(f"[alternatives] called  image={'None' if image is None else image.size}", flush=True)
    if image is None:
        return None, [], "請先上傳圖片", gr.update(choices=[], value=None), []

    # reuse object_qa 的 detect / draw_boxes
    from demos.object_qa import detect, draw_boxes
    try:
        resized, objs = detect(image, query, threshold=threshold, top_k=8)
    except Exception as e:
        return None, [], f"❌ 偵測失敗:{e}", gr.update(choices=[], value=None), []

    if not objs:
        return resized, [], "沒偵測到任何物件", gr.update(choices=[], value=None), []

    annotated = draw_boxes(resized, objs)

    # 為每個物件裁框 + embed + 找類似
    for i, o in enumerate(objs):
        x1, y1, x2, y2 = o["box"]
        crop = resized.crop((x1, y1, x2, y2))
        emb = _embed_crop(crop)
        if emb is None:
            o["alternatives"] = []
        else:
            o["alternatives"] = find_similar_cases(emb, top_k=int(top_k_per_obj))
        print(f"  #{i+1} {o['name']}: {len(o['alternatives'])} alts", flush=True)

    summary = f"偵測到 **{len(objs)}** 個物件,各自找了 {top_k_per_obj} 張類似案例參考"
    choices = [(f"#{i+1} {o['name']} ({o['score']*100:.0f}%)", i) for i, o in enumerate(objs)]
    first_alts = _alt_gallery_items(objs[0]) if objs else []
    return annotated, objs, summary, gr.update(choices=choices, value=0), first_alts


def _alt_gallery_items(obj: dict) -> list:
    alts = obj.get("alternatives", []) or []
    items = []
    for a in alts:
        caption = f"{a['designer_name']} · 相似 {a['similarity']*100:.0f}%"
        items.append((a["url"], caption))
    return items


def switch_object(idx, objs):
    if idx is None or not objs or idx >= len(objs):
        return []
    return _alt_gallery_items(objs[idx])


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🔁",
            title="找類似案例 (向量檢索)",
            subtitle="把你喜歡的圖逐物件拆解,在 57,833 張 hhh 案例中找最對味的靈感參考",
            tools=[
                ("Grounding DINO Tiny", "從上傳圖偵測物件並裁框"),
                ("OpenCLIP ViT-L/14", "對每個物件裁框做 embedding"),
                ("Cosine Similarity", "在 57,833 × 768 維 cache 中向量檢索"),
            ],
            cost="$0",
            cost_detail="完全本機,無 API 依賴",
            time="3-5 秒",
            time_detail="DINO 偵測 + 數次向量乘法",
            badges=["離線可用", "MPS 加速", "57k 圖檢索"],
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:上傳一張喜歡的圖,AI 把每個物件拆出來,從 hhh 57,833 張案例中找出風格最對味的靈感參考 — 用來找平價替代款或同風格其他案例。</div>')
                in_img = gr.Image(type="pil", label="上傳一張靈感圖", height=300)
                query = gr.Textbox(
                    value="沙發、燈具、椅子、桌子、植栽、窗簾、抱枕、地毯、藝術品、窗戶、電視",
                    label="要拆解哪些元素?",
                    lines=2,
                    placeholder="例:沙發、椅子、燈具",
                    info="用頓號或逗號分隔。中英皆可,可自由增減。",
                )
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("✨ 從案例庫找對味", variant="primary", scale=2)
                with gr.Accordion("⚙️ 進階設定", open=False, elem_classes=["demo-advanced"]):
                    threshold = gr.Slider(0.15, 0.6, value=0.3, step=0.05, label="偵測敏感度",
                                          info="越高越保守,越低越貪心。預設適用多數情況。")
                    k = gr.Slider(3, 12, value=6, step=1, label="每個物件推薦幾張案例")
                summary_md = gr.Markdown()
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                out_img = gr.Image(type="pil", label="AI 標記出的物件", height=380)
                gr.Markdown("### 對味案例", elem_classes=["demo-section-title"])
                picker = gr.Dropdown(label="想看哪個物件的相似案例?", choices=[], value=None, interactive=True)
                gallery = gr.Gallery(label="hhh 案例庫匹配", columns=3, height=420, object_fit="cover")

        state_objs = gr.State([])

        btn.click(detect_and_recommend, [in_img, query, threshold, k], [out_img, state_objs, summary_md, picker, gallery])
        picker.change(switch_object, [picker, state_objs], gallery)

    return demo
