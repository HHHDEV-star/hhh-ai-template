"""🏗️ 3 分鐘完整翻新提案 — Orchestrator,串多個既有 demo 變完整故事。

流程 (約 2-3 分鐘):
  1. 上傳家裡現況照片
  2. 填預算 / 坪數 / 風格偏好
  3. 按下「給我完整翻新計畫」
  4. AI pipeline 順序執行:
     · 步驟 1: 4 種風格渲染 (Interior Design SDXL)
     · 步驟 2: 預算分析 (Claude)
     · 步驟 3: 推薦對味設計師 Top 5 (CLIP + 既有 cache)
     · 步驟 4: AI 設計理念解說 (Claude vision)
  5. 最後組合成完整 markdown 報告
  6. 可下載為 .html 或 .md
"""

from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import requests
from PIL import Image


REPLICATE_API = "https://api.replicate.com/v1"
STAGING_VERSION = "a3c091059a25590ce2d5ea13651fab63f447f21760e50c358d4b850e844f59ee"
CASE_DB = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "embeddings.sqlite"

# 4 種風格 (少於 staging 的 6 種,節省時間/成本)
STYLES = {
    "🇸🇪 北歐極簡": "masterfully designed scandinavian minimalist interior, light oak wood, white walls, soft natural daylight, plants, photorealistic, 8k",
    "🏭 工業風":   "masterfully designed industrial loft interior, exposed brick, edison bulb pendant, leather sofa, dark wood accents, moody lighting",
    "🍵 日式禪風": "masterfully designed japanese zen interior, tatami platform, washi paper lamp, natural cedar wood, minimalist furniture, warm soft light",
    "💎 輕奢風":   "masterfully designed modern soft luxury interior, beige boucle sofa, brass accents, marble coffee table, neutral palette, elegant",
}


def _token() -> str:
    t = os.environ.get("REPLICATE_API_TOKEN", "").strip()
    if not t or "請貼" in t or t.startswith("r8_xxx"):
        raise RuntimeError("REPLICATE_API_TOKEN 未設定")
    return t


def _claude_client():
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY 未設定")
    return Anthropic(api_key=key)


def _to_data_url(img: Image.Image) -> str:
    img = img.copy()
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return f"data:image/png;base64,{base64.standard_b64encode(buf.getvalue()).decode()}"


def _img_b64(img: Image.Image) -> str:
    img = img.copy()
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def _replicate_run(version: str, inputs: dict, timeout: int = 300) -> list[str]:
    token = _token()
    h_post = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    h_get = {"Authorization": f"Bearer {token}"}
    body = {"version": version, "input": inputs}
    pred = None
    for attempt in range(6):
        r = requests.post(f"{REPLICATE_API}/predictions", headers=h_post, json=body, timeout=90)
        if r.status_code == 429:
            wait = int(r.json().get("retry_after", 6)) + 2
            print(f"[full_plan] rate-limited, 等 {wait}s ({attempt+1}/6)", flush=True)
            time.sleep(wait)
            continue
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Replicate API {r.status_code}: {r.text[:300]}")
        pred = r.json()
        break
    if pred is None:
        raise RuntimeError("Replicate 持續 429")
    pred_id = pred["id"]
    status = pred["status"]
    deadline = time.time() + timeout
    while status not in ("succeeded", "failed", "canceled") and time.time() < deadline:
        time.sleep(3)
        r = requests.get(f"{REPLICATE_API}/predictions/{pred_id}", headers=h_get, timeout=30)
        pred = r.json()
        status = pred["status"]
    if status != "succeeded":
        raise RuntimeError(f"Replicate {status}: {pred.get('error') or pred.get('logs','')[:300]}")
    out = pred.get("output")
    if out is None:
        raise RuntimeError("No output")
    return [out] if isinstance(out, str) else list(out)


def _render_style(image: Image.Image, style_prompt: str) -> Image.Image:
    out = _replicate_run(STAGING_VERSION, {
        "image": _to_data_url(image),
        "prompt": style_prompt,
        "negative_prompt": "ugly, deformed, blurry, low quality",
        "depth_strength": 0.85,
        "promax_strength": 0.85,
        "refiner_strength": 0.4,
        "guidance_scale": 7.5,
        "num_inference_steps": 35,
    }, timeout=300)
    return Image.open(io.BytesIO(requests.get(out[0], timeout=120).content)).convert("RGB")


# ===== CLIP 推薦設計師 =====
_CLIP_MODEL = None
_CLIP_PREP = None
_CASE_CACHE = None


def _load_clip():
    global _CLIP_MODEL, _CLIP_PREP
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREP
    import torch, open_clip
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, prep = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device, quick_gelu=True)
    model.eval()
    _CLIP_MODEL, _CLIP_PREP = model, prep
    return model, prep


def _load_case_cache():
    global _CASE_CACHE
    if _CASE_CACHE is not None:
        return _CASE_CACHE
    if not CASE_DB.exists():
        return None
    con = sqlite3.connect(CASE_DB)
    rows = con.execute(
        "SELECT i.url, i.embedding, i.hdesigner_id, d.name FROM images i "
        "JOIN designers d ON d.hdesigner_id = i.hdesigner_id"
    ).fetchall()
    con.close()
    if not rows:
        return None
    urls = np.array([r[0] for r in rows], dtype=object)
    embs = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    dids = np.array([r[2] for r in rows], dtype=np.int64)
    names = np.array([r[3] or "?" for r in rows], dtype=object)
    unique_dids = np.unique(dids)
    fp = np.zeros((len(unique_dids), embs.shape[1]), dtype=np.float32)
    name_by_did = {}
    counts = {}
    for k, did in enumerate(unique_dids):
        mask = dids == did
        avg = embs[mask].mean(axis=0)
        n = np.linalg.norm(avg)
        if n > 0:
            avg /= n
        fp[k] = avg
        name_by_did[int(did)] = str(names[mask][0])
        counts[int(did)] = int(mask.sum())
    _CASE_CACHE = {
        "urls": urls, "embs": embs, "designer_ids": dids,
        "fp_dids": unique_dids, "fp": fp, "name_by_did": name_by_did, "fp_counts": counts,
    }
    return _CASE_CACHE


def _embed_image(img: Image.Image) -> np.ndarray:
    import torch
    model, prep = _load_clip()
    device = next(model.parameters()).device
    tensor = prep(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0].astype(np.float32)


def _top_designers(taste: np.ndarray, k: int = 5):
    cache = _load_case_cache()
    if cache is None:
        return []
    sims = cache["fp"] @ taste
    order = np.argsort(-sims)[:k]
    out = []
    for j in order:
        did = int(cache["fp_dids"][j])
        mask = cache["designer_ids"] == did
        urls = cache["urls"][mask]
        emb_sub = cache["embs"][mask]
        sim_per = emb_sub @ taste
        best = int(np.argmax(sim_per))
        out.append({
            "name": cache["name_by_did"][did],
            "designer_id": did,
            "similarity": float(sims[j]),
            "thumb": str(urls[best]),
            "image_count": cache["fp_counts"][did],
        })
    return out


# ===== Claude:預算分析 =====
def _budget_analyze(area: float, budget: int, style: str, family: str) -> dict:
    prompt = f"""屋主條件:
- 坪數: {area} 坪
- 預算: {budget} 萬
- 偏好風格: {style}
- 家庭成員: {family}

請用台灣 2026 行情分析,以純 JSON 回傳:
{{
  "achievement_pct": 75,
  "summary": "一句話總結 (≤ 30 字)",
  "breakdown": [
    {{"category": "基礎工程", "amount": 50, "note": "拆除/水電/泥作"}},
    {{"category": "木作/系統櫃", "amount": 60, "note": "..."}},
    {{"category": "風格表現", "amount": 30, "note": "..."}},
    {{"category": "家具家電", "amount": 35, "note": "..."}},
    {{"category": "設計費", "amount": 25, "note": "..."}}
  ],
  "top_compromises": ["建議妥協 1", "建議妥協 2", "建議妥協 3"]
}}
"""
    client = _claude_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1500,
        system="你是台灣裝修預算顧問,熟悉 2026 市場行情。",
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    return json.loads(text)


# ===== Claude:設計理念 =====
def _design_philosophy(image: Image.Image, style: str, budget: int) -> str:
    client = _claude_client()
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=600,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _img_b64(image)}},
                {"type": "text", "text": f"""你是資深室內設計顧問,看著這張屋主家裡現況照,要寫一段「設計理念」放在提案書最上面。

屋主想要的風格:{style}
預算:{budget} 萬

請寫一段 200-280 字的設計理念,內容要:
1. 點出**現況的優點** (要保留的元素)
2. 點出**現況的問題** (需改善的地方,例如動線、收納、採光)
3. 說明**新風格如何整合**這些優缺點
4. 給一個**整體願景** (這個家會變成什麼樣的生活感)

口吻:溫暖專業,像跟客戶面談時的設計師。繁體中文,可以用粗體強調重點。"""},
            ],
        }],
    )
    return msg.content[0].text.strip()


# ===== 主 pipeline (generator,yield 階段更新) =====
def generate_plan(image: Image.Image, area: float, budget: int, family: str, style_pref: str):
    """逐步 yield 各階段結果。"""
    print(f"[full_plan] start  area={area} budget={budget} style={style_pref}", flush=True)
    if image is None:
        yield "請先上傳家裡現況照片。", None, None, None, None, "", None
        return

    # 進度初始化
    progress = "## 🔄 AI Pipeline 進行中\n\n- ⏳ 步驟 1/4:生成 4 種風格渲染...\n"
    yield progress, None, None, None, None, "", None

    # ===== Step 1: 4 風格渲染 =====
    render_results = []
    for style_key, prompt in STYLES.items():
        try:
            print(f"[full_plan] render {style_key}", flush=True)
            img = _render_style(image, prompt)
            render_results.append((img, style_key))
            progress += f"   · ✓ {style_key}\n"
        except Exception as e:
            progress += f"   · ❌ {style_key}: {str(e)[:50]}\n"
            print(f"[full_plan] render {style_key} 失敗: {e}", flush=True)
        yield progress, render_results, None, None, None, "", None

    progress += "\n- ⏳ 步驟 2/4:預算分析...\n"
    yield progress, render_results, None, None, None, "", None

    # ===== Step 2: 預算分析 =====
    budget_data = None
    try:
        budget_data = _budget_analyze(area, budget, style_pref, family)
        progress += f"   · ✓ 達成度 **{budget_data.get('achievement_pct',0)}%** — {budget_data.get('summary','')}\n"
    except Exception as e:
        progress += f"   · ❌ 預算分析失敗: {e}\n"
        print(f"[full_plan] budget 失敗: {e}", flush=True)
    yield progress, render_results, budget_data, None, None, "", None

    progress += "\n- ⏳ 步驟 3/4:推薦對味設計師...\n"
    yield progress, render_results, budget_data, None, None, "", None

    # ===== Step 3: 推薦設計師 =====
    designers = []
    try:
        taste = _embed_image(image)
        designers = _top_designers(taste, k=5)
        progress += f"   · ✓ 找到 {len(designers)} 位對味設計師\n"
    except Exception as e:
        progress += f"   · ❌ 推薦失敗: {e}\n"
        print(f"[full_plan] designer 失敗: {e}", flush=True)
    designer_gallery = [(d["thumb"], f"{d['name']} · 相似 {d['similarity']*100:.0f}% · {d['image_count']} 作品") for d in designers]
    yield progress, render_results, budget_data, designer_gallery, None, "", None

    progress += "\n- ⏳ 步驟 4/4:Claude 撰寫設計理念...\n"
    yield progress, render_results, budget_data, designer_gallery, None, "", None

    # ===== Step 4: 設計理念 =====
    philosophy = ""
    try:
        philosophy = _design_philosophy(image, style_pref, budget)
        progress += "   · ✓ 完成\n"
    except Exception as e:
        progress += f"   · ❌ 失敗: {e}\n"
        print(f"[full_plan] philosophy 失敗: {e}", flush=True)
    yield progress, render_results, budget_data, designer_gallery, None, philosophy, None

    # ===== 組合成完整報告 =====
    progress += "\n## ✅ 所有步驟完成,正在組合報告...\n"
    report_md = _compose_report(philosophy, budget_data, designers, render_results, style_pref)
    # 存 HTML 給下載
    html_path = _save_report_html(report_md)
    progress += f"\n📄 [點此下載完整報告 HTML]({html_path})\n"

    yield progress, render_results, budget_data, designer_gallery, report_md, philosophy, html_path


def _compose_report(philosophy: str, budget_data: dict, designers: list, renders: list, style_pref: str) -> str:
    lines = ["# 🏠 你的完整翻新計畫書\n"]
    lines.append(f"_由 hhh AI Lab 多模型協作生成 · 風格:{style_pref}_\n\n---\n")

    if philosophy:
        lines.append("## 🤖 AI 設計理念\n")
        lines.append(philosophy + "\n\n---\n")

    if renders:
        lines.append("## 🎨 4 種風格渲染預覽\n")
        lines.append("以下是 AI 根據你家現況,套用不同風格的渲染預覽。實際施作可以這些為靈感參考。\n\n")
        # markdown 圖片 (注意:這在 Gradio 內可能顯示不出來,因為 render_results 是 PIL 不是 URL;此處用文字描述)
        for _, name in renders:
            lines.append(f"- ✓ {name}\n")
        lines.append("\n_詳細圖片請見上方並排比較區_\n\n---\n")

    if budget_data:
        pct = budget_data.get("achievement_pct", 0)
        lines.append("## 💸 預算分析\n")
        lines.append(f"### {pct}% 達成度\n\n{budget_data.get('summary', '')}\n\n")
        lines.append("### 預算分配建議\n\n")
        lines.append("| 項目 | 金額 (萬) | 說明 |\n|---|---|---|\n")
        for b in budget_data.get("breakdown", []):
            lines.append(f"| {b['category']} | {b['amount']} | {b.get('note','')} |\n")
        lines.append("\n### 建議妥協 (省錢方向)\n\n")
        for c in budget_data.get("top_compromises", []):
            lines.append(f"- {c}\n")
        lines.append("\n---\n")

    if designers:
        lines.append("## 👤 對味設計師 Top 5\n")
        lines.append("基於你家現況的視覺特徵,從 135 位 hhh 設計師中找出風格最對味的:\n\n")
        for i, d in enumerate(designers, 1):
            lines.append(f"{i}. **{d['name']}** — 相似度 {d['similarity']*100:.0f}% · {d['image_count']} 件作品\n")
        lines.append("\n_詳細作品請見上方設計師畫廊_\n\n---\n")

    lines.append("\n## 下一步\n\n")
    lines.append("1. 📞 從上方 5 位設計師選 2-3 位實際面談\n")
    lines.append("2. 帶這份報告書 (附下載連結) 給設計師看\n")
    lines.append("3. 用「預算分析」段落跟設計師討論可行性\n")
    lines.append("4. 用「4 風格渲染」幫助溝通你想要的風格\n\n")
    lines.append("_本報告由 hhh AI Lab 自動生成,實際裝修以設計師現場勘查為準_\n")

    return "".join(lines)


def _save_report_html(report_md: str) -> str:
    """把 markdown 包成 HTML 存到 tmp,gr.File 可下載。"""
    # 簡易 markdown → html (用 Claude API 沒必要,純 Python 處理)
    try:
        import markdown
        body = markdown.markdown(report_md, extensions=["tables", "fenced_code"])
    except ImportError:
        # fallback: 純文字包進 <pre>
        body = f"<pre>{report_md}</pre>"

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<title>HHH AI Lab — 翻新計畫書</title>
<style>
body {{ font-family: 'PingFang TC', 'Microsoft JhengHei', -apple-system, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px; color: #1e293b; line-height: 1.7; }}
h1 {{ font-size: 28px; border-bottom: 3px solid #0f172a; padding-bottom: 12px; }}
h2 {{ font-size: 20px; color: #0f172a; margin-top: 30px; border-left: 4px solid #3b82f6; padding-left: 12px; }}
h3 {{ font-size: 16px; color: #1e293b; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
th {{ background: #f8fafc; }}
hr {{ border: none; border-top: 1px solid #e2e8f0; margin: 20px 0; }}
</style></head><body>{body}</body></html>"""
    fd, path = tempfile.mkstemp(suffix=".html", prefix="hhh_plan_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🏗️",
            title="3 分鐘完整翻新計畫書",
            subtitle="一張家裡現況照 → AI 自動跑完 4 大維度分析,組合成可下載的完整翻新計畫書",
            tools=[
                ("Interior Design SDXL (Depth)", "4 種風格渲染,結構保留"),
                ("Claude Sonnet 4.6", "預算分析 + 設計理念撰寫"),
                ("OpenCLIP ViT-L/14", "從 135 位設計師找對味 Top 5"),
                ("Python markdown → HTML", "可下載完整提案書"),
            ],
            cost="$0.20",
            cost_detail="4 風格渲染 + 2 次 Claude vision call",
            time="2-3 分鐘",
            time_detail="4 渲染主要耗時,Claude 各 5-10s",
            badges=["完整流程", "Pipeline 整合", "可下載"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:上傳家裡現況照 + 填預算/坪數/風格 → 按下「給我完整翻新計畫」→ AI 在 2-3 分鐘內跑完 4 個步驟,組合成可下載的提案書。<br/>📌 適合:屋主決策前的整體規劃、跟設計師談前的準備。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                in_img = gr.Image(type="pil", label="家裡現況照", height=300)
                area = gr.Number(label="坪數", value=25, minimum=5, maximum=200)
                budget = gr.Number(label="預算 (萬)", value=200, minimum=50, maximum=2000)
                family = gr.Radio(label="家庭成員", choices=["夫妻", "夫妻+1孩", "夫妻+2孩", "單身", "其他"], value="夫妻")
                style_pref = gr.Dropdown(label="偏好風格", choices=list(STYLES.keys()), value="🇸🇪 北歐極簡")
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("🏗️ 給我完整翻新計畫 (2-3 分鐘)", variant="primary", scale=2)
                progress_md = gr.Markdown()

            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                gr.Markdown("### 4 種風格渲染", elem_classes=["demo-section-title"])
                renders_gallery = gr.Gallery(label=None, show_label=False, columns=2, height=380, object_fit="contain")
                gr.Markdown("### 對味設計師", elem_classes=["demo-section-title"])
                designers_gallery = gr.Gallery(label=None, show_label=False, columns=5, rows=1, height=180, object_fit="cover")
                gr.Markdown("### 設計理念", elem_classes=["demo-section-title"])
                philosophy_md = gr.Markdown()
                gr.Markdown("### 完整報告", elem_classes=["demo-section-title"])
                report_md = gr.Markdown()
                gr.Markdown("### 預算分析 (Raw JSON)", elem_classes=["demo-section-title"])
                budget_json = gr.JSON()
                gr.Markdown("### 下載提案書", elem_classes=["demo-section-title"])
                download_file = gr.File(label=None, show_label=False, interactive=False)

        btn.click(
            generate_plan,
            inputs=[in_img, area, budget, family, style_pref],
            outputs=[progress_md, renders_gallery, budget_json, designers_gallery, report_md, philosophy_md, download_file],
        )

    return demo
