"""🕰️ AI 我家 Time Machine — 看見家的 10 年未來。

流程:
  1. 上傳家裡現況 + 填家庭規劃 (現在組成 / 5-10 年計劃)
  2. Claude 看圖 + 規劃,為 5 個時間點生成個人化 prompt
  3. SDXL (Interior Design) 渲染 5 個時間切片 — 同房間,不同時期
  4. Claude 撰寫每階段的「生活感」+ 該添購/汰換家具 + 維護成本
  5. 5 張並排 + 時間軸故事呈現

技術:
  - Claude Sonnet 4.6 (vision) 規劃時間軸 + 寫生活描述
  - Interior Design SDXL (depth ControlNet) 渲染每個時間切片
"""

from __future__ import annotations

import base64
import io
import json
import os
import time

import gradio as gr
import requests
from PIL import Image


REPLICATE_API = "https://api.replicate.com/v1"
STAGING_VERSION = "a3c091059a25590ce2d5ea13651fab63f447f21760e50c358d4b850e844f59ee"

TIMELINE = [
    ("📅 現在",      "now",       "現況"),
    ("⏱️ 1 年後",     "1y",        "剛安頓好"),
    ("📆 3 年後",     "3y",        "家庭變化期"),
    ("🗓️ 5 年後",     "5y",        "孩子上學 / 寵物成熟"),
    ("⏳ 10 年後",    "10y",       "家的成長期"),
]


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
            print(f"[time_machine] rate-limited, 等 {wait}s ({attempt+1}/6)", flush=True)
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


def _plan_timeline(image: Image.Image, family_now: str, plan_5y: str, plan_10y: str, style: str) -> dict:
    """請 Claude 看圖 + 家庭規劃,為 5 個時間點各生成 1 個英文 SDXL prompt + 中文生活描述。"""
    client = _claude_client()
    instruction = f"""你是資深室內設計師,要為這個屋主規劃「10 年家庭演變」。

屋主資訊:
- 偏好風格: {style}
- 現在家庭組成: {family_now}
- 5 年內計劃: {plan_5y}
- 10 年內計劃: {plan_10y}

請看著現況照,為 5 個時間切片各生成:
1. `prompt_en`: 給 SDXL 用的英文 prompt,描述該時期房間樣貌
   - 必須保持 {style} 主軸
   - 包含該時期特定元素 (例如 1y 後加新家具、3y 後加小孩遊戲區、5y 後增收納、10y 後成熟感)
   - 必須含 "photorealistic, 8k, professional interior photography"
2. `lifestyle_zh`: 該時期的「生活感描述」 (60-100 字繁中)
3. `additions`: 該時期該添購的 2-3 件家具/物品 (list of zh strings)
4. `cost_estimate`: 該時期的維護/添購成本 (例如 "8-12 萬")

純 JSON 回傳 (不要 markdown fence):
{{
  "now":  {{"prompt_en": "...", "lifestyle_zh": "...", "additions": ["..."], "cost_estimate": "..."}},
  "1y":   {{"prompt_en": "...", "lifestyle_zh": "...", "additions": ["..."], "cost_estimate": "..."}},
  "3y":   {{"prompt_en": "...", "lifestyle_zh": "...", "additions": ["..."], "cost_estimate": "..."}},
  "5y":   {{"prompt_en": "...", "lifestyle_zh": "...", "additions": ["..."], "cost_estimate": "..."}},
  "10y":  {{"prompt_en": "...", "lifestyle_zh": "...", "additions": ["..."], "cost_estimate": "..."}}
}}
"""
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=3000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _img_b64(image)}},
                {"type": "text", "text": instruction},
            ],
        }],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    return json.loads(text)


def _render_period(image: Image.Image, prompt: str) -> Image.Image:
    """SDXL render for one time period — keeps structure via depth, changes content."""
    out = _replicate_run(STAGING_VERSION, {
        "image": _to_data_url(image),
        "prompt": prompt,
        "negative_prompt": "ugly, deformed, blurry, low quality, distorted",
        "depth_strength": 0.80,
        "promax_strength": 0.85,
        "refiner_strength": 0.4,
        "guidance_scale": 7.5,
        "num_inference_steps": 35,
    }, timeout=300)
    return Image.open(io.BytesIO(requests.get(out[0], timeout=120).content)).convert("RGB")


def time_machine(image: Image.Image, family_now: str, plan_5y: str, plan_10y: str, style: str):
    """逐步 yield 各時間切片的渲染進度 + 最終時間軸故事。"""
    print(f"[time_machine] start  family={family_now!r} 5y={plan_5y!r}", flush=True)
    empty_returns = (
        "請上傳現況照,並填寫家庭規劃",
        [], "", None, None, None, None, None,
    )
    if image is None:
        yield empty_returns
        return

    progress = "## 🔄 Pipeline 進行中\n\n### 步驟 1/2 · Claude 規劃時間軸\n⏳ 看圖 + 整合家庭規劃...\n"
    yield (progress, [], "", None, None, None, None, None)

    # ===== Step 1: Claude 規劃 5 時段 =====
    try:
        plan = _plan_timeline(image, family_now, plan_5y, plan_10y, style)
        progress += "✓ Claude 規劃完成。各時段勾畫生活情境。\n\n### 步驟 2/2 · 渲染 5 個時間切片\n"
    except Exception as e:
        import traceback; traceback.print_exc()
        yield (f"❌ Claude 規劃失敗: {e}", [], "", None, None, None, None, None)
        return

    yield (progress, [], "", None, None, None, None, None)

    # ===== Step 2: 逐個渲染 + 累積 gallery =====
    rendered_imgs: dict[str, Image.Image] = {}
    individual_outputs = [None, None, None, None, None]  # now/1y/3y/5y/10y

    for i, (display, key, brief) in enumerate(TIMELINE):
        progress += f"\n⏳ 正在渲染 **{display}** ({brief})..."
        # 把現有的 rendered 順序組成 gallery,讓使用者邊看邊更新
        gallery = [(rendered_imgs[k], TIMELINE[idx][0]) for idx, (_, k, _) in enumerate(TIMELINE) if k in rendered_imgs]
        yield (progress, gallery, "", *individual_outputs)

        try:
            period_data = plan.get(key, {})
            prompt = period_data.get("prompt_en", f"{style} interior, photorealistic, 8k")
            img = _render_period(image, prompt)
            rendered_imgs[key] = img
            individual_outputs[i] = img
            progress += f"  ✓"
        except Exception as e:
            progress += f"  ❌ {str(e)[:60]}"
            print(f"[time_machine] {key} 失敗: {e}", flush=True)

        gallery = [(rendered_imgs[k], TIMELINE[idx][0]) for idx, (_, k, _) in enumerate(TIMELINE) if k in rendered_imgs]
        yield (progress, gallery, "", *individual_outputs)

    # ===== Step 3: 組成時間軸故事 =====
    progress += "\n\n### ✅ 所有時段完成,組合時間軸故事...\n"

    story_lines = ["# 🕰️ 你家的 10 年成長故事\n"]
    story_lines.append(f"_由 hhh AI Lab 生成 · 偏好風格:{style}_\n\n---\n")

    total_addition_cost = 0
    for display, key, brief in TIMELINE:
        period_data = plan.get(key, {})
        story_lines.append(f"\n## {display} — {brief}\n")
        lifestyle = period_data.get("lifestyle_zh", "")
        story_lines.append(f"_{lifestyle}_\n\n")
        adds = period_data.get("additions", [])
        if adds:
            story_lines.append("**該時期建議添購:**\n")
            for a in adds:
                story_lines.append(f"- {a}\n")
        cost = period_data.get("cost_estimate", "")
        if cost:
            story_lines.append(f"\n💰 預估維護/添購成本:**{cost}**\n")
        story_lines.append("\n---\n")

    story_lines.append("\n## 💡 致家的承諾\n")
    story_lines.append(
        "家不是一次裝修就定型的 — 它跟著你的生活一起呼吸、長大。\n\n"
        "這份「10 年計劃」不是預言,是讓你**今天裝修決策時看得更遠**:\n"
        "- 選沙發時想到 5 年後孩子會跳上去 → 選耐磨易清的布料\n"
        "- 選收納時想到 10 年後雜物會多兩倍 → 預留 30% 餘量\n"
        "- 選地板時想到狗狗會跑來跑去 → 選不易刮傷的材質\n\n"
        "_本報告 6 個月後再回來重跑一次,看你家的故事是不是按計劃走_\n"
    )

    story_md = "".join(story_lines)
    progress += "\n📄 時間軸故事完成!\n"

    gallery = [(rendered_imgs[k], display) for display, k, _ in TIMELINE if k in rendered_imgs]
    yield (progress, gallery, story_md, *individual_outputs)


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="🕰️",
            title="我家 Time Machine — 看見家的 10 年未來",
            subtitle="同一個房間,5 個時間切片 — 從現在到 10 年後,AI 為你預演家會如何陪伴你長大",
            tools=[
                ("Claude Sonnet 4.6 (vision)", "看圖 + 家庭規劃,為 5 時段各生成個人化 prompt + 生活描述"),
                ("Interior Design SDXL + Depth", "保留現況結構,渲染各時期樣貌"),
                ("Python markdown report", "組成 10 年成長故事"),
            ],
            cost="$0.20-0.25",
            cost_detail="5 次 SDXL render + 1 次 Claude vision",
            time="2-3 分鐘",
            time_detail="每個時段 ~20 秒,Claude 規劃 ~10 秒",
            badges=["旗艦體驗", "時間維度", "情感連結"],
        )

        gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:上傳家裡現況 + 填家庭規劃,AI 在 2-3 分鐘內生成「現在 / 1 年 / 3 年 / 5 年 / 10 年」5 張同房間不同時期的樣貌 + 每階段生活描述 + 該添購家具清單。<br/>📌 看見家的未來 → 今天裝修決策更有遠見。</div>')

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                in_img = gr.Image(type="pil", label="上傳家裡現況照", height=300)
                style = gr.Dropdown(
                    label="主軸風格",
                    choices=["北歐極簡", "現代簡約", "輕奢風", "日式禪風", "工業風", "混搭風"],
                    value="北歐極簡",
                )
                family_now = gr.Radio(
                    label="現在家庭組成",
                    choices=["單身", "夫妻", "夫妻+1孩", "夫妻+2孩", "與長輩同住"],
                    value="夫妻",
                )
                plan_5y = gr.Textbox(
                    label="5 年內計劃",
                    placeholder="例:打算明年生小孩、預計養 1 隻狗",
                    lines=2,
                )
                plan_10y = gr.Textbox(
                    label="10 年願景",
                    placeholder="例:小孩會上小學,需要書房;父母年邁可能搬來同住",
                    lines=2,
                )
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("🕰️ 看見我家 10 年未來 (2-3 分鐘)", variant="primary", scale=2)
                progress_md = gr.Markdown()

            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                gr.Markdown("### 時間切片並排", elem_classes=["demo-section-title"])
                gallery = gr.Gallery(
                    label=None, show_label=False,
                    columns=5, rows=1, height=240, object_fit="cover",
                )

                gr.Markdown("### 5 個時段細看", elem_classes=["demo-section-title"])
                with gr.Row():
                    img_now = gr.Image(type="pil", label="📅 現在",     height=220, interactive=False)
                    img_1y  = gr.Image(type="pil", label="⏱️ 1 年後",   height=220, interactive=False)
                    img_3y  = gr.Image(type="pil", label="📆 3 年後",   height=220, interactive=False)
                    img_5y  = gr.Image(type="pil", label="🗓️ 5 年後",   height=220, interactive=False)
                    img_10y = gr.Image(type="pil", label="⏳ 10 年後",  height=220, interactive=False)

                gr.Markdown("### 10 年成長故事", elem_classes=["demo-section-title"])
                story_md = gr.Markdown()

        btn.click(
            time_machine,
            inputs=[in_img, family_now, plan_5y, plan_10y, style],
            outputs=[progress_md, gallery, story_md, img_now, img_1y, img_3y, img_5y, img_10y],
        )

    return demo
