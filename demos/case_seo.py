"""AI 案例文案 / SEO 自動生成。

輸入:
  - 案例圖 (上傳 OR 從 _hcase 表撈)
  - (選填) hcase_id → 拉設計師、坪數、風格等資料增強 prompt

輸出 (Claude 看圖一次生):
  - SEO title (≤ 60 字)
  - meta description (140-160 字)
  - <img> alt text
  - 社群貼文 3 版 (FB / Threads / IG caption)
  - 案例描述段落 (~150 字)
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any

import gradio as gr
import requests
from PIL import Image


_CLAUDE = None


def _claude():
    global _CLAUDE
    if _CLAUDE is not None:
        return _CLAUDE
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key or key.startswith("sk-ant-api03-xxx"):
        raise RuntimeError("ANTHROPIC_API_KEY 未設定")
    _CLAUDE = Anthropic(api_key=key)
    return _CLAUDE


def _fetch_case_context(hcase_id: int) -> dict | None:
    """從 xoops _hcase + _hdesigner 表撈 metadata 加進 prompt。"""
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from db_utils import connect
        conn = connect()
    except Exception as e:
        print(f"[case_seo] DB 連線失敗 (略過 context): {e}", flush=True)
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.hcase_id, c.caption, c.short_desc, c.style, c.type, c.area, c.areaDesc,
                       c.layout, c.location, c.member, d.name AS designer_name, d.style AS designer_style
                FROM _hcase c JOIN _hdesigner d ON d.hdesigner_id = c.hdesigner_id
                WHERE c.hcase_id = %s AND c.onoff = 1
                """,
                (hcase_id,),
            )
            row = cur.fetchone()
        conn.close()
        return row
    except Exception as e:
        print(f"[case_seo] DB query 失敗: {e}", flush=True)
        try: conn.close()
        except: pass
        return None


def _img_b64(img: Image.Image) -> str:
    img = img.copy()
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode()


def generate_seo(image: Image.Image, hcase_id_str: str):
    print(f"[case_seo] generate_seo  image={'None' if image is None else image.size}  hcase_id={hcase_id_str!r}", flush=True)
    if image is None and not hcase_id_str.strip():
        return "請上傳圖片,或輸入 hcase_id 從案例庫拉圖"

    # 從 DB 拿 context + 圖 (如果 hcase_id 有給)
    ctx = None
    if hcase_id_str.strip().isdigit():
        ctx = _fetch_case_context(int(hcase_id_str.strip()))
        if ctx and image is None:
            # 嘗試從 _hcase_img 拿封面圖
            try:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                from db_utils import connect
                conn = connect()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT name FROM _hcase_img WHERE hcase_id = %s AND is_cover = 1 LIMIT 1",
                        (int(hcase_id_str),),
                    )
                    r = cur.fetchone()
                    if not r:
                        cur.execute(
                            "SELECT name FROM _hcase_img WHERE hcase_id = %s ORDER BY hcase_img_id LIMIT 1",
                            (int(hcase_id_str),),
                        )
                        r = cur.fetchone()
                conn.close()
                if r and r["name"].startswith("http"):
                    img_resp = requests.get(r["name"], timeout=15)
                    image = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    print(f"[case_seo] 從 DB 撈到圖: {r['name']}", flush=True)
            except Exception as e:
                print(f"[case_seo] 拉案例圖失敗: {e}", flush=True)

    if image is None:
        return "請上傳圖片"

    # 建 context block
    ctx_block = ""
    if ctx:
        ctx_block = f"""
參考資料 (來自我們的案例資料庫,你可以引用):
- 設計師: {ctx.get('designer_name') or '(未知)'}
- 設計師擅長風格: {(ctx.get('designer_style') or '')[:80]}
- 案例風格: {ctx.get('style') or ''}
- 案例類型: {ctx.get('type') or ''}
- 坪數: {ctx.get('areaDesc') or ''}
- 格局: {(ctx.get('layout') or '')[:100]}
- 地點: {ctx.get('location') or ''}
- 居住成員: {ctx.get('member') or ''}
- 原始標題: {ctx.get('caption') or ''}
- 原始簡介前段: {(ctx.get('short_desc') or '')[:200]}
"""

    prompt = f"""你是 hhh 幸福空間的**SEO 與內容編輯**,擅長把室內設計案例寫得既吸引人又有 SEO 效益。
看著這張案例照片,生成以下 6 個內容版本。{ctx_block}
請以**純 JSON** 回傳 (不要 markdown code fence),格式:

{{
  "seo_title": "≤ 60 字繁體中文,有風格+空間+亮點,自然帶關鍵字",
  "meta_description": "140-160 字繁體,概括設計亮點,含 call-to-action",
  "alt_text": "≤ 100 字,描述圖片內容供視障/搜尋引擎理解,具體描寫家具/材質/光線",
  "fb_post": "Facebook 貼文版,150-250 字,可帶 emoji,結尾 #hashtag 3-5 個",
  "threads_post": "Threads 貼文,80-120 字短句感,可有故事性",
  "ig_caption": "IG caption,80-150 字,前 3 行要抓眼球,結尾 8-10 個相關 #hashtag",
  "description": "案例描述段落,150-200 字,適合放在案例頁正文,流暢有溫度"
}}

要求:
- **全部繁體中文**
- 看著照片講具體細節 (例如「全室淺木色搭配米白沙發」),不要泛泛
- SEO 友善:有「室內設計」「裝修」「{(ctx.get('style','設計') if ctx else '設計')}風」等自然關鍵字
- 不要重複內容 — 每個欄位寫法要有差異 (社群貼文有人味、SEO description 精煉、alt 純客觀)
"""

    try:
        msg = _claude().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _img_b64(image)}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.rstrip("`").strip()
        data = json.loads(text)
    except Exception as e:
        import traceback; traceback.print_exc()
        return f"❌ 失敗:{type(e).__name__}: {e}"

    # 格式化為 markdown
    lines = []
    lines.append("## 🏷️ SEO 元素\n")
    lines.append(f"**Title** ({len(data.get('seo_title',''))}字)\n```\n{data.get('seo_title','')}\n```\n")
    lines.append(f"**Meta Description** ({len(data.get('meta_description',''))}字)\n```\n{data.get('meta_description','')}\n```\n")
    lines.append(f"**Image Alt** ({len(data.get('alt_text',''))}字)\n```\n{data.get('alt_text','')}\n```\n")
    lines.append("\n---\n## 📱 社群媒體版本\n")
    lines.append("### Facebook\n```\n" + data.get('fb_post','') + "\n```\n")
    lines.append("### Threads\n```\n" + data.get('threads_post','') + "\n```\n")
    lines.append("### Instagram\n```\n" + data.get('ig_caption','') + "\n```\n")
    lines.append("\n---\n## 📝 案例頁主文案\n")
    lines.append(data.get('description',''))
    return "\n".join(lines)


def build() -> gr.Blocks:
    from demos._ui import render_meta_header
    with gr.Blocks() as demo:
        render_meta_header(
            icon="✍️",
            title="AI 案例文案 / SEO 生成",
            subtitle="一張圖一鍵生 SEO 三劍客 (title/meta/alt) + 社群三平台貼文 + 案例頁本文,可吃 DB 設計師資料當 context",
            tools=[
                ("Claude Sonnet 4.6 (vision)", "看圖+設計師 metadata 一次生 7 個文案版本"),
                ("xoops _hcase + _hdesigner", "選填 hcase_id 從 RDS 拉真實案例 context"),
            ],
            cost="~$0.05",
            cost_detail="單次 Claude vision call,2500 tokens 上限",
            time="5-10 秒",
            time_detail="一次 API call 拿全套",
            badges=["Claude API", "DB 整合", "全自動 SEO"],
        )
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["demo-input-pane"]):
                gr.HTML('<div class="demo-hint">💡 <strong>怎麼玩</strong>:一張案例圖,AI 為你生成 SEO 標題、Meta description、圖片 alt、FB/Threads/IG 三平台社群貼文,以及案例頁主文案 — 一鍵全套搞定。<br/>📌 輸入 <code>hcase_id</code> 可從 hhh 資料庫拉真實設計師資料,文案會更精準。</div>')
                with gr.Tabs():
                    with gr.Tab("📷 上傳案例圖"):
                        in_img = gr.Image(type="pil", label="案例圖", height=300)
                    with gr.Tab("🔖 從 hhh 案例庫"):
                        hcase_id = gr.Textbox(
                            label="hcase_id",
                            placeholder="例如:3",
                            info="輸入後會自動撈圖 + 設計師/案例資料,文案會更精準",
                        )
                with gr.Row(elem_classes=["demo-cta"]):
                    btn = gr.Button("✍️ 一鍵生成全套文案", variant="primary", scale=2)
            with gr.Column(scale=2, elem_classes=["demo-output-pane"]):
                out_md = gr.Markdown()

        btn.click(generate_seo, [in_img, hcase_id], out_md)

    return demo
