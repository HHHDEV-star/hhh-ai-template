"""🏠 Welcome — HHH AI Lab 首頁。

整體 demo 概覽 + 技術 stack + 怎麼開始,給第一印象「這是一個產品」而不是「一堆 demo」。
"""

from __future__ import annotations

import gradio as gr


# 全部 demo 的「銷售文案」介紹 (跟 app.py DEMOS_BY_CATEGORY 保持一致)
DEMOS_INFO = [
    # category, items: [(icon, name, tagline, badges, estimated_cost)]
    ("🌟 旗艦體驗", [
        ("🕰️", "我家 Time Machine", "看見家的 10 年未來 — 同房間 5 個時間切片演化", ["旗艦", "情感連結"], "$0.20-0.25"),
        ("🏗️", "3 分鐘完整翻新計畫", "一張現況照,自動跑完 4 維度分析,組合成可下載提案書", ["旗艦", "Pipeline"], "$0.20"),
    ]),
    ("互動式 AI", [
        ("🧬", "風格 DNA Tinder", "Tinder 式刷卡測你的設計品味 → 推薦對味設計師", ["遊戲化"], "免費"),
        ("💸", "預算焦慮治療師", "AI 算你的預算能做到幾成 + 該妥協哪些項目", ["決策助手"], "$0.05"),
        ("📐", "AI 平面圖生成器", "用講的就能畫平面圖 — 屋主填條件,即時看見格局", ["決策工具"], "$0.02"),
        ("🔍", "物件偵測 + Q&A", "上傳案例照 → AI 框出每件家具,點選獲得專業解答", ["互動 Q&A"], "$0.05"),
    ]),
    ("資料洞察", [
        ("🔮", "設計趨勢儀表板", "從 17k 案例 × 15 年資料,看設計風格如何演變", ["B2B 媒體"], "$0.05"),
        ("👤", "設計師風格指紋", "上傳一張喜歡的圖,從 135 位設計師找對味前 5 名", ["離線", "0 成本"], "免費"),
        ("🗺️", "設計師風格地圖", "UMAP 把 135 位設計師壓到 2D 平面看風格相似度", ["視覺化"], "免費"),
        ("🔁", "找類似案例", "逐物件拆解,從 57,833 張案例找相似靈感", ["離線", "向量檢索"], "免費"),
    ]),
    ("內容生成", [
        ("🎨", "AI 室內設計師", "線稿或現況照 → 6 種風格擬真渲染,自動切後端", ["雲端", "6 風格"], "$0.02/張"),
        ("📸", "AI 數位攝影師", "手機隨手拍 → Magnific 等級增強 → 雜誌劇照", ["攝影增強"], "$0.04/張"),
        ("🚶", "案例沉浸漫遊", "案例圖 → SVD 生成漫遊短片 + Claude 導覽詞", ["影片生成"], "$0.15"),
        ("✍️", "AI 案例文案 / SEO", "一張圖一鍵生 SEO + FB/Threads/IG 全套文案", ["SEO 工具"], "$0.05"),
        ("🪄", "一鍵空房", "AI 自動識別並移除所有家具,還原成空房狀態", ["房產 staging"], "$0.05-0.08"),
        ("🎯", "切割 → 3D", "切出單一物件 → 變可在瀏覽器拖旋的 3D 模型", ["3D 輸出"], "$0.05-0.10"),
    ]),
    ("基礎工具", [
        ("🌈", "配色 DNA", "圖片主色萃取 — 用於配色搜尋或案例色彩指紋", ["離線", "0 成本"], "免費"),
    ]),
]


def _badge(text: str) -> str:
    return f'<span class="welcome-badge">{text}</span>'


def _card(icon: str, name: str, tagline: str, badges: list[str], cost: str) -> str:
    badges_html = " ".join(_badge(b) for b in badges)
    return f"""
<div class="welcome-card">
  <div class="welcome-card-head">
    <div class="welcome-icon">{icon}</div>
    <div class="welcome-cost">{cost}</div>
  </div>
  <div class="welcome-name">{name}</div>
  <div class="welcome-tag">{tagline}</div>
  <div class="welcome-badges">{badges_html}</div>
</div>
"""


def build() -> gr.Blocks:
    total_demos = sum(len(items) for _, items in DEMOS_INFO)

    # 生成 demo cards HTML
    categories_html = []
    for category, items in DEMOS_INFO:
        cards = "".join(_card(*it) for it in items)
        categories_html.append(f"""
<div class="welcome-category">
  <h3 class="welcome-cat-title">{category}</h3>
  <div class="welcome-grid">{cards}</div>
</div>
""")

    page_html = f"""
<div class="welcome-page">

  <!-- HERO -->
  <div class="welcome-hero">
    <div class="welcome-hero-content">
      <div class="welcome-mark">H</div>
      <h1 class="welcome-h1">HHH AI Lab</h1>
      <p class="welcome-sub">室內設計 × 多模態 AI · {total_demos} 個 production-ready demos</p>
      <div class="welcome-meta">
        <span class="welcome-pill"><span class="dot"></span> 在線</span>
        <span class="welcome-pill">CLIP · Claude · SDXL · SAM · InstantMesh · SVD</span>
        <span class="welcome-pill">📊 57,833 圖 / 135 設計師 / 17,274 案例</span>
      </div>
    </div>
  </div>

  <!-- HOW TO START -->
  <div class="welcome-howto">
    <div class="howto-step"><span class="howto-num">1</span><div>從**左側選單**選想試的 demo</div></div>
    <div class="howto-step"><span class="howto-num">2</span><div>每個 demo 頂部有 <strong>📊 Tech Specs</strong> 看時間/成本/技術</div></div>
    <div class="howto-step"><span class="howto-num">3</span><div>進階參數收在「⚙️ 進階設定」,不用看也能跑</div></div>
  </div>

  <!-- DEMOS GRID -->
  {"".join(categories_html)}

  <!-- TECH STACK -->
  <div class="welcome-stack">
    <h3 class="welcome-cat-title">⚙️ 技術 stack</h3>
    <div class="stack-grid">
      <div class="stack-item"><strong>視覺理解</strong><span>OpenCLIP ViT-L/14 · Grounding DINO · SAM 2 · Grounded SAM</span></div>
      <div class="stack-item"><strong>視覺生成</strong><span>SDXL + ControlNet (Canny/Depth) · SVD · InstantMesh · Clarity Upscaler</span></div>
      <div class="stack-item"><strong>語言模型</strong><span>Claude Sonnet 4.6 (vision-enabled) · Anthropic API</span></div>
      <div class="stack-item"><strong>資料層</strong><span>SQLite (CLIP cache, 57k 圖) · MySQL (xoops 案例 DB) · Replicate API</span></div>
      <div class="stack-item"><strong>互動框架</strong><span>Gradio 5.50 · Python 3.14 · Apple MPS / CUDA</span></div>
    </div>
  </div>

  <!-- FOOTER -->
  <div class="welcome-footer">
    <div>Powered by HHH AI Lab · Built with Gradio + Anthropic + Replicate + OpenCLIP</div>
    <div class="welcome-sub-small">本平台所有 demo 為實驗性質,實際裝修以設計師現場勘查為準</div>
  </div>

</div>
"""

    with gr.Blocks() as demo:
        gr.HTML(page_html, elem_classes=["demo-hero-html", "welcome-html"])
    return demo
