"""Demo UI 共用元件 — 標準化每個 demo 頂部的 hero + tech specs tooltip。

每個 demo 的 build() 開頭呼叫 render_meta_header() 即可。
"""

from __future__ import annotations

import html

import gradio as gr


# 全域 CSS — 套用到所有 demo
GLOBAL_CSS = """
/* ===== Reset / 全域字型 ===== */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang TC', 'Microsoft JhengHei', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.gradio-container, .gradio-container * { box-sizing: border-box; }

/* 重點:Gradio gr.HTML 預設 wrapper (.block.hide-container.auto-margin) 會給 height/overflow:auto,
   導致 hero 出現垂直捲軸。targets 掛了 .demo-hero-html class 的 wrapper 強制改寫 */
.demo-hero-html,
.demo-hero-html .prose,
.demo-hero-html .html-container,
.demo-hero-html > div,
.demo-hero-html > * {
    overflow: visible !important;
    height: auto !important;
    max-height: none !important;
    min-height: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
}

/* ===== Demo Hero ===== */
.demo-hero {
    position: relative;
    padding: 32px 36px;
    margin: 4px 0 28px 0;
    border-radius: 16px;
    background:
        radial-gradient(ellipse at top right, rgba(59,130,246,0.15) 0%, transparent 60%),
        linear-gradient(135deg, #0b1220 0%, #1a2436 55%, #2a3a55 100%);
    color: #f1f5f9;
    overflow: visible;  /* 給 popup 用 */
    border: 1px solid rgba(255,255,255,0.06);
}
.demo-hero .hero-top {
    display: flex; justify-content: space-between; align-items: flex-start; gap: 24px;
}
.demo-hero .hero-title { flex: 1; min-width: 0; }
.demo-hero h1 {
    font-size: 28px !important; font-weight: 700 !important;
    margin: 0 0 8px 0 !important;
    color: #f8fafc !important;
    letter-spacing: -0.025em;
    line-height: 1.2;
}
.demo-hero p.subtitle {
    font-size: 15px !important;
    color: #cbd5e1 !important;
    margin: 0 !important;
    line-height: 1.55;
    max-width: 680px;
}
.demo-hero .badges {
    margin-top: 18px;
    display: flex; gap: 8px; flex-wrap: wrap;
}
.demo-hero .badge {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 11px;
    color: #cbd5e1;
    font-weight: 500;
    letter-spacing: 0.01em;
}

/* ===== Tech Specs Tooltip ===== */
.tech-trigger {
    position: relative;
    flex-shrink: 0;
}
.tech-btn {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #e2e8f0;
    padding: 9px 14px;
    border-radius: 9px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: help;
    transition: all 0.15s ease;
    user-select: none;
    letter-spacing: 0.01em;
}
.tech-btn:hover {
    background: rgba(255,255,255,0.13);
    border-color: rgba(255,255,255,0.22);
    transform: translateY(-1px);
}
.tech-btn .arrow {
    font-size: 10px;
    opacity: 0.7;
    transition: transform 0.15s ease;
}
.tech-trigger:hover .arrow,
.tech-trigger:focus-within .arrow { transform: rotate(180deg); }

.tech-popup {
    position: absolute;
    top: calc(100% + 14px);
    right: 0;
    width: 420px;
    max-width: calc(100vw - 64px);
    background: #ffffff;
    color: #0f172a;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 0;
    box-shadow:
        0 25px 50px -12px rgba(15,23,42,0.35),
        0 10px 20px -5px rgba(15,23,42,0.12);
    opacity: 0; visibility: hidden;
    transform: translateY(-6px);
    transition: opacity 0.18s ease, transform 0.18s ease, visibility 0.18s ease;
    z-index: 1000;
    overflow: hidden;
}
.tech-trigger:hover .tech-popup,
.tech-trigger:focus-within .tech-popup {
    opacity: 1; visibility: visible;
    transform: translateY(0);
}
/* 小三角指示器 */
.tech-popup::before {
    content: '';
    position: absolute;
    top: -7px; right: 24px;
    width: 12px; height: 12px;
    background: #ffffff;
    border-left: 1px solid #e2e8f0;
    border-top: 1px solid #e2e8f0;
    transform: rotate(45deg);
}

.tech-popup .specs {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    border-bottom: 1px solid #f1f5f9;
}
.tech-popup .spec {
    padding: 16px 20px;
    border-right: 1px solid #f1f5f9;
}
.tech-popup .spec:last-child { border-right: none; }
.tech-popup .spec-label {
    font-size: 10.5px;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    display: flex; align-items: center; gap: 5px;
    margin-bottom: 6px;
}
.tech-popup .spec-value {
    font-size: 19px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.01em;
    line-height: 1.2;
}
.tech-popup .spec-sub {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 4px;
    line-height: 1.4;
}

.tech-popup .stack {
    padding: 16px 20px 18px;
    background: #fafbfc;
}
.tech-popup .stack-label {
    font-size: 10.5px;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    display: flex; align-items: center; gap: 5px;
    margin-bottom: 10px;
}
.tech-popup .stack-list {
    display: flex; flex-direction: column; gap: 7px;
}
.tech-popup .stack-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    padding: 9px 13px;
    border-radius: 6px;
    line-height: 1.4;
}
.tech-popup .stack-item .tool-name {
    font-family: 'JetBrains Mono', ui-monospace, Consolas, monospace;
    font-size: 12px;
    font-weight: 600;
    color: #0f172a;
    display: block;
}
.tech-popup .stack-item .tool-desc {
    font-size: 11.5px;
    color: #64748b;
    margin-top: 3px;
    display: block;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .tech-popup { background: #1e293b; border-color: #334155; color: #f1f5f9; }
    .tech-popup::before { background: #1e293b; border-color: #334155; }
    .tech-popup .specs, .tech-popup .spec { border-color: #334155; }
    .tech-popup .spec-label, .tech-popup .stack-label { color: #94a3b8; }
    .tech-popup .spec-value { color: #f8fafc; }
    .tech-popup .spec-sub { color: #64748b; }
    .tech-popup .stack { background: #0f172a; }
    .tech-popup .stack-item { background: #1e293b; border-color: #334155; }
    .tech-popup .stack-item .tool-name { color: #e2e8f0; }
    .tech-popup .stack-item .tool-desc { color: #94a3b8; }
}

/* ============================================================
 *  App Header
 * ============================================================ */
.app-header {
    padding: 18px 28px 18px 28px;
    margin: -8px -8px 16px -8px;
    border-bottom: 1px solid #e2e8f0;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}
.app-header .brand { display: flex; align-items: center; gap: 14px; }
.app-header .logo-mark {
    width: 38px; height: 38px;
    border-radius: 10px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: white;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: 800; letter-spacing: -0.02em;
}
.app-header .brand-text h1 {
    font-size: 17px !important; font-weight: 700 !important;
    margin: 0 !important; color: #0f172a !important;
    letter-spacing: -0.015em;
}
.app-header .brand-text .tagline {
    font-size: 12px; color: #64748b; margin-top: 1px;
    letter-spacing: 0.01em;
}
.app-header .header-meta {
    display: flex; gap: 16px; align-items: center;
    font-size: 11px; color: #64748b;
}
.app-header .meta-pill {
    background: #f1f5f9; border: 1px solid #e2e8f0;
    padding: 4px 10px; border-radius: 999px;
    font-weight: 500; color: #475569;
}
.app-header .meta-pill .dot {
    display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; background: #22c55e;
    margin-right: 5px; vertical-align: 1px;
}

/* ============================================================
 *  Sidebar Nav
 * ============================================================ */
.sidebar-col { padding-right: 8px !important; }
.sidebar-nav {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 10px;
    height: fit-content;
    position: sticky;
    top: 12px;
}
.sidebar-nav .cat-label {
    font-size: 10.5px;
    color: #94a3b8;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 10px 12px 6px;
}
.sidebar-nav .cat-label:first-child { padding-top: 4px; }
.sidebar-nav button.nav-btn {
    background: transparent !important;
    border: none !important;
    color: #334155 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 8px 12px !important;
    margin: 1px 0 !important;
    border-radius: 7px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100%;
    box-shadow: none !important;
    min-height: auto !important;
    transition: all 0.12s ease !important;
}
.sidebar-nav button.nav-btn:hover {
    background: #f1f5f9 !important;
    color: #0f172a !important;
}
.sidebar-nav button.nav-btn.active {
    background: #0f172a !important;
    color: #ffffff !important;
}

/* Dark mode for header / sidebar */
@media (prefers-color-scheme: dark) {
    .app-header { background: #0f172a; border-color: #1e293b; }
    .app-header .brand-text h1 { color: #f8fafc !important; }
    .app-header .brand-text .tagline { color: #94a3b8; }
    .app-header .meta-pill { background: #1e293b; border-color: #334155; color: #cbd5e1; }
    .sidebar-nav { background: #1e293b; border-color: #334155; }
    .sidebar-nav .cat-label { color: #64748b; }
    .sidebar-nav button.nav-btn { color: #cbd5e1 !important; }
    .sidebar-nav button.nav-btn:hover { background: #0f172a !important; color: #f8fafc !important; }
    .sidebar-nav button.nav-btn.active { background: #3b82f6 !important; color: #ffffff !important; }
}

/* ============================================================
 *  Polish — 統一 demo 內部視覺
 * ============================================================ */

/* 主要操作區卡片 */
.demo-input-pane, .demo-output-pane {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
@media (prefers-color-scheme: dark) {
    .demo-input-pane, .demo-output-pane { background: #1e293b; border-color: #334155; }
}

/* 主要 CTA 大按鈕 */
.demo-cta button {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%) !important;
    color: white !important;
    border: none !important;
    height: 52px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 10px -2px rgba(15,23,42,0.25) !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em;
}
.demo-cta button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 16px -4px rgba(15,23,42,0.35) !important;
}

/* 次要按鈕 */
.demo-secondary button {
    background: transparent !important;
    color: #475569 !important;
    border: 1px solid #cbd5e1 !important;
    height: 42px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}
.demo-secondary button:hover { background: #f1f5f9 !important; }

/* 進階設定摺疊 */
.demo-advanced {
    margin-top: 12px;
    border: 1px dashed #cbd5e1 !important;
    border-radius: 10px;
    background: transparent !important;
}
.demo-advanced > .label-wrap, .demo-advanced summary {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* 引導提示 */
.demo-hint {
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
    background: #f8fafc;
    border-left: 3px solid #3b82f6;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 8px 0;
}
@media (prefers-color-scheme: dark) {
    .demo-hint { background: #0f172a; color: #cbd5e1; }
}

/* Section 標題 */
.demo-section-title {
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 16px 0 10px 0 !important;
    padding-left: 10px;
    border-left: 4px solid #3b82f6;
}
@media (prefers-color-scheme: dark) {
    .demo-section-title { color: #f1f5f9 !important; }
}

/* Empty state 提示 */
.demo-empty {
    text-align: center;
    padding: 40px 20px;
    color: #94a3b8;
    font-size: 13px;
}
.demo-empty .icon { font-size: 36px; margin-bottom: 12px; opacity: 0.5; }

/* ============================================================
 *  全域防破版 / 排版整理
 * ============================================================ */

/* 主內容 column 內距更舒服 */
.gradio-container .block { box-sizing: border-box; }

/* 圖片元件:避免上傳框跟結果框高度不一致導致 row 破版 */
.gradio-container .image-frame { overflow: hidden !important; }
.gradio-container .image-container img { max-width: 100% !important; height: auto !important; }

/* Gallery 不要被 max-height 限制爛 */
.gradio-container .gallery-container { max-height: none !important; }
.gradio-container .gallery-container > div { box-sizing: border-box; }

/* Markdown 標題不要太大喧賓奪主 */
.gradio-container .markdown h3 { font-size: 16px !important; margin: 14px 0 8px !important; }
.gradio-container .markdown h4 { font-size: 14px !important; margin: 10px 0 6px !important; }
.gradio-container .markdown table {
    width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 13px;
}
.gradio-container .markdown table th,
.gradio-container .markdown table td {
    border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left;
}
.gradio-container .markdown table th { background: #f8fafc; font-weight: 600; }
.gradio-container .markdown blockquote {
    border-left: 3px solid #cbd5e1;
    padding: 4px 12px; margin: 8px 0;
    color: #475569; font-style: italic;
    background: #f8fafc; border-radius: 4px;
}

/* Code block 美化 */
.gradio-container .markdown code {
    background: #f1f5f9; color: #0f172a;
    padding: 2px 6px; border-radius: 4px;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px;
}
.gradio-container .markdown pre {
    background: #0f172a; color: #e2e8f0;
    padding: 12px 16px; border-radius: 8px;
    overflow-x: auto; font-size: 12.5px;
    line-height: 1.6;
}

/* 進階 accordion 收起時更低調 */
.gradio-container .demo-advanced .label-wrap { padding: 8px 12px !important; }

/* CTA + secondary 在 row 內保持平均 */
.gradio-container .demo-cta, .gradio-container .demo-secondary { padding: 0 !important; }

/* Dark mode tweaks */
@media (prefers-color-scheme: dark) {
    .gradio-container .markdown table th,
    .gradio-container .markdown table td { border-color: #334155; }
    .gradio-container .markdown table th { background: #0f172a; }
    .gradio-container .markdown blockquote {
        background: #1e293b; color: #94a3b8; border-color: #475569;
    }
    .gradio-container .markdown code { background: #1e293b; color: #e2e8f0; }
}

/* ============================================================
 *  Welcome / Landing 頁專用
 * ============================================================ */
.welcome-page { padding: 0 4px; }

.welcome-hero {
    background: linear-gradient(135deg, #0b1220 0%, #1a2436 60%, #2a3a55 100%);
    border-radius: 20px;
    padding: 56px 48px;
    color: #f8fafc;
    margin-bottom: 28px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.welcome-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 30% 20%, rgba(59,130,246,0.18) 0%, transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(236,72,153,0.12) 0%, transparent 50%);
    pointer-events: none;
}
.welcome-hero-content { position: relative; z-index: 2; }
.welcome-mark {
    width: 64px; height: 64px;
    margin: 0 auto 20px;
    border-radius: 14px;
    background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%);
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 32px; font-weight: 800;
    letter-spacing: -0.04em;
    box-shadow: 0 12px 32px -8px rgba(59,130,246,0.5);
}
.welcome-h1 {
    font-size: 38px !important;
    font-weight: 800 !important;
    margin: 0 0 10px 0 !important;
    letter-spacing: -0.03em;
    color: #f8fafc !important;
    line-height: 1.1;
}
.welcome-sub {
    font-size: 16px;
    color: #cbd5e1;
    margin: 0 0 22px;
    letter-spacing: 0.01em;
}
.welcome-meta {
    display: inline-flex; gap: 8px; flex-wrap: wrap; justify-content: center;
    max-width: 800px; margin: 0 auto;
}
.welcome-pill {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 6px 13px;
    border-radius: 999px;
    font-size: 12px;
    color: #e2e8f0;
    font-weight: 500;
}
.welcome-pill .dot {
    display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; background: #22c55e;
    margin-right: 5px; vertical-align: 1px;
    box-shadow: 0 0 8px #22c55e;
}

/* How to use steps */
.welcome-howto {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 28px;
}
@media (max-width: 800px) { .welcome-howto { grid-template-columns: 1fr; } }
.howto-step {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 18px;
    display: flex; align-items: center; gap: 14px;
    font-size: 13.5px;
    color: #334155;
    line-height: 1.5;
}
.howto-step strong { color: #0f172a; }
.howto-num {
    flex-shrink: 0;
    width: 32px; height: 32px;
    background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
    color: white;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700;
    font-size: 15px;
}

/* Category title */
.welcome-cat-title {
    font-size: 14px !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 28px 0 12px 0 !important;
    padding-left: 12px;
    border-left: 4px solid #3b82f6;
}

/* Demo cards grid */
.welcome-category { margin-bottom: 8px; }
.welcome-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 14px;
}
.welcome-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 18px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.welcome-card:hover {
    transform: translateY(-2px);
    border-color: #cbd5e1;
    box-shadow: 0 8px 16px -4px rgba(15,23,42,0.08);
}
.welcome-card-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 8px;
}
.welcome-icon {
    font-size: 28px;
    line-height: 1;
}
.welcome-cost {
    font-size: 11px;
    color: #64748b;
    font-weight: 500;
    background: #f1f5f9;
    padding: 3px 10px;
    border-radius: 999px;
}
.welcome-name {
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}
.welcome-tag {
    font-size: 12.5px;
    color: #64748b;
    line-height: 1.5;
    min-height: 38px;
}
.welcome-badges {
    margin-top: 10px;
    display: flex; gap: 5px; flex-wrap: wrap;
}
.welcome-badge {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #475569;
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 10.5px;
    font-weight: 500;
}

/* Tech stack section */
.welcome-stack { margin-top: 28px; }
.stack-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
}
.stack-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 8px;
    padding: 12px 16px;
}
.stack-item strong {
    display: block;
    font-size: 12px;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.stack-item span {
    font-size: 12.5px;
    color: #475569;
    line-height: 1.5;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
}

/* Footer */
.welcome-footer {
    margin-top: 32px;
    padding: 20px;
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    border-top: 1px solid #e2e8f0;
}
.welcome-sub-small {
    margin-top: 6px;
    font-size: 11px;
    opacity: 0.7;
}

/* Dark mode for welcome */
@media (prefers-color-scheme: dark) {
    .howto-step { background: #1e293b; border-color: #334155; color: #cbd5e1; }
    .howto-step strong { color: #f1f5f9; }
    .welcome-cat-title { color: #f1f5f9 !important; }
    .welcome-card { background: #1e293b; border-color: #334155; }
    .welcome-card:hover { border-color: #475569; }
    .welcome-name { color: #f8fafc; }
    .welcome-tag { color: #94a3b8; }
    .welcome-cost { background: #0f172a; color: #cbd5e1; }
    .welcome-badge { background: #0f172a; border-color: #334155; color: #94a3b8; }
    .stack-item { background: #1e293b; border-color: #334155; }
    .stack-item strong { color: #f1f5f9; }
    .stack-item span { color: #94a3b8; }
    .welcome-footer { color: #64748b; border-color: #334155; }
}
"""


def _esc(s: str) -> str:
    return html.escape(str(s)) if s else ""


def render_meta_header(
    *,
    icon: str,
    title: str,
    subtitle: str,
    tools: list[tuple[str, str]],
    cost: str,
    cost_detail: str = "",
    time: str,
    time_detail: str = "",
    badges: list[str] | None = None,
) -> None:
    """在 demo 頂部畫 hero + tech specs tooltip。

    Tooltip 預設隱藏,hover 或鍵盤 focus「📊 Tech Specs」按鈕後展開。
    內容包含執行時間、成本、AI 工具堆疊。
    """
    badges_html = ""
    if badges:
        badges_html = '<div class="badges">' + "".join(
            f'<span class="badge">{_esc(b)}</span>' for b in badges
        ) + '</div>'

    stack_items = "".join(
        f'<div class="stack-item">'
        f'  <span class="tool-name">{_esc(name)}</span>'
        f'  <span class="tool-desc">{_esc(desc)}</span>'
        f'</div>'
        for name, desc in tools
    )

    time_sub = f'<div class="spec-sub">{_esc(time_detail)}</div>' if time_detail else ""
    cost_sub = f'<div class="spec-sub">{_esc(cost_detail)}</div>' if cost_detail else ""

    hero_html = f"""
<div class="demo-hero">
  <div class="hero-top">
    <div class="hero-title">
      <h1>{_esc(icon)} {_esc(title)}</h1>
      <p class="subtitle">{_esc(subtitle)}</p>
    </div>
    <div class="tech-trigger" tabindex="0">
      <span class="tech-btn">📊 Tech Specs <span class="arrow">▾</span></span>
      <div class="tech-popup" role="tooltip">
        <div class="specs">
          <div class="spec">
            <div class="spec-label">⚡ 平均執行時間</div>
            <div class="spec-value">{_esc(time)}</div>
            {time_sub}
          </div>
          <div class="spec">
            <div class="spec-label">💰 平均成本 / 次</div>
            <div class="spec-value">{_esc(cost)}</div>
            {cost_sub}
          </div>
        </div>
        <div class="stack">
          <div class="stack-label">🤖 AI 工具堆疊</div>
          <div class="stack-list">{stack_items}</div>
        </div>
      </div>
    </div>
  </div>
  {badges_html}
</div>
"""
    gr.HTML(hero_html, elem_classes=["demo-hero-html"])
