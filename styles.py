def load_css(dark_mode=True):
    if dark_mode:
        bg        = "#0a0a0f"
        surface   = "#13131f"
        card_bg   = "#1a1a2e"
        border    = "rgba(129,140,248,0.18)"
        border2   = "rgba(129,140,248,0.35)"
        accent    = "#818cf8"
        accent2   = "#38bdf8"
        text      = "#e2e8f0"
        text_muted= "#64748b"
        text_dim  = "#94a3b8"
        logo_col  = "#a5b4fc"
        input_bg  = "#13131f"
        input_brd = "rgba(129,140,248,0.25)"
        hero_bg   = "linear-gradient(135deg,#1e1b4b 0%,#312e81 50%,#1e1b4b 100%)"
        stat_num  = "#a5b4fc"
        stat_lbl  = "rgba(224,231,255,0.45)"
        scroll_bg = "#0a0a0f"
        scroll_th = "#312e81"
        metric_bg = "#13131f"
        metric_v  = "#818cf8"
        exp_bg    = "#13131f"
        exp_sum   = "#e2e8f0"
        tab_inact = "#475569"
        tab_act   = "#818cf8"
        tab_brd   = "rgba(129,140,248,0.12)"
        prog_tr   = "rgba(255,255,255,0.06)"
        hr_col    = "rgba(129,140,248,0.12)"
        step_pend = "#1e1b4b"
        step_plbl = "#312e81"
        topnav_brd= "rgba(129,140,248,0.15)"
    else:
        bg        = "#fafafa"
        surface   = "#ffffff"
        card_bg   = "#ffffff"
        border    = "rgba(99,102,241,0.2)"
        border2   = "rgba(99,102,241,0.45)"
        accent    = "#6366f1"
        accent2   = "#0ea5e9"
        text      = "#0f172a"
        text_muted= "#64748b"
        text_dim  = "#334155"
        logo_col  = "#6366f1"
        input_bg  = "#ffffff"
        input_brd = "rgba(99,102,241,0.3)"
        hero_bg   = "linear-gradient(135deg,#4338ca 0%,#6366f1 50%,#4338ca 100%)"
        stat_num  = "#c7d2fe"
        stat_lbl  = "rgba(224,231,255,0.7)"
        scroll_bg = "#fafafa"
        scroll_th = "#a5b4fc"
        metric_bg = "#ffffff"
        metric_v  = "#6366f1"
        exp_bg    = "#ffffff"
        exp_sum   = "#0f172a"
        tab_inact = "#475569"
        tab_act   = "#6366f1"
        tab_brd   = "rgba(99,102,241,0.15)"
        prog_tr   = "rgba(99,102,241,0.1)"
        hr_col    = "rgba(99,102,241,0.15)"
        step_pend = "#ede9fe"
        step_plbl = "#6366f1"
        topnav_brd= "rgba(99,102,241,0.2)"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Space Grotesk', sans-serif !important;
    background: {bg} !important;
    color: {text} !important;
}}
h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    color: {text} !important;
}}
p, span, li, div, label, caption {{
    color: {text} !important;
}}
/* Force Streamlit's own text elements */
.stMarkdown, .stMarkdown p, .stMarkdown span,
.stText, [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {{
    color: {text} !important;
}}
#MainMenu, footer, header, .stDeployButton {{ display: none !important; }}
.main .block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 900px !important;
}}

/* ── Inputs ── */
input[type="text"],
input[type="password"],
input[type="email"],
input[type="number"],
textarea,
.stTextInput input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {{
    background: {input_bg} !important;
    color: {text} !important;
    border: 2px solid {input_brd} !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important;
    caret-color: {accent} !important;
}}
input[type="text"]:focus, input[type="password"]:focus, .stTextInput input:focus {{
    border-color: {accent} !important;
    box-shadow: 0 0 0 3px {border} !important;
    outline: none !important;
}}
input::placeholder {{ color: {text_muted} !important; }}

/* Fix password eye icon visibility */
[data-testid="stTextInput"] button {{
    background: transparent !important;
    border: none !important;
}}
[data-testid="stTextInput"] button svg {{
    color: {accent} !important;
    fill: {accent} !important;
    stroke: {accent} !important;
}}

/* Number input buttons */
[data-testid="stNumberInput"] button {{
    background: {card_bg} !important;
    border: 1px solid {border} !important;
    color: {text} !important;
    border-radius: 8px !important;
}}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {{
    background: {input_bg} !important;
    border: 2px solid {input_brd} !important;
    border-radius: 12px !important;
    color: {text} !important;
    font-family: 'Space Grotesk', sans-serif !important;
}}
[data-testid="stSelectbox"] > div > div > div {{ color: {text} !important; }}
[data-testid="stSelectbox"] svg {{ color: {accent} !important; }}
[data-baseweb="popover"], [role="listbox"] {{ background: {card_bg} !important; }}
[role="option"] {{
    background: {card_bg} !important;
    color: {text} !important;
    font-family: 'Space Grotesk', sans-serif !important;
}}
[role="option"]:hover {{ background: {border} !important; }}

/* ── Radio & Checkboxes ── */
[data-testid="stRadio"] label {{
    color: {text_dim} !important;
    font-size: 15px !important;
    font-family: 'Space Grotesk', sans-serif !important;
}}
[data-testid="stCheckbox"] label {{
    color: {text} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important;
}}

/* ── Buttons ── */
.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {{
    background: linear-gradient(135deg, {accent}, {accent2}) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    min-height: 50px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px {border} !important;
}}
.stButton > button[kind="primary"]:hover {{
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px {border2} !important;
}}
.stButton > button[kind="secondary"],
button[data-testid="baseButton-secondary"] {{
    background: {border} !important;
    color: {accent} !important;
    border: 2px solid {border2} !important;
    border-radius: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}}
.stButton > button[kind="secondary"]:hover {{
    background: {border2} !important;
    transform: translateY(-1px) !important;
}}
.stButton > button {{
    background: {card_bg} !important;
    color: {text_dim} !important;
    border: 2px solid {border} !important;
    border-radius: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    border-color: {accent} !important;
    color: {accent} !important;
    transform: translateY(-1px) !important;
}}

/* Specific fix for Camera Input 'Take Photo' button white text issue */
[data-testid="stCameraInputButton"],
[data-testid="stCameraInput"] button {{
    background-color: {accent} !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent !important;
    border-bottom: 2px solid {tab_brd} !important;
    gap: 8px !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {tab_inact} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 24px !important;
}}
.stTabs [aria-selected="true"] {{
    color: {tab_act} !important;
    border-bottom: 3px solid {tab_act} !important;
    background: {border} !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {exp_bg} !important;
    border: 2px solid {border} !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}}
[data-testid="stExpander"] summary {{
    background: {exp_bg} !important;
    color: {exp_sum} !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 16px 20px !important;
}}
[data-testid="stExpander"] summary:hover {{ background: {border} !important; }}
[data-testid="stExpander"] svg {{ color: {accent} !important; }}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {metric_bg} !important;
    border: 2px solid {border} !important;
    border-radius: 16px !important;
    padding: 20px !important;
}}
[data-testid="stMetricValue"] {{
    color: {metric_v} !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
}}
[data-testid="stMetricLabel"] {{
    color: {text_muted} !important;
    font-size: 12px !important;
}}

/* ── Alerts, file uploader, progress ── */
[data-testid="stAlert"] {{ border-radius: 14px !important; font-family: 'Space Grotesk', sans-serif !important; }}
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] *,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzone"] div {{
    background: {input_bg} !important;
    background-color: {input_bg} !important;
    color: {text} !important;
}}
[data-testid="stFileUploader"] {{
    border: 2px dashed {border2} !important;
    border-radius: 16px !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color: {accent} !important; }}
[data-testid="stFileUploader"] button {{
    background: {card_bg} !important;
    color: {accent} !important;
    border: 2px solid {border2} !important;
    border-radius: 10px !important;
}}
[data-testid="stDownloadButton"] button {{
    background: linear-gradient(135deg, {accent}, {accent2}) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    width: 100% !important;
}}
[data-testid="stDownloadButton"] button:hover {{
    opacity: 0.9 !important;
    transform: translateY(-2px) !important;
}}
.stProgress > div > div {{ background: linear-gradient(90deg, {accent}, {accent2}) !important; border-radius: 4px !important; }}
.stProgress > div {{ background: {prog_tr} !important; border-radius: 4px !important; }}
hr {{ border-color: {hr_col} !important; }}
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {scroll_bg}; }}
::-webkit-scrollbar-thumb {{ background: {scroll_th}; border-radius: 3px; }}
/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {surface} !important;
    border-right: 2px solid {border} !important;
}}
[data-testid="stSidebar"] * {{
    color: {text} !important;
}}
[data-testid="stSidebar"] .stButton > button {{
    background: {card_bg} !important;
    color: {text} !important;
    border: 2px solid {border} !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    width: 100% !important;
    text-align: left !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    border-color: {accent} !important;
    color: {accent} !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {accent}, {accent2}) !important;
    color: #ffffff !important;
    border: none !important;
}}
[data-testid="stSidebar"] [data-testid="stMetric"] {{
    background: {card_bg} !important;
    border: 1px solid {border} !important;
    border-radius: 12px !important;
    padding: 12px !important;
}}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {{
    color: {metric_v} !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: {hr_col} !important;
}}

/* ══ Custom components ══ */
.nayana-hero {{
    background: {hero_bg};
    border-radius: 28px;
    padding: 56px 48px;
    text-align: center;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px {border2};
}}
.nayana-hero::before {{
    content: '';
    position: absolute;
    top: -80px; left: 50%;
    transform: translateX(-50%);
    width: 500px; height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}}
.nayana-wordmark {{
    font-family: 'Space Mono', monospace;
    font-size: 68px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 10px;
    text-shadow: 0 2px 20px rgba(0,0,0,0.3);
}}
.nayana-meaning {{
    font-size: 12px;
    color: rgba(255,255,255,0.55);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 16px;
    font-style: italic;
}}
.nayana-tagline {{
    font-size: 17px;
    color: rgba(255,255,255,0.8);
    font-weight: 400;
    max-width: 500px;
    margin: 0 auto 36px;
    line-height: 1.8;
}}
.stat-row {{ display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; margin-bottom: 40px; }}
.stat-item {{ text-align: center; }}
.stat-num {{ font-family: 'Space Mono', monospace; font-size: 30px; font-weight: 700; color: {stat_num}; line-height: 1; }}
.stat-lbl {{ font-size: 11px; color: {stat_lbl}; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 5px; }}

.portal-card {{
    background: {card_bg};
    border-radius: 20px;
    padding: 36px 24px;
    text-align: center;
    transition: all 0.2s;
    border: 2px solid {border};
    width: 100%;
}}
.portal-card:hover {{ transform: translateY(-4px); border-color: {accent}; box-shadow: 0 8px 30px {border2}; }}
.portal-icon {{ font-size: 44px; margin-bottom: 16px; }}
.portal-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: {text}; margin-bottom: 10px; }}
.portal-sub {{ font-size: 13px; color: {text_muted}; line-height: 1.6; }}

.topnav {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 0 18px;
    border-bottom: 2px solid {topnav_brd};
    margin-bottom: 32px;
}}
.topnav-brand {{ font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; color: {logo_col}; letter-spacing: -1px; }}
.topnav-user {{ font-size: 13px; color: {text_muted}; font-weight: 500; }}

.page-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 30px; font-weight: 700; color: {text}; letter-spacing: -0.5px; margin-bottom: 4px; }}
.page-sub {{ font-size: 15px; color: {text_muted}; font-weight: 400; margin-bottom: 28px; }}

.card {{ background: {card_bg}; border: 2px solid {border}; border-radius: 18px; padding: 24px; margin-bottom: 20px; }}
.card.highlight {{ border-color: {accent}; background: {border}; }}
.card.danger {{ border-color: #f87171; }}
.card.warning {{ border-color: #fb923c; }}

.quality-num {{ font-family: 'Space Mono', monospace; font-size: 54px; font-weight: 700; line-height: 1; margin-bottom: 6px; }}

.risk-pill {{ display: inline-block; padding: 6px 18px; border-radius: 20px; font-size: 13px; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }}
.risk-high {{ background: rgba(248,113,113,0.15); color: #fca5a5; border: 2px solid rgba(248,113,113,0.3); }}
.risk-moderate {{ background: rgba(251,146,60,0.15); color: #fdba74; border: 2px solid rgba(251,146,60,0.3); }}
.risk-low {{ background: rgba(52,211,153,0.15); color: #6ee7b7; border: 2px solid rgba(52,211,153,0.3); }}

.status-pending {{ background: rgba(251,146,60,0.15); color: #fdba74; border: 2px solid rgba(251,146,60,0.3); padding: 3px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; }}
.status-reviewed {{ background: rgba(52,211,153,0.15); color: #6ee7b7; border: 2px solid rgba(52,211,153,0.3); padding: 3px 12px; border-radius: 20px; font-size: 11px; font-weight: 700; }}

.step-bar {{ display: flex; align-items: center; margin-bottom: 32px; background: {card_bg}; border: 2px solid {border}; border-radius: 16px; padding: 16px 24px; }}
.step {{ display: flex; align-items: center; gap: 10px; flex: 1; }}
.step-dot {{ width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; font-family: 'Space Mono', monospace; flex-shrink: 0; }}
.step-dot.done {{ background: {accent}; color: white; }}
.step-dot.active {{ background: {accent2}; color: white; box-shadow: 0 0 0 4px {border}; }}
.step-dot.pending {{ background: {step_pend}; color: {step_plbl}; }}
.step-label {{ font-size: 13px; font-weight: 600; font-family: 'Space Grotesk', sans-serif; }}
.step-label.done {{ color: {accent}; }}
.step-label.active {{ color: {text}; }}
.step-label.pending {{ color: {text_muted}; }}
.step-line {{ flex: 1; height: 2px; background: {border}; margin: 0 8px; max-width: 40px; }}
.step-line.done {{ background: {accent}; }}

.section-label {{ font-family: 'Space Grotesk', sans-serif; font-size: 11px; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: {text_muted}; margin-bottom: 10px; }}

.empty-state {{ text-align: center; padding: 64px 32px; background: {card_bg}; border: 2px dashed {border}; border-radius: 22px; }}
.empty-icon {{ font-size: 52px; margin-bottom: 16px; opacity: 0.5; }}
.empty-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 22px; font-weight: 700; color: {text}; margin-bottom: 8px; }}
.empty-sub {{ font-size: 14px; color: {text_muted}; }}

.doc-card {{ background: {card_bg}; border: 2px solid {border}; border-radius: 16px; padding: 20px; }}
.doc-name {{ font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 700; color: {text}; }}
.doc-meta {{ font-size: 13px; color: {text_muted}; margin-top: 4px; }}
</style>
"""
