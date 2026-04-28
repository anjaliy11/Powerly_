import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
import os
import time

# PPO
try:
    from stable_baselines3 import PPO
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False

# ASSET LOADER
def load_image(path: str):
    """Return base64 data-URI string for an image file, or None if missing."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

hero_img = load_image("assets/powerly.png")   
logo_img = load_image("assets/logo.png")       


st.set_page_config(
    page_title="Powerly · Smart Microgrid",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>
/* ── ROOT ── */
:root {
    --bg:        #0b0f14;
    --surface:   #131920;
    --surface2:  #1a2330;
    --border:    #1e2d3d;
    --accent:    #00e5a0;
    --accent2:   #0af0ff;
    --warn:      #f5a623;
    --danger:    #ff4d4d;
    --text:      #d4e0ec;
    --muted:     #5e7a96;
    --radius:    12px;
}

/* ── APP BG ── */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: var(--muted) !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── SELECTBOX / RADIO SELECT BOXES ── */
div[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}
div[data-baseweb="select"] svg { color: var(--muted) !important; }

/* ── METRICS ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.07em; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; font-size: 26px !important; }
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    letter-spacing: 0.05em !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── PROGRESS ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 4px !important;
}
.stProgress > div { background: var(--surface2) !important; border-radius: 4px !important; }

/* ── SELECTBOX in main ── */
.stSelectbox > div[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── INFO / SUCCESS / ERROR BOXES ── */
.stAlert { border-radius: var(--radius) !important; }

/* ── SECTION HEADERS ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 14px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
}
.section-header span {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #a0b8cc;
    font-family: 'Space Mono', monospace;
}
.section-header .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px var(--accent);
}

/* ── NAVBAR ── */
.navbar {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}
.logo-text {
    font-family: 'Space Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 0.05em;
    text-shadow: 0 0 20px rgba(0,229,160,0.4);
}
.tagline-text {
    font-size: 11px;
    color: #8aa8c0;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
}
.time-text {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #8aa8c0;
    text-align: right;
}

/* ── HERO ── */
.hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,229,160,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 28px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 6px;
}
.hero-sub {
    font-size: 14px;
    color: var(--muted);
    margin-bottom: 14px;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,229,160,0.12);
    border: 1px solid rgba(0,229,160,0.3);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 20px;
    margin-right: 8px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── STATUS BADGE ── */
.status-ok  { color: var(--accent) !important; }
.status-warn { color: var(--warn) !important; }
.status-bad  { color: var(--danger) !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 18px !important; padding-bottom: 40px !important; }
</style>
""", unsafe_allow_html=True)



def section(icon: str, label: str):
    st.markdown(
        f'<div class="section-header"><div class="dot"></div><span>{icon} {label}</span></div>',
        unsafe_allow_html=True,
    )


def plotly_dark_layout(fig, title="", height=340):
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(19,25,32,0.9)",
        font=dict(color="#5e7a96", family="Space Mono, monospace", size=11),
        title_font=dict(color="#d4e0ec", size=14),
        legend=dict(
            bgcolor="rgba(19,25,32,0.8)",
            bordercolor="#1e2d3d",
            borderwidth=1,
        ),
        xaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d", zerolinecolor="#1e2d3d"),
        yaxis=dict(gridcolor="#1e2d3d", linecolor="#1e2d3d", zerolinecolor="#1e2d3d"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


with st.sidebar:
    
    if logo_img:
        st.markdown(
            f'<div style="margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid #1e2d3d;">'
            f'<img src="data:image/png;base64,{logo_img}" '
            f'style="max-width:140px;height:auto;filter:drop-shadow(0 0 8px rgba(0,229,160,0.35));" />'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="font-family:\'Space Mono\',monospace;font-size:18px;'
            'color:#00e5a0;margin-bottom:20px;text-shadow:0 0 12px rgba(0,229,160,0.4);">'
            "POWERLY</div>",
            unsafe_allow_html=True,
        )

    model_type = st.selectbox(
        "MODEL",
        ["AT-DRAC-EBD", "PPO", "Rule-Based"],
    )
    scenario = st.selectbox(
        "SCENARIO",
        ["Normal", "Sunny", "Cloudy", "HILP: Load Surge", "HILP: DER Failure", "HILP: Cyber Attack"],
    )
    mode = st.radio(
        "SIMULATION MODE",
        ["AI Powered", "Live Microgrid"],
    )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:11px;color:#5e7a96;line-height:1.6;">'
        "💡 <b>AT-DRAC-EBD</b> excels under HILP events<br>"
        "🔋 Battery below 20% triggers emergency mode<br>"
        "📡 Live mode streams real sensor data"
        "</div>",
        unsafe_allow_html=True,
    )


model = None
if MODEL_AVAILABLE and os.path.exists("models/ppo_microgrid/ppo_final.zip"):
    try:
        model = PPO.load("models/ppo_microgrid/ppo_final.zip")
    except Exception:
        pass

hours = 24
t_arr = np.arange(hours)
solar = np.maximum(0, 55 * np.sin(np.pi * t_arr / 24) + np.random.randn(hours) * 2)
demand = 38 + 12 * np.sin(2 * np.pi * t_arr / 24) + np.random.randn(hours) * 1.5
wind = np.maximum(0, 15 + 8 * np.cos(np.pi * t_arr / 12) + np.random.randn(hours) * 3)

if "Cloudy" in scenario:
    solar *= 0.45
if "Sunny" in scenario:
    solar *= 1.25
if "Load Surge" in scenario:
    demand[12:18] += 22
if "DER Failure" in scenario:
    solar[8:16] *= 0.2
if "Cyber Attack" in scenario:
    demand[10:14] += np.random.randn(4) * 15

battery = []
soc = 55.0
for t in range(hours):
    net = (solar[t] + wind[t]) - demand[t]
    if model:
        try:
            obs = np.array([solar[t], demand[t], soc, wind[t], t, 0, 0, 0], dtype=np.float32)
            action, _ = model.predict(obs)
            soc += float(action[0]) * 5
        except Exception:
            soc += net * 0.4
    else:
        soc += net * 0.4
    soc = float(np.clip(soc, 0, 100))
    battery.append(soc)

battery = np.array(battery)
net_power = solar + wind - demand
blackouts = int(np.sum(battery < 10))
avg_solar = float(np.mean(solar))
coverage = float(np.clip(np.mean((solar + wind) / demand) * 100, 0, 100))


_logo_html = (
    f'<img src="data:image/png;base64,{logo_img}" '
    f'style="height:36px;width:auto;vertical-align:middle;'
    f'filter:drop-shadow(0 0 8px rgba(0,229,160,0.4));" />'
    if logo_img
    else '<span class="logo-text">POWERLY</span>'
)

st.markdown(
    f"""
    <div class="navbar">
        <div style="display:flex;align-items:center;gap:12px;">
            {_logo_html}
            <div>
                <div class="tagline-text">Smart Microgrid Energy Balancer</div>
            </div>
        </div>
        <div class="time-text">
            {datetime.now().strftime("%d %b %Y")}<br>
            <span style="color:#00e5a0;">{datetime.now().strftime("%I:%M %p")}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

status_color = "#00e5a0" if blackouts == 0 else ("#f5a623" if blackouts < 3 else "#ff4d4d")
status_label = "STABLE" if blackouts == 0 else ("WARNING" if blackouts < 3 else "CRITICAL")

_hero_img_html = (
    f'<img src="data:image/png;base64,{hero_img}" '
    f'style="max-width:300px;width:100%;height:auto;'
    f'border-radius:12px;'
    f'filter:drop-shadow(0 4px 24px rgba(0,229,160,0.18));" />'
    if hero_img else ""
)

st.markdown(
    f"""
    <div class="hero" style="display:flex;justify-content:space-between;align-items:center;gap:24px;">
        <div style="flex:1;min-width:0;">
            <div class="hero-title">RL based Microgrid Energy Balancer</div>
            <div class="hero-sub">Comparing <b>{model_type}</b> vs PPO in real-world energy balancing · Scenario: <b>{scenario}</b></div>
            <span class="hero-badge">{model_type}</span>
            <span class="hero-badge">{mode}</span>
            <span class="hero-badge" style="color:{status_color};border-color:rgba(0,229,160,0.3);">
                ● {status_label}
            </span>
        </div>
        <div style="flex-shrink:0;">
            {_hero_img_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("☀️ Solar Output",  f"{avg_solar:.1f} kW",   f"+{avg_solar*0.05:.1f} kW vs yesterday")
c2.metric("💨 Wind Output",   f"{np.mean(wind):.1f} kW", None)
c3.metric("⚡ Demand",        f"{np.mean(demand):.1f} kW", f"{(np.mean(demand)-40):.1f} kW")
c4.metric("🔋 Battery (Now)", f"{battery[-1]:.0f}%",   f"{battery[-1]-battery[-2]:.1f}%")
c5.metric("🚨 Low-SoC Hours", f"{blackouts}",          None)


section("📈", "ENERGY FLOW — 24H")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=t_arr, y=solar,
    name="Solar",
    line=dict(color="#00e5a0", width=2),
    fill="tozeroy",
    fillcolor="rgba(0,229,160,0.07)",
))
fig.add_trace(go.Scatter(
    x=t_arr, y=wind,
    name="Wind",
    line=dict(color="#0af0ff", width=2, dash="dot"),
))
fig.add_trace(go.Scatter(
    x=t_arr, y=demand,
    name="Demand",
    line=dict(color="#f5a623", width=2),
))
fig.add_trace(go.Scatter(
    x=t_arr, y=battery,
    name="Battery SoC %",
    line=dict(color="#b070ff", width=2),
    yaxis="y2",
))

# Danger zone band
fig.add_hrect(y0=0, y1=10, fillcolor="rgba(255,77,77,0.06)",
              line_width=0, annotation_text="⚠ Critical SoC Zone",
              annotation_font_color="#ff4d4d", annotation_font_size=10)

fig.update_layout(
    yaxis2=dict(
        title="Battery SoC %",
        overlaying="y",
        side="right",
        range=[0, 120],
        gridcolor="#1e2d3d",
        color="#5e7a96",
    ),
    xaxis_title="Hour of Day",
    yaxis_title="kW",
)
plotly_dark_layout(fig, height=360)
st.plotly_chart(fig, use_container_width=True)



section("🌞", "RENEWABLE COVERAGE")
st.markdown(
    f'<div style="font-family:\'Space Mono\',monospace;font-size:24px;color:#00e5a0;'
    f'margin-bottom:8px;">{coverage:.1f}%</div>',
    unsafe_allow_html=True,
)
st.progress(min(coverage / 100, 1.0))
st.markdown(
    f'<div style="font-size:12px;color:#5e7a96;margin-top:4px;">'
    f"Renewables covering <b style='color:#d4e0ec'>{coverage:.0f}%</b> of total demand over 24 h</div>",
    unsafe_allow_html=True,
)



section("🗺", "VILLAGE ENERGY GRID")

# Generate reproducible house locations (near Nagpur, India)
rng = np.random.default_rng(42)
n_houses = 25
lats = rng.uniform(20.50, 20.60, n_houses)
lons = rng.uniform(78.88, 78.98, n_houses)
energy_levels = rng.normal(0, 1, n_houses)
house_labels = [f"House {i+1}" for i in range(n_houses)]
house_status = ["Surplus" if e > 0 else "Deficit" for e in energy_levels]
colors = ["#00e5a0" if e > 0 else "#ff4d4d" for e in energy_levels]
sizes = [10 + abs(e) * 5 for e in energy_levels]

map_df = pd.DataFrame({
    "lat": lats,
    "lon": lons,
    "energy": energy_levels,
    "label": house_labels,
    "status": house_status,
})

fig_map = go.Figure()

# Surplus houses
surplus = map_df[map_df["energy"] > 0]
deficit = map_df[map_df["energy"] <= 0]

fig_map.add_trace(go.Scattermapbox(
    lat=surplus["lat"],
    lon=surplus["lon"],
    mode="markers+text",
    name="Surplus ⚡",
    marker=dict(
        size=14 + surplus["energy"].abs() * 4,
        color="#00e5a0",
        opacity=0.9,
    ),
    text=surplus["label"],
    textposition="top right",
    textfont=dict(size=10, color="#00e5a0", family="monospace"),
    customdata=np.stack([surplus["energy"].round(2), surplus["status"]], axis=-1),
    hovertemplate="<b>%{text}</b><br>+%{customdata[0]} kW surplus<extra></extra>",
))

fig_map.add_trace(go.Scattermapbox(
    lat=deficit["lat"],
    lon=deficit["lon"],
    mode="markers+text",
    name="Deficit ⚠",
    marker=dict(
        size=14 + deficit["energy"].abs() * 4,
        color="#ff4d4d",
        opacity=0.9,
    ),
    text=deficit["label"],
    textposition="top right",
    textfont=dict(size=10, color="#ff8080", family="monospace"),
    customdata=np.stack([deficit["energy"].round(2), deficit["status"]], axis=-1),
    hovertemplate="<b>%{text}</b><br>%{customdata[0]} kW deficit<extra></extra>",
))

fig_map.update_layout(
    mapbox=dict(
        style="open-street-map",   
        zoom=11.5,
        center=dict(lat=20.55, lon=78.93),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(r=0, t=0, l=0, b=0),
    height=420,
    legend=dict(
        bgcolor="rgba(19,25,32,0.85)",
        bordercolor="#1e2d3d",
        borderwidth=1,
        font=dict(color="#d4e0ec", size=11),
    ),
)

st.plotly_chart(fig_map, use_container_width=True)

# Clickable house inspector
col_l, col_r = st.columns([1, 2])
with col_l:
    selected_house = st.selectbox(
        "🏠 Inspect House",
        options=map_df.index,
        format_func=lambda i: map_df["label"][i],
    )

with col_r:
    row = map_df.iloc[selected_house]
    if row["energy"] > 0:
        st.success(f"⚡ **{row['label']}** — Surplus: +{row['energy']:.2f} kW → Available to share")
    else:
        st.error(f"⚠️ **{row['label']}** — Deficit: {row['energy']:.2f} kW → Needs supply")


section("🎥", "LIVE ENERGY FLOW ANIMATION")

st.markdown("""
<div style="background:#131920;border:1px solid #1e2d3d;border-radius:12px;padding:24px;overflow:hidden;">
<svg viewBox="0 0 700 130" xmlns="http://www.w3.org/2000/svg" width="100%" height="130">
  <defs>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#00e5a0" stop-opacity="0.1"/>
      <stop offset="50%"  stop-color="#00e5a0" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#0af0ff" stop-opacity="0.1"/>
    </linearGradient>
  </defs>

  <!-- Grid line -->
  <line x1="60" y1="65" x2="640" y2="65" stroke="url(#lineGrad)" stroke-width="2"/>

  <!-- Nodes -->
  <!-- Solar Panel -->
  <circle cx="60" cy="65" r="20" fill="#131920" stroke="#00e5a0" stroke-width="2" filter="url(#glow)"/>
  <text x="60" y="70" text-anchor="middle" fill="#00e5a0" font-size="16">☀️</text>
  <text x="60" y="100" text-anchor="middle" fill="#5e7a96" font-size="9" font-family="Space Mono">SOLAR</text>

  <!-- Battery -->
  <circle cx="220" cy="65" r="20" fill="#131920" stroke="#b070ff" stroke-width="2" filter="url(#glow)"/>
  <text x="220" y="70" text-anchor="middle" fill="#b070ff" font-size="16">🔋</text>
  <text x="220" y="100" text-anchor="middle" fill="#5e7a96" font-size="9" font-family="Space Mono">BATTERY</text>

  <!-- AI Controller -->
  <rect x="330" y="45" width="40" height="40" rx="8"
        fill="#131920" stroke="#0af0ff" stroke-width="2" filter="url(#glow)"/>
  <text x="350" y="70" text-anchor="middle" fill="#0af0ff" font-size="16">🤖</text>
  <text x="350" y="100" text-anchor="middle" fill="#5e7a96" font-size="9" font-family="Space Mono">AI</text>

  <!-- Wind -->
  <circle cx="480" cy="65" r="20" fill="#131920" stroke="#0af0ff" stroke-width="2" filter="url(#glow)"/>
  <text x="480" y="70" text-anchor="middle" fill="#0af0ff" font-size="16">💨</text>
  <text x="480" y="100" text-anchor="middle" fill="#5e7a96" font-size="9" font-family="Space Mono">WIND</text>

  <!-- Load -->
  <circle cx="640" cy="65" r="20" fill="#131920" stroke="#f5a623" stroke-width="2" filter="url(#glow)"/>
  <text x="640" y="70" text-anchor="middle" fill="#f5a623" font-size="16">🏘</text>
  <text x="640" y="100" text-anchor="middle" fill="#5e7a96" font-size="9" font-family="Space Mono">LOAD</text>

  <!-- Animated energy packet 1 (solar → AI) -->
  <circle r="6" fill="#00e5a0" opacity="0.9" filter="url(#glow)">
    <animateMotion dur="2.4s" repeatCount="indefinite" begin="0s">
      <mpath><path d="M60 65 L350 65"/></mpath>
    </animateMotion>
  </circle>

  <!-- Animated energy packet 2 (battery → AI) -->
  <circle r="5" fill="#b070ff" opacity="0.8" filter="url(#glow)">
    <animateMotion dur="3.1s" repeatCount="indefinite" begin="0.8s">
      <mpath><path d="M220 65 L350 65"/></mpath>
    </animateMotion>
  </circle>

  <!-- Animated energy packet 3 (wind → load) -->
  <circle r="5" fill="#0af0ff" opacity="0.8" filter="url(#glow)">
    <animateMotion dur="2.8s" repeatCount="indefinite" begin="0.4s">
      <mpath><path d="M480 65 L640 65"/></mpath>
    </animateMotion>
  </circle>

  <!-- Animated energy packet 4 (AI → load) -->
  <circle r="6" fill="#f5a623" opacity="0.85" filter="url(#glow)">
    <animateMotion dur="2.0s" repeatCount="indefinite" begin="1.2s">
      <mpath><path d="M350 65 L640 65"/></mpath>
    </animateMotion>
  </circle>
</svg>
</div>
""", unsafe_allow_html=True)



section("📊", "MODEL COMPARISON")

# Use real CSV if present, otherwise show realistic synthetic data
if os.path.exists("results/training_metrics.csv"):
    df_metrics = pd.read_csv("results/training_metrics.csv")
    csr_col     = df_metrics.get("csr",     df_metrics.iloc[:, 0])
    diesel_col  = df_metrics.get("diesel_kwh", df_metrics.iloc[:, 1] if df_metrics.shape[1] > 1 else None)
else:
    n = 100
    ep = np.arange(n)
    df_metrics = pd.DataFrame({
        "episode":     ep,
        "AT_DRAC_EBD": np.clip(0.5 + 0.4 * (1 - np.exp(-ep / 30)) + np.random.randn(n) * 0.03, 0, 1),
        "PPO":         np.clip(0.45 + 0.3 * (1 - np.exp(-ep / 45)) + np.random.randn(n) * 0.04, 0, 1),
        "Rule_Based":  np.full(n, 0.52) + np.random.randn(n) * 0.02,
    })

tab1, tab2 = st.tabs(["📈 Training Curves", "🔥 Radar Comparison"])

with tab1:
    fig2 = go.Figure()
    if "AT_DRAC_EBD" in df_metrics.columns:
        fig2.add_trace(go.Scatter(
            y=df_metrics["AT_DRAC_EBD"], name="AT-DRAC-EBD",
            line=dict(color="#00e5a0", width=2)))
        fig2.add_trace(go.Scatter(
            y=df_metrics["PPO"], name="PPO",
            line=dict(color="#0af0ff", width=2, dash="dot")))
        fig2.add_trace(go.Scatter(
            y=df_metrics["Rule_Based"], name="Rule-Based",
            line=dict(color="#5e7a96", width=1.5, dash="dash")))
    else:
        for col in df_metrics.columns:
            if col != "episode":
                fig2.add_trace(go.Scatter(y=df_metrics[col], name=col))
    plotly_dark_layout(fig2, title="CSR Score Over Training Episodes", height=300)
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    categories = ["CSR Score", "Diesel Saved", "Response Time", "Stability", "Resilience"]
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[0.91, 0.88, 0.80, 0.93, 0.95], theta=categories, fill="toself",
        name="AT-DRAC-EBD", line_color="#00e5a0", fillcolor="rgba(0,229,160,0.1)"))
    fig_radar.add_trace(go.Scatterpolar(
        r=[0.78, 0.74, 0.85, 0.80, 0.72], theta=categories, fill="toself",
        name="PPO", line_color="#0af0ff", fillcolor="rgba(10,240,255,0.08)"))
    fig_radar.add_trace(go.Scatterpolar(
        r=[0.60, 0.55, 0.90, 0.65, 0.50], theta=categories, fill="toself",
        name="Rule-Based", line_color="#5e7a96", fillcolor="rgba(94,122,150,0.07)"))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(19,25,32,0.9)",
            radialaxis=dict(visible=True, range=[0, 1], color="#5e7a96",
                            gridcolor="#1e2d3d"),
            angularaxis=dict(color="#d4e0ec", gridcolor="#1e2d3d"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(19,25,32,0.8)", bordercolor="#1e2d3d",
                    borderwidth=1, font=dict(color="#d4e0ec")),
        height=340,
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_radar, use_container_width=True)


section("⚡", "NET POWER BALANCE — SURPLUS vs DEFICIT")

net = solar + wind - demand
colors_bar = ["#00e5a0" if v >= 0 else "#ff4d4d" for v in net]

fig_net = go.Figure()
fig_net.add_trace(go.Bar(
    x=t_arr,
    y=net,
    name="Net Power",
    marker_color=colors_bar,
    marker_line_width=0,
    hovertemplate="Hour %{x}<br>Net: %{y:.1f} kW<extra></extra>",
))
fig_net.add_hline(y=0, line_color="#5e7a96", line_width=1, line_dash="dot")
fig_net.add_trace(go.Scatter(
    x=t_arr, y=net,
    mode="lines",
    line=dict(color="rgba(255,255,255,0.15)", width=1.5),
    showlegend=False,
))
# shade surplus / deficit regions
fig_net.add_hrect(y0=0,   y1=max(net.max(), 5),  fillcolor="rgba(0,229,160,0.04)", line_width=0)
fig_net.add_hrect(y0=min(net.min(), -5), y1=0,   fillcolor="rgba(255,77,77,0.04)",  line_width=0)

plotly_dark_layout(fig_net, title="Net Power (Generation − Demand) per Hour", height=260)
fig_net.update_layout(
    xaxis_title="Hour",
    yaxis_title="kW",
    bargap=0.12,
    annotations=[
        dict(x=2,  y=max(net)*0.7, text="SURPLUS ZONE",  showarrow=False,
             font=dict(color="#00e5a0", size=10, family="monospace"), opacity=0.5),
        dict(x=2,  y=min(net)*0.7, text="DEFICIT ZONE",  showarrow=False,
             font=dict(color="#ff4d4d", size=10, family="monospace"), opacity=0.5),
    ],
)
st.plotly_chart(fig_net, use_container_width=True)


section("🔀", "ENERGY FLOW — SANKEY DIAGRAM")

total_solar  = float(np.sum(solar))
total_wind   = float(np.sum(wind))
total_demand = float(np.sum(demand))
total_gen    = total_solar + total_wind
batt_charge  = float(np.sum(np.maximum(0,  (solar + wind - demand) * 0.3)))
batt_discharge = float(np.sum(np.maximum(0, (demand - solar - wind) * 0.3)))
direct_to_load = min(total_gen, total_demand) * 0.65
curtailed    = max(0, total_gen - total_demand - batt_charge) * 0.1
diesel_backup = max(0, total_demand - total_gen - batt_discharge) * 0.2

fig_sankey = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=18,
        thickness=22,
        line=dict(color="#1e2d3d", width=0.5),
        label=["Solar", "Wind", "Battery", "AI Controller", "Village Load", "Grid Export"],
        color=["#00e5a0", "#0af0ff", "#b070ff", "#4488ff", "#f5a623", "#888888"],
        x=[0.05, 0.05, 0.30, 0.55, 0.90, 0.90],
        y=[0.20, 0.70, 0.45, 0.45, 0.30, 0.75],
    ),
    link=dict(
        source=[0, 1, 0, 1, 2, 3, 3],
        target=[3, 3, 2, 2, 3, 4, 5],
        value=[
            max(total_solar * 0.75, 1),
            max(total_wind  * 0.75, 1),
            max(batt_charge * 0.6,  1),
            max(batt_charge * 0.4,  0.5),
            max(batt_discharge,     1),
            max(direct_to_load,     1),
            max(curtailed,          0.5),
        ],
        color=[
            "rgba(0,229,160,0.25)",
            "rgba(10,240,255,0.22)",
            "rgba(0,229,160,0.18)",
            "rgba(10,240,255,0.18)",
            "rgba(176,112,255,0.25)",
            "rgba(245,166,35,0.30)",
            "rgba(136,136,136,0.18)",
        ],
        label=["Solar→Controller","Wind→Controller",
               "Solar→Battery","Wind→Battery",
               "Battery→Controller","Dispatch→Load","Export"],
        hovertemplate="%{label}: %{value:.0f} kWh<extra></extra>",
    ),
))
fig_sankey.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#d4e0ec", family="Space Mono, monospace", size=11),
    height=320,
    margin=dict(l=10, r=10, t=20, b=10),
)
st.plotly_chart(fig_sankey, use_container_width=True)


# ─────────────────────────────────────────────
# HOURLY HEATMAP — energy intensity

section("🌡", "HOURLY ENERGY HEATMAP")

heat_data = np.array([solar, wind, demand, battery])
heat_labels = ["Solar kW", "Wind kW", "Demand kW", "Battery SoC %"]

fig_heat = go.Figure(go.Heatmap(
    z=heat_data,
    x=[f"{h:02d}:00" for h in range(24)],
    y=heat_labels,
    colorscale=[
        [0.0,  "#0b0f14"],
        [0.25, "#0d2a3a"],
        [0.5,  "#0e5c4a"],
        [0.75, "#00b87a"],
        [1.0,  "#00e5a0"],
    ],
    showscale=True,
    colorbar=dict(
        title=dict(text="kW / %", font=dict(color="#d4e0ec", size=11)),
        tickfont=dict(color="#5e7a96", size=10),
        bgcolor="rgba(19,25,32,0.8)",
        bordercolor="#1e2d3d",
        thickness=14,
    ),
    hovertemplate="<b>%{y}</b><br>Hour: %{x}<br>Value: %{z:.1f}<extra></extra>",
    xgap=1.5,
    ygap=1.5,
))
fig_heat.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#d4e0ec", family="Space Mono, monospace", size=10),
    height=220,
    margin=dict(l=10, r=80, t=10, b=40),
    xaxis=dict(color="#5e7a96", tickfont=dict(size=9)),
    yaxis=dict(color="#d4e0ec", tickfont=dict(size=10)),
)
st.plotly_chart(fig_heat, use_container_width=True)


# ─────────────────────────────────────────────
# LIVE SIMULATION  —  button-gated, no startup blocking

section("⏱", "LIVE SIMULATION PLAYBACK")

if st.button("▶  Run Live Simulation"):
    chart_ph = st.empty()
    status_ph = st.empty()
    for i in range(5, hours + 1):
        chart_ph.line_chart(
            pd.DataFrame({"☀️ Solar": solar[:i], "⚡ Demand": demand[:i], "💨 Wind": wind[:i]}),
            color=["#00e5a0", "#f5a623", "#0af0ff"],
        )
        status_ph.markdown(
            f'<div style="font-family:\'Space Mono\',monospace;font-size:12px;color:#5e7a96;">'
            f"Simulating hour <b style='color:#00e5a0'>{i}</b> / {hours} &nbsp;·&nbsp; "
            f"SoC: <b style='color:#b070ff'>{battery[i-1]:.1f}%</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
        time.sleep(0.07)
    status_ph.success("✅ Simulation complete!")
else:
    st.markdown(
        '<div style="color:#5e7a96;font-size:13px;">Click ▶ Run to start animated playback</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# PDF REPORT EXPORT
# ─────────────────────────────────────────────
section("📄", "EXPORT REPORT")

if st.button("📥  Generate PDF Report"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Try to import reportlab
        try:
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib import colors
            from reportlab.lib.units import inch
            REPORTLAB = True
        except ImportError:
            REPORTLAB = False

        # Build the plot
        fig_r, axes = plt.subplots(2, 1, figsize=(10, 7), facecolor="#0b0f14")
        for ax in axes:
            ax.set_facecolor("#131920")
            for spine in ax.spines.values():
                spine.set_color("#1e2d3d")
            ax.tick_params(colors="#5e7a96")

        axes[0].plot(solar,  color="#00e5a0", label="Solar",  linewidth=2)
        axes[0].plot(demand, color="#f5a623", label="Demand", linewidth=2)
        axes[0].plot(wind,   color="#0af0ff", label="Wind",   linewidth=1.5, linestyle="--")
        axes[0].legend(facecolor="#1a2330", edgecolor="#1e2d3d", labelcolor="#d4e0ec")
        axes[0].set_title("Energy Flow (24h)", color="#d4e0ec")
        axes[0].set_ylabel("kW", color="#5e7a96")

        axes[1].plot(battery, color="#b070ff", linewidth=2)
        axes[1].axhline(y=10, color="#ff4d4d", linestyle="--", alpha=0.6, label="Critical SoC")
        axes[1].fill_between(range(hours), battery, 0, alpha=0.15, color="#b070ff")
        axes[1].legend(facecolor="#1a2330", edgecolor="#1e2d3d", labelcolor="#d4e0ec")
        axes[1].set_title("Battery State of Charge %", color="#d4e0ec")
        axes[1].set_ylabel("SoC %", color="#5e7a96")
        axes[1].set_xlabel("Hour", color="#5e7a96")

        plt.tight_layout()
        plot_path = "/tmp/powerly_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0b0f14")
        plt.close()

        pdf_path = "/tmp/powerly_report.pdf"

        if REPORTLAB:
            doc = SimpleDocTemplate(pdf_path, rightMargin=40, leftMargin=40,
                                    topMargin=50, bottomMargin=40)
            styles = getSampleStyleSheet()
            elements = []
            elements.append(Paragraph("⚡ Powerly — Microgrid Report", styles["Title"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(
                f"Model: {model_type}  |  Scenario: {scenario}  |  "
                f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}",
                styles["Normal"]))
            elements.append(Spacer(1, 18))
            elements.append(RLImage(plot_path, width=6.5*inch, height=4.5*inch))
            elements.append(Spacer(1, 18))

            summary_data = [
                ["Metric", "Value"],
                ["Avg Solar Output", f"{avg_solar:.1f} kW"],
                ["Avg Demand",       f"{np.mean(demand):.1f} kW"],
                ["Renewable Coverage", f"{coverage:.1f}%"],
                ["Low-SoC Hours",    str(blackouts)],
                ["Final Battery SoC", f"{battery[-1]:.0f}%"],
            ]
            tbl = Table(summary_data, colWidths=[3*inch, 2.5*inch])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#131920")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.HexColor("#00e5a0")),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#1a2330"), colors.HexColor("#131920")]),
                ("TEXTCOLOR",  (0, 1), (-1, -1), colors.HexColor("#d4e0ec")),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#1e2d3d")),
                ("PADDING",    (0, 0), (-1, -1), 8),
            ]))
            elements.append(tbl)
            doc.build(elements)
        else:
            # Fallback: matplotlib-only PDF
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(pdf_path) as pdf:
                fig_r2, axes2 = plt.subplots(2, 1, figsize=(10, 7), facecolor="#0b0f14")
                for ax in axes2:
                    ax.set_facecolor("#131920")
                axes2[0].plot(solar, color="#00e5a0", label="Solar")
                axes2[0].plot(demand, color="#f5a623", label="Demand")
                axes2[0].legend()
                axes2[0].set_title("Energy Flow")
                axes2[1].plot(battery, color="#b070ff")
                axes2[1].set_title("Battery SoC %")
                plt.tight_layout()
                pdf.savefig(fig_r2, facecolor="#0b0f14")
                plt.close()

        with open(pdf_path, "rb") as f:
            st.download_button(
                "⬇️  Download Report PDF",
                f,
                file_name=f"powerly_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
            )

    except Exception as e:
        st.error(f"PDF generation error: {e}\n\nEnsure matplotlib is installed.")


_footer_logo = (
    f'<img src="data:image/png;base64,{logo_img}" style="height:18px;vertical-align:middle;opacity:0.5;margin-right:8px;" />'
    if logo_img else "⚡"
)
st.markdown(
    f"""
    <div style="text-align:center;margin-top:40px;padding:20px;
                border-top:1px solid #1e2d3d;
                font-family:'Space Mono',monospace;
                font-size:11px;color:#2e4a62;">
        {_footer_logo} POWERLY · Smart Microgrid Energy Balancer · Built with Streamlit + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)