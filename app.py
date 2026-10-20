
# =========================================================
# 2. SELECTABLE DASHBOARD THEMES — COMPLETE CODE
# =========================================================

THEMES = {
    "Emerald Forest": {
        "bg": "#141A17",
        "surface": "#1D2721",
        "surface2": "#25332A",
        "border": "#3A5142",
        "accent": "#45A36B",
        "accent2": "#78D99A",
        "gold": "#E7C76A",
        "text": "#F3F6F1",
        "muted": "#B6C5B9",
        "hero1": "#1E4931",
        "hero2": "#286B43",
        "chart1": "#45A36B",
        "chart2": "#E7C76A",
    },

    "Cyber Blue": {
        "bg": "#111827",
        "surface": "#1B2638",
        "surface2": "#25364D",
        "border": "#3B5270",
        "accent": "#3B82F6",
        "accent2": "#60A5FA",
        "gold": "#22D3EE",
        "text": "#F3F7FF",
        "muted": "#B5C5DC",
        "hero1": "#172554",
        "hero2": "#1D4ED8",
        "chart1": "#3B82F6",
        "chart2": "#22D3EE",
    },

    "Royal Purple": {
        "bg": "#18131F",
        "surface": "#261D30",
        "surface2": "#342641",
        "border": "#59416C",
        "accent": "#8B5CF6",
        "accent2": "#A78BFA",
        "gold": "#F0ABFC",
        "text": "#FAF5FF",
        "muted": "#D0C1DF",
        "hero1": "#3B1F58",
        "hero2": "#6D28D9",
        "chart1": "#8B5CF6",
        "chart2": "#F0ABFC",
    },

    "Sunset Copper": {
        "bg": "#1C1513",
        "surface": "#2B201C",
        "surface2": "#3B2A23",
        "border": "#654538",
        "accent": "#C47745",
        "accent2": "#F0A66E",
        "gold": "#F4C27A",
        "text": "#FFF7F0",
        "muted": "#D9BFB0",
        "hero1": "#592E21",
        "hero2": "#A6532D",
        "chart1": "#C47745",
        "chart2": "#F4C27A",
    },

    "Ocean Teal": {
        "bg": "#101D20",
        "surface": "#1A2B2E",
        "surface2": "#243B3D",
        "border": "#365C5B",
        "accent": "#168C87",
        "accent2": "#6DD5C7",
        "gold": "#A7F3E8",
        "text": "#F0FBFA",
        "muted": "#B6D4D0",
        "hero1": "#134E4A",
        "hero2": "#0F766E",
        "chart1": "#168C87",
        "chart2": "#6DD5C7",
    },
}


# Initialize the default theme.
if "dashboard_theme" not in st.session_state:
    st.session_state["dashboard_theme"] = "Emerald Forest"


# Theme selector in the sidebar.
st.sidebar.markdown("## 🎨 Appearance")

selected_theme = st.sidebar.selectbox(
    "Choose your dashboard theme",
    options=list(THEMES.keys()),
    key="dashboard_theme",
)

T = THEMES[selected_theme]


# Theme preview.
st.sidebar.markdown(
    f"""
    <div style="
        background: {T['surface']};
        border: 1px solid {T['border']};
        border-radius: 12px;
        padding: 12px;
        margin: 5px 0 15px 0;
    ">
        <div style="
            font-size: 12px;
            color: {T['muted']};
            margin-bottom: 8px;
        ">CURRENT THEME</div>
        <div style="
            font-size: 17px;
            font-weight: 800;
            color: {T['text']};
            margin-bottom: 10px;
        ">{selected_theme}</div>
        <div style="display:flex;gap:6px;">
            <div style="
                height: 10px;
                flex: 1;
                border-radius: 5px;
                background: {T['accent']};
            "></div>
            <div style="
                height: 10px;
                flex: 1;
                border-radius: 5px;
                background: {T['accent2']};
            "></div>
            <div style="
                height: 10px;
                flex: 1;
                border-radius: 5px;
                background: {T['gold']};
            "></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# Apply selected colors to the whole dashboard.
st.markdown(
    f"""
    <style>
    :root {{
        --background: {T["bg"]};
        --surface: {T["surface"]};
        --surface-2: {T["surface2"]};
        --border: {T["border"]};
        --accent: {T["accent"]};
        --accent-2: {T["accent2"]};
        --gold: {T["gold"]};
        --text: {T["text"]};
        --muted: {T["muted"]};
    }}

    .stApp,
    [data-testid="stAppViewContainer"] {{
        background: var(--background) !important;
        color: var(--text);
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    .block-container {{
        max-width: 1500px;
        padding-top: 1.7rem;
        padding-bottom: 2.5rem;
    }}

    [data-testid="stSidebar"] {{
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }}

    [data-testid="stSidebar"] * {{
        color: var(--text);
    }}

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp p,
    .stApp label {{
        color: var(--text);
    }}

    .hero {{
        background: linear-gradient(
            120deg,
            {T["hero1"]},
            {T["hero2"]}
        );
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }}

    .hero h1 {{
        color: #FFFFFF !important;
        font-size: 34px;
        font-weight: 800;
        line-height: 1.2;
        margin: 8px 0 12px 0;
    }}

    .hero p {{
        color: {T["text"]} !important;
        font-size: 15px;
        line-height: 1.7;
    }}

    .eyebrow,
    .section-heading {{
        color: var(--gold) !important;
    }}

    .eyebrow {{
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
    }}

    .section-heading {{
        font-size: 21px;
        font-weight: 750;
        margin: 16px 0 12px 0;
    }}

    .small-muted,
    .footer {{
        color: var(--muted) !important;
    }}

    .small-muted {{
        font-size: 13px;
        line-height: 1.6;
    }}

    .panel {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 12px;
    }}

    div[data-testid="stMetric"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 18px 20px;
        min-height: 120px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.12);
    }}

    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {{
        color: var(--muted) !important;
    }}

    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {{
        color: var(--text) !important;
        font-weight: 800;
    }}

    div[data-testid="stMetricDelta"] {{
        color: var(--gold) !important;
    }}

    .recommendation {{
        border-radius: 16px;
        padding: 23px;
        margin: 8px 0 16px 0;
        background: var(--surface);
    }}

    .recommendation-on {{
        border: 1px solid var(--accent);
        border-left: 6px solid var(--accent-2);
        background: var(--surface-2);
    }}

    .recommendation-off {{
        border: 1px solid var(--gold);
        border-left: 6px solid var(--gold);
        background: var(--surface-2);
    }}

    .recommendation h2 {{
        color: var(--text) !important;
        font-size: 25px;
        margin: 0 0 8px 0;
    }}

    .recommendation p {{
        color: var(--muted) !important;
        font-size: 14px;
        line-height: 1.6;
    }}

    .status-pill {{
        display: inline-block;
        padding: 5px 11px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 12px;
        background: var(--surface);
        color: var(--gold);
    }}

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {{
        background: var(--accent);
        color: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 10px;
        min-height: 43px;
        font-weight: 700;
        transition: 0.2s ease;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {{
        background: var(--surface-2);
        color: var(--text);
        border-color: var(--gold);
    }}

    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput div[data-baseweb="input"],
    .stTextInput div[data-baseweb="input"] {{
        background: var(--surface-2);
        border-color: var(--border);
    }}

    .stSelectbox div[data-baseweb="select"] *,
    .stNumberInput input,
    .stTextInput input {{
        color: var(--text) !important;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        gap: 5px;
        padding: 6px;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: var(--muted);
        border-radius: 8px;
    }}

    .stTabs [aria-selected="true"] {{
        background: var(--accent) !important;
        color: #FFFFFF !important;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 10px;
    }}

    hr {{
        border-color: var(--border);
    }}

    .stAlert {{
        border-radius: 10px;
    }}

    .footer {{
        text-align: center;
        font-size: 12px;
        line-height: 1.8;
        padding: 25px 0 8px 0;
    }}

    @media (max-width: 768px) {{
        .block-container {{
            padding: 1rem;
        }}

        .hero {{
            padding: 22px;
        }}

        .hero h1 {{
            font-size: 26px;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
