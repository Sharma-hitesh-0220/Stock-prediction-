"""
styles.py
---------
Holds the custom CSS (fintech dark theme) for the PredictStock dashboard.
Kept separate from app.py so the visual theme can be tweaked without
touching any app logic.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #0b0e14 0%, #0e1117 100%);
    color: #e6e9ef;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
#MainMenu, footer {visibility: hidden;}

.ps-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 18px 26px;
    background: linear-gradient(90deg, #121722 0%, #161b26 100%);
    border: 1px solid #232a37; border-radius: 14px; margin-bottom: 22px;
}
.ps-header h1 { font-size: 26px; font-weight: 700; margin: 0; color: #f3f5f9; letter-spacing: 0.3px; }
.ps-header p { margin: 2px 0 0 0; font-size: 13px; color: #8b93a7; }
.ps-badge {
    background: #1b2a20; color: #4ade80; border: 1px solid #234a30;
    padding: 6px 14px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
}

div[data-testid="stMetric"] {
    background: #131722; border: 1px solid #232a37; padding: 16px 18px;
    border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
div[data-testid="stMetricLabel"] { color: #8b93a7 !important; font-size: 13px !important; }
div[data-testid="stMetricValue"] { color: #f3f5f9 !important; font-weight: 700 !important; }

section[data-testid="stSidebar"] { background: #0d1017; border-right: 1px solid #1f2530; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: #f3f5f9; font-size: 14px; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 18px;
}

.ps-section-title {
    font-size: 18px; font-weight: 700; color: #f3f5f9;
    margin: 26px 0 6px 0; border-left: 4px solid #3b82f6; padding-left: 10px;
}
.ps-sub { color: #8b93a7; font-size: 13px; margin-bottom: 14px; }

.ps-signal {
    display: inline-block; padding: 10px 20px; border-radius: 10px;
    font-weight: 700; font-size: 16px; letter-spacing: 0.4px;
}
.ps-buy { background: #16241a; color: #4ade80; border: 1px solid #245a34; }
.ps-sell { background: #2a1416; color: #f87171; border: 1px solid #5c2226; }
.ps-hold { background: #241f14; color: #facc15; border: 1px solid #5c4f22; }

.ps-footer {
    margin-top: 40px; padding: 16px 0; border-top: 1px solid #232a37;
    color: #6b7280; font-size: 12px; text-align: center;
}

.stButton > button {
    background: #3b82f6; color: white; border: none; border-radius: 8px;
    padding: 8px 18px; font-weight: 600;
}
.stButton > button:hover { background: #2563eb; }
</style>
"""


def inject_css() -> None:
    """Injects the PredictStock custom CSS into the current Streamlit page."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
