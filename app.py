import streamlit as st
import pandas as pd
import os
import joblib
import re
import PyPDF2
import numpy as np
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from io import BytesIO
import streamlit.components.v1 as components
import json
from aws_service import career_aws
# import base64 (not needed for st.video)

# ==================================================================
# CONFIGURATION & API SETUP
# ==================================================================
# Get API Key from environment variable (Best practice for deployment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDntYbI6uGq81o_jW5o0MLCE6PYBmbo9Bo") 
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Initialize LangChain Gemini model (Modern LCEL approach)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# Page Configuration
st.set_page_config(
    page_title="NexGen AI Career Agent",
    page_icon="🚀",
    layout="wide"
)

# ==================================================================
# VIDEO BACKGROUND — robust HTML injection approach
# ==================================================================
def set_video_bg(video_file):
    import base64

    video_html = ""
    if os.path.exists(video_file):
        with open(video_file, "rb") as f:
            video_bytes = f.read()
        b64 = base64.b64encode(video_bytes).decode()
        video_html = f"""
        <video autoplay loop muted playsinline
            style="position:fixed;top:0;left:0;width:100vw;height:100vh;
                   object-fit:cover;z-index:-1;
                   filter:brightness(0.4) contrast(1.1);">
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        </video>
        """

    st.markdown(f"""
    {video_html}
    <style>
    .stApp, [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}
    [data-testid="stHeader"],
    [data-testid="stToolbar"] {{
        background: transparent !important;
    }}
    section[data-testid="stSidebar"] {{
        background: rgba(0,0,0,0.85) !important;
    }}
    </style>
    """, unsafe_allow_html=True)

set_video_bg("a.mp4")

# ==================================================================
# BLACK & WHITE DESIGN SYSTEM (CSS)
# ==================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600;700;900&display=swap');

/* ── TOKENS ─────────────────────────────────────────── */
:root {
    --bg-glass: rgba(0, 0, 0, 0.72);
    --bg-card: rgba(8, 8, 8, 0.85);
    --bg-hover: rgba(20, 20, 20, 0.95);
    --ink1: #ffffff;
    --ink2: #ffffff;
    --ink3: #eeeeee;
    --ink4: #dddddd;
    --border: rgba(255, 255, 255, 0.12);
    --border-hi: rgba(255, 255, 255, 0.5);
    --border-active: #ffffff;
    --accent: #ffffff;
    --r-lg: 0px;
    --r-xl: 0px;
}

*, *::before, *::after { box-sizing: border-box; }

html, .stApp {
    color: var(--ink1);
    font-family: 'Outfit', sans-serif;
    background: transparent !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #fff; }

/* Layout */
.block-container {
    padding: 3rem 2rem !important;
    max-width: 1280px !important;
}

/* ── HERO ────────────────────────────────────────────── */
.main-title {
    font-family: 'Space Mono', monospace !important;
    font-size: clamp(2rem, 5.5vw, 3.5rem) !important;
    font-weight: 800 !important;
    text-align: center;
    color: #ffffff !important;
    margin-bottom: 0.5rem !important;
    text-transform: uppercase;
    letter-spacing: 8px;
    -webkit-text-fill-color: unset !important;
    text-shadow: 0 0 7px #fff, 0 0 10px #fff, 0 0 21px #fff;
    width: 100%;
}

.sub-title {
    text-align: center;
    color: var(--ink1);
    font-size: 0.85rem;
    margin-bottom: 3rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    opacity: 0.9;
}

/* ── GLASS CARD ──────────────────────────────────────── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px) saturate(100%);
    -webkit-backdrop-filter: blur(20px) saturate(100%);
    border: 1px solid var(--border);
    border-radius: 0;
    padding: 2.2rem;
    margin-bottom: 1.8rem;
    transition: all 0.3s ease;
    position: relative;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 40px; height: 2px;
    background: #fff;
}

.glass-card:hover {
    border-color: var(--border-hi);
    background: var(--bg-hover);
}

.agent-header h3 {
    margin: 0;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1rem;
    color: #ffffff;
    text-transform: uppercase;
    letter-spacing: 3px;
    -webkit-text-fill-color: #ffffff;
    background: none;
    -webkit-background-clip: unset;
}

/* ── BUTTONS ─────────────────────────────────────────── */
.stButton > button {
    background: #000000 !important;
    border: 1px solid #ffffff !important;
    border-radius: 0 !important;
    color: white !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 0.8rem 1.2rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #ffffff !important;
    color: #000000 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── TABS ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background-color: #000;
    padding: 0;
    border-radius: 0;
    border-bottom: 1px solid rgba(255,255,255,0.15);
}

.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 0;
    color: var(--ink3);
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 2px solid transparent;
    padding: 0 1.2rem;
}

.stTabs [aria-selected="true"] {
    background-color: transparent !important;
    color: white !important;
    border-bottom: 2px solid #ffffff !important;
}

/* ── FILE UPLOADER ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(255,255,255,0.2) !important;
    border-radius: 0 !important;
}

/* ── AGENT HEADER ────────────────────────────────────── */
.agent-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

.agent-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.08);
}

/* ── RESULT AREA ─────────────────────────────────────── */
.result-area {
    line-height: 1.8;
    font-size: 0.95rem;
    color: var(--ink2);
    font-weight: 500;
}

.result-area h1, .result-area h2, .result-area h3, .result-area h4 {
    font-family: 'Space Mono', monospace;
    color: #ffffff;
    -webkit-text-fill-color: #ffffff;
    background: none;
    -webkit-background-clip: unset;
    margin-top: 1.5rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 0.9rem;
    border-left: 2px solid #fff;
    padding-left: 0.75rem;
}

/* ── HOW IT WORKS ────────────────────────────────────── */
.how-step {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1.4rem;
}

.how-step-num {
    background: #000;
    border: 1px solid #fff;
    width: 28px; height: 28px;
    border-radius: 0;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700;
    color: white;
    flex-shrink: 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}

.how-step-text {
    font-size: 0.9rem;
    color: var(--ink2);
    font-weight: 500;
    line-height: 1.6;
}

.how-step-text b { color: #fff; }

/* ── EMPTY STATE ─────────────────────────────────────── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 350px;
    text-align: center;
    opacity: 0.5;
}

.empty-state-icon { font-size: 3rem; margin-bottom: 1rem; filter: grayscale(1); }
.empty-state-text {
    font-family: 'Space Mono', monospace;
    color: var(--ink3);
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    line-height: 1.8;
}

/* ── HR / DIVIDER ────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: rgba(255,255,255,0.1) !important;
    margin: 2.5rem 0 !important;
}

/* ── FOOTER ──────────────────────────────────────────── */
.footer-text {
    text-align: center;
    color: var(--ink4);
    font-size: 0.72rem;
    margin-top: 4rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}

/* ── METRIC VALUES ───────────────────────────────────── */
.metric-val { color: #ffffff !important; }

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 0 !important;
    color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #fff !important;
    box-shadow: none !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #ffffff !important; }

/* Info box */
.stInfo {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 0 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ==================================================================
# AGENT LOGIC (MODERN LANGCHAIN) — UNCHANGED
# ==================================================================

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        return text.strip()
    except Exception as e:
        st.error(f"Text Extraction Error: {e}")
        return None

def generate_visual_roadmap(report_text):
    """
    Second stage: Turn the detailed report into an architecture diagram
    """
    prompt = ChatPromptTemplate.from_template("""
    Refine the following Career Analysis into a professional Mermaid.js flowchart (graph TD).
    
    REPORT:
    {report}
    
    INSTRUCTIONS:
    - Focus on the '6-MONTH ACCELERATED ROADMAP' and 'SKILL GAPS'.
    - Use clean Node IDs like A, B, C.
    - Quote all labels: A["Start Here"].
    - Flow: Current_State --> Gap_Analysis --> Month_1 --> Month_2 --> ... --> Success.
    - Return ONLY the mermaid code starting with ```mermaid.
    """)
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"report": report_text})
    except:
        return ""

def run_skill_gap_agent(resume_text):
    """
    Modern LangChain Skill Gap Agent - Stage 1: Detailed Report
    """
    prompt = ChatPromptTemplate.from_template("""
    You are the 'NexGen Career Optimization Agent'. 
    Analyze the resume and provide a deep-dive report.
    
    RESUME:
    {resume}

    Structure:
    1. CURRENT PROFILE SUMMARY
    2. THE TECHNICAL GAP
    3. THE SOFT SKILL GAP
    4. RECOMMENDED CERTIFICATIONS
    5. 6-MONTH ACCELERATED ROADMAP
    6. LEARNING RESOURCES & LINKS
    
    Format as clean, professional markdown.
    """)
    
    chain = prompt | llm | StrOutputParser()
    try:
        return chain.invoke({"resume": resume_text})
    except Exception as e:
        return f"Error: {e}"

def run_role_matcher(resume_text):
    """
    Modern Role Prediction Logic
    """
    prompt = ChatPromptTemplate.from_template("""
    Analyze the following resume and predict the most suitable job role. 
    Format your response exactly as follows:
    PREDICTED ROLE: [Role Name]
    MATCH SCORE: [X]%
    TECHNICAL SKILLS: [List]
    RATIONALE: [Brief explanation]

    RESUME: {resume}
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        return chain.invoke({"resume": resume_text})
    except Exception as e:
        return f"Error in Matching: {e}"

# ==================================================================
# MAIN INTERFACE
# ==================================================================

# ── HERO ─────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">NexGen Career Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">LangChain &amp; Gemini 2.0 Flash &nbsp;·&nbsp; Deep-dive career intelligence</p>', unsafe_allow_html=True)

# ── COLUMNS ──────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:

    # Resume Portal Card
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('''
        <div class="agent-header">
            <h3>📤 Resume Portal</h3>
        </div>
    ''', unsafe_allow_html=True)

    tab_upload, tab_paste = st.tabs(["📤 Upload PDF", "✍️ Paste Text"])
    
    with tab_upload:
        uploaded_file = st.file_uploader("Drop your PDF here", type=["pdf"], label_visibility="collapsed")
    
    with tab_paste:
        pasted_text = st.text_area("Paste your resume content here", height=180, placeholder="Type or paste your resume content, skills, and experience here...", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        match_btn = st.button("🔍 Match Job Role")
    with col_bt2:
        gap_btn = st.button("🧠 Skill Gap Agent")
    
    aws_btn = st.button("☁️ AWS Counselor (Bedrock)")

    st.markdown('</div>', unsafe_allow_html=True)

with right_col:

    if uploaded_file or (pasted_text and pasted_text.strip()):
        if match_btn or gap_btn or aws_btn:
            with st.spinner("Agent is thinking..."):
                if uploaded_file:
                    text = extract_text_from_pdf(uploaded_file)
                    # Requirement: S3 Storage
                    if text:
                        career_aws.ensure_bucket_exists()
                        safe_name = re.sub(r'[^a-zA-Z0-9.]', '_', uploaded_file.name)
                        career_aws.s3.put_object(Bucket=career_aws.bucket_name, Key=f"resumes/{safe_name}", Body=uploaded_file.getvalue())
                else:
                    text = pasted_text
                
                if text:
                    if match_btn:
                        result = run_role_matcher(text)
                        header = "🎯 Job Match Analysis"
                    elif gap_btn:
                        result = run_skill_gap_agent(text)
                        header = "🚀 Skill Gap Career Roadmap"
                    else:
                        result = career_aws.get_bedrock_analysis(text)
                        header = "☁️ AWS Bedrock Counselor"

                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="agent-header"><h3>{header}</h3></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="result-area">', unsafe_allow_html=True)
                    
                    if "Skill Gap" in header:
                        tab1, tab2 = st.tabs(["📝 Detailed Report", "🏗️ Visual Architecture"])
                        with tab1:
                            st.markdown(result)
                        with tab2:
                            with st.spinner("Generating architecture from report..."):
                                roadmap_code = generate_visual_roadmap(result)
                                match = re.search(r'```mermaid\s+(.*?)```', roadmap_code, re.DOTALL | re.IGNORECASE)
                                if match:
                                    mermaid_code = match.group(1).strip()
                                    if not (mermaid_code.startswith("graph ") or mermaid_code.startswith("flowchart ")):
                                        mermaid_code = "graph TD\n" + mermaid_code
                                    
                                    html_string = f"""
                                    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
                                    <div class="mermaid" style="background:#0a0a0a; padding:20px; border:1px solid #333; display:flex; justify-content:center;">
                                        {mermaid_code}
                                    </div>
                                    <script>
                                        mermaid.initialize({{
                                            startOnLoad: true,
                                            theme: 'base',
                                            themeVariables: {{
                                                'primaryColor': '#1a1a1a',
                                                'primaryTextColor': '#ffffff',
                                                'primaryBorderColor': '#ffffff',
                                                'lineColor': '#888888',
                                                'secondaryColor': '#111111',
                                                'tertiaryColor': '#0a0a0a',
                                                'nodeBkg': '#1a1a1a',
                                                'nodeTextColor': '#ffffff',
                                                'edgeLabelBackground': '#000000',
                                                'clusterBkg': '#0d0d0d'
                                            }}
                                        }});
                                    </script>
                                    """
                                    components.html(html_string, height=500, scrolling=True)
                                    safe_code = mermaid_code.encode('utf-8').hex()
                                    st.info(f"🔗 [Open Roadmap Image](https://mermaid.ink/img/{safe_code})")
                                else:
                                    st.warning("Could not derive architecture. Please try again.")
                    else:
                        # JOB MATCH RESULT
                        st.markdown('<div class="agent-header" style="justify-content:space-between;"><h3>🎯 Best Match Found</h3><span style="border:1px solid rgba(255,255,255,0.3); color:#ffffff; padding:3px 10px; font-size:0.6rem; font-family:\'Space Mono\',monospace; letter-spacing:2px;">AI VERIFIED</span></div>', unsafe_allow_html=True)
                        
                        score_match = re.search(r'(\d+)%', result)
                        role_match = re.search(r'PREDICTED ROLE:\s*(.*)', result, re.I)
                        
                        score = score_match.group(1) if score_match else "0"
                        role = role_match.group(1).strip() if role_match else "N/A"
                        
                        st.markdown(f"""
                        <div style="display:flex; align-items:center; gap:2rem; margin:1.5rem 0; padding:1.5rem; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);">
                            <div style="text-align:center; border-right:1px solid rgba(255,255,255,0.1); padding-right:2rem;">
                                <div style="font-size:0.6rem; color:#ffffff; text-transform:uppercase; letter-spacing:2px; font-family:'Space Mono',monospace;">Match Score</div>
                                <div style="font-size:2.5rem; font-weight:700; color:#ffffff; font-family:'Space Mono',monospace;">{score}%</div>
                            </div>
                            <div>
                                <div style="font-size:0.6rem; color:#ffffff; text-transform:uppercase; letter-spacing:2px; font-family:'Space Mono',monospace;">Recommended For</div>
                                <div style="font-size:1.1rem; font-weight:600; color:#fff; font-family:'Space Mono',monospace; margin-top:0.25rem;">{role}</div>
                            </div>
                        </div>
                        <div style="margin-top:1.5rem;">
                            <div style="font-size:0.6rem; color:#ffffff; font-family:'Space Mono',monospace; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.75rem; border-left:2px solid #fff; padding-left:0.5rem;">Candidate Highlights</div>
                            <div style="font-size:0.88rem; color:#ffffff; line-height:1.8;">{result}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown('</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="glass-card" style="border-style:dashed; border-color:rgba(255,255,255,0.12);">
            <div class="empty-state">
                <div class="empty-state-icon">🚀</div>
                <div class="empty-state-text">Upload or paste a resume<br>to initialise the NexGen Agent</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('''
<p class="footer-text">
    NexGen AI Agent System &nbsp;·&nbsp; LangChain Framework &nbsp;·&nbsp; Gemini 2.0 Flash
</p>
''', unsafe_allow_html=True)