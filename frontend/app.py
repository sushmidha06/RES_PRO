import sys
import os
# Add parent directory to sys.path to allow importing from other folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
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
from aws.aws_service import career_aws
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# ==================================================================
# CONFIGURATION & API SETUP
# ==================================================================
# Get API Key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY. Please check your .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

# Initialize LangChain Gemini model (Using 2.5 Flash as requested)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Page Configuration
st.set_page_config(
    page_title="NexGen AI Career Agent",
    page_icon="None",
    layout="wide"
)

# ==================================================================
# VIDEO BACKGROUND — robust HTML injection approach
# ==================================================================
def set_video_bg():
    import base64
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_file = os.path.join(script_dir, "a.mp4")
    
    video_html = ""
    if os.path.exists(video_file):
        with open(video_file, "rb") as f:
            video_bytes = f.read()
        b64 = base64.b64encode(video_bytes).decode()
        video_html = f"""
        <video autoplay loop muted playsinline id="bg-video"
            style="position:fixed; top:0; left:0; min-width:100%; min-height:100%; 
                   width:auto; height:auto; z-index:-1; object-fit:cover;
                   filter:brightness(0.4) contrast(1.1);">
            <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        </video>
        """
    return video_html

video_html = set_video_bg()

# ==================================================================
# BLACK & WHITE DESIGN SYSTEM (CSS)
# ==================================================================
st.markdown(video_html + """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Outfit:wght@300;400;500;600;700;900&display=swap');

/* Hide the empty container created by the very first background markdown block */
.stApp > div:first-child [data-testid="stVerticalBlock"] > div:first-child {
    margin: 0 !important;
    padding: 0 !important;
    height: 0 !important;
}

#bg-video {
    position: fixed;
    right: 0;
    bottom: 0;
    min-width: 100%; 
    min-height: 100%;
}
.stApp, [data-testid="stAppViewContainer"] {
    background: transparent !important;
}
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background: transparent !important;
}
section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.85) !important;
}

/* ── TOKENS ─────────────────────────────────────────── */
:root {
    --bg-glass: rgba(0, 0, 0, 0.82);
    --bg-card: rgba(8, 8, 8, 0.88);
    --bg-hover: rgba(15, 15, 15, 0.98);
    --neon-orange: #FF8C00;
    --neon-blue: #00F2FF;
    --ink1: var(--neon-orange);
    --ink2: #ffffff;
    --ink3: var(--neon-blue);
    --ink4: #aaaaaa;
    --border: rgba(0, 242, 255, 0.15);
    --border-hi: rgba(0, 242, 255, 0.5);
    --border-active: var(--neon-blue);
    --accent: var(--neon-orange);
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
    color: var(--neon-orange) !important;
    margin-bottom: 0.5rem !important;
    text-transform: uppercase;
    letter-spacing: 8px;
    -webkit-text-fill-color: unset !important;
    text-shadow: 0 0 7px var(--neon-orange), 0 0 15px rgba(255, 140, 0, 0.5);
    width: 100%;
}

.sub-title {
    text-align: center;
    color: var(--neon-blue);
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    opacity: 1;
    text-shadow: 0 0 5px var(--neon-blue);
}

/* ── GLASS CARD ──────────────────────────────────────── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px) saturate(100%);
    -webkit-backdrop-filter: blur(20px) saturate(100%);
    border: 1px solid var(--border);
    border-radius: 0;
    padding: 2.2rem;
    margin-bottom: 6rem;
    transition: all 0.3s ease;
    position: relative;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 40px; height: 2px;
    background: var(--neon-orange);
    box-shadow: 0 0 10px var(--neon-orange);
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
    color: var(--neon-orange);
    text-transform: uppercase;
    letter-spacing: 3px;
    -webkit-text-fill-color: var(--neon-orange);
    background: none;
    -webkit-background-clip: unset;
    text-shadow: 0 0 5px var(--neon-orange);
}

/* ── BUTTONS ─────────────────────────────────────────── */
.stButton {
    margin-top: 10rem !important;
}

.stButton > button {
    background: #000000 !important;
    border: 1px solid var(--neon-orange) !important;
    border-radius: 0 !important;
    color: var(--neon-orange) !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 1rem 1.2rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    height: 3.5rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: inset 0 0 5px rgba(255, 140, 0, 0.2);
    overflow: hidden;
    position: relative;
}

.stButton > button:hover {
    background: var(--neon-orange) !important;
    color: #000000 !important;
    box-shadow: 0 0 20px var(--neon-orange) !important;
    transform: none !important;
}

/* Button scroll effect simulation */
.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 2px;
    background: white;
    transition: all 0.5s ease;
    opacity: 0.5;
}

.stButton > button:hover::before {
    left: 100%;
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
    color: var(--neon-orange) !important;
    border-bottom: 2px solid var(--neon-orange) !important;
}

/* ── FILE UPLOADER ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255, 140, 0, 0.02) !important;
    border: 1px dashed var(--neon-orange) !important;
    border-radius: 0 !important;
    margin-top: 2rem !important;
}

/* ── AGENT HEADER ────────────────────────────────────── */
.agent-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
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
    background: rgba(30, 30, 30, 0.9) !important;
    border: 1px solid rgba(255, 140, 0, 0.3) !important;
    border-radius: 0 !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
    padding: 1.2rem !important;
    transition: all 0.2s ease;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    background: rgba(50, 50, 50, 0.95) !important;
    border-color: var(--neon-orange) !important;
    box-shadow: 0 0 10px rgba(255, 140, 0, 0.2) !important;
}

.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: rgba(255, 140, 0, 0.5) !important;
    font-size: 0.85rem;
}

/* Specific Label Styling */
div[data-testid="stRadio"] label, 
[data-testid="stFileUploader"] section div {
    color: var(--neon-orange) !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 1px;
}

div[data-testid="stRadio"] p {
    color: var(--neon-orange) !important;
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
/* Glitch Animation */
@keyframes glitch-anim {
  0% { clip: rect(24px, 9999px, 50px, 0); transform: skew(0.5deg); }
  5% { clip: rect(85px, 9999px, 140px, 0); transform: skew(0.2deg); }
  10% { clip: rect(10px, 9999px, 30px, 0); transform: skew(0.8deg); }
  15% { clip: rect(60px, 9999px, 80px, 0); transform: skew(0.1deg); }
  20% { clip: rect(120px, 9999px, 150px, 0); transform: skew(0.3deg); }
  25% { clip: rect(40px, 9999px, 90px, 0); transform: skew(0.6deg); }
  100% { clip: rect(24px, 9999px, 50px, 0); transform: skew(0.5deg); }
}

.glitch-corner {
    position: fixed;
    width: 60px;
    height: 60px;
    z-index: 9999;
    pointer-events: none;
}

.corner-tl {
    top: 20px;
    left: 20px;
    border-top: 3px solid var(--neon-orange);
    border-left: 3px solid var(--neon-orange);
}

.corner-br {
    bottom: 20px;
    right: 20px;
    border-bottom: 3px solid var(--neon-blue);
    border-right: 3px solid var(--neon-blue);
}

.glitch-overlay {
    position: absolute;
    top: -2px;
    left: -2px;
    width: 100%;
    height: 100%;
    background: transparent;
    opacity: 0.5;
}

.corner-tl .glitch-overlay {
    border-top: 3px solid var(--neon-blue);
    border-left: 3px solid var(--neon-blue);
    animation: glitch-anim 2s infinite linear alternate-reverse;
}

.corner-br .glitch-overlay {
    border-bottom: 3px solid var(--neon-orange);
    border-right: 3px solid var(--neon-orange);
    animation: glitch-anim 3s infinite linear alternate-reverse;
}

/* Decor dots */
.glitch-corner::after {
    content: '01 10 11';
    position: absolute;
    font-family: 'Space Mono', monospace;
    font-size: 8px;
    color: white;
    opacity: 0.3;
}

.corner-tl::after { bottom: -15px; left: 0; }
.corner-br::after { top: -15px; right: 0; }

</style>

<div class="glitch-corner corner-tl"><div class="glitch-overlay"></div></div>
<div class="glitch-corner corner-br"><div class="glitch-overlay"></div></div>
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
st.markdown('<p class="sub-title">LangChain &amp; Gemini 2.5 Flash &nbsp;·&nbsp; Deep-dive career intelligence</p>', unsafe_allow_html=True)

# ── INTERFACE (Bento Layout) ─────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="agent-header"><h3>Resume Portal</h3></div>', unsafe_allow_html=True)

    input_mode = st.radio("Choose Input Type", ["File Upload", "Paste Text"], horizontal=True, label_visibility="collapsed")
    
    if "File" in input_mode:
        uploaded_file = st.file_uploader("Drop your PDF here", type=["pdf"], label_visibility="collapsed")
        pasted_text = ""
    else:
        pasted_text = st.text_area("Paste your resume content here", height=250, placeholder="Type or paste your resume content, skills, and experience here...", label_visibility="collapsed")
        uploaded_file = None

    col_bt1, col_bt2 = st.columns(2, gap="small")
    with col_bt1:
        match_btn = st.button("Role Match")
    with col_bt2:
        gap_btn = st.button("Skill Gap Analysis")

    st.markdown('</div>', unsafe_allow_html=True) # Close portal card

with col_right:
    # ── RESULTS AREA (Isolated Right Side) ──────────────────────────────
    # Determine input source
    text = None
    source_label = ""
    
    if uploaded_file:
        with st.spinner("Analyzing PDF Document..."):
            text = extract_text_from_pdf(uploaded_file)
            source_label = f"PDF: {uploaded_file.name}"
            if text:
                career_aws.ensure_bucket_exists()
                safe_name = re.sub(r'[^a-zA-Z0-9.]', '_', uploaded_file.name)
                career_aws.s3.put_object(Bucket=career_aws.bucket_name, Key=f"resumes/{safe_name}", Body=uploaded_file.getvalue())
    elif pasted_text and pasted_text.strip():
        text = pasted_text
        source_label = "Pasted Content"
    
    if (match_btn or gap_btn) and text:
        with st.spinner("Agent processing..."):
            st.caption(f"Analysis Source: {source_label}")
            if match_btn:
                result = run_role_matcher(text)
                header = "Job Match Analysis"
            else:
                result = run_skill_gap_agent(text)
                header = "Skill Gap Career Roadmap"

            st.markdown('<div class="glass-card result-card" style="margin-bottom:2rem;">', unsafe_allow_html=True)
            st.markdown(f'<div class="agent-header"><h3>{header}</h3></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-area">', unsafe_allow_html=True)
            
            if "Skill Gap" in header:
                tab1, tab2 = st.tabs(["Detailed Report", "Visual Architecture"])
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
                            st.info(f"[Open Roadmap Image](https://mermaid.ink/img/{safe_code})")
                        else:
                            st.warning("Could not derive architecture. Please try again.")
            else:
                # JOB MATCH RESULT
                st.markdown('<div class="agent-header" style="justify-content:space-between;"><h3>Best Match Found</h3><span style="border:1px solid rgba(0, 242, 255, 0.4); color:#00F2FF; padding:3px 10px; font-size:0.6rem; font-family:\'Space Mono\',monospace; letter-spacing:2px; text-shadow: 0 0 5px var(--neon-blue);">AI VERIFIED</span></div>', unsafe_allow_html=True)
                
                score_match = re.search(r'(\d+)%', result)
                role_match = re.search(r'PREDICTED ROLE:\s*(.*)', result, re.I)
                
                score = score_match.group(1) if score_match else "0"
                role = role_match.group(1).strip() if role_match else "N/A"
                
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:2rem; margin:1.5rem 0; padding:1.5rem; background:rgba(0, 242, 255, 0.03); border:1px solid var(--border);">
                    <div style="text-align:center; border-right:1px solid var(--border); padding-right:2rem;">
                        <div style="font-size:0.6rem; color:var(--neon-blue); text-transform:uppercase; letter-spacing:2px; font-family:'Space Mono',monospace;">Match Score</div>
                        <div style="font-size:2.5rem; font-weight:700; color:var(--neon-orange); font-family:'Space Mono',monospace; text-shadow: 0 0 10px var(--neon-orange);">{score}%</div>
                    </div>
                    <div>
                        <div style="font-size:0.6rem; color:var(--neon-blue); text-transform:uppercase; letter-spacing:2px; font-family:'Space Mono',monospace;">Recommended For</div>
                        <div style="font-size:1.1rem; font-weight:600; color:#fff; font-family:'Space Mono',monospace; margin-top:0.25rem; text-shadow: 0 0 5px #fff;">{role}</div>
                    </div>
                </div>
                <div style="margin-top:1.5rem;">
                    <div style="font-size:0.6rem; color:var(--neon-orange); font-family:'Space Mono',monospace; letter-spacing:2px; text-transform:uppercase; margin-bottom:0.75rem; border-left:2px solid var(--neon-orange); padding-left:0.5rem;">Candidate Highlights</div>
                    <div style="font-size:0.88rem; color:#ffffff; line-height:1.8;">{result}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown('</div></div>', unsafe_allow_html=True) # Close result-area and glass-card
    else:
        # Placeholder for Right Side
        st.markdown('''
            <div class="glass-card" style="height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; opacity: 0.6; min-height: 500px;">
                <div style="font-family: 'Space Mono', monospace; color: var(--neon-blue); font-size: 0.8rem; letter-spacing: 2px; text-transform: uppercase;">Awaiting Intelligence Data</div>
                <div style="font-size: 0.7rem; color: #666; margin-top: 0.5rem;">Upload or paste a resume to begin real-time analysis</div>
            </div>
        ''', unsafe_allow_html=True)
