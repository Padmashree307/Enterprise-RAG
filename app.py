import streamlit as st
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.rag_pipeline import query_rag_stream

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Page Config ---
st.set_page_config(
    page_title="UNIDO Intelligence",
    page_icon="✨",
    layout="wide",
)

# --- Design Tokens (Hardlocked to Dark Mode) ---
# Color Palette (Warm & Sophisticated Dark)
PRIMARY_BG = "#0A0A0A" 
CARD_BG = "#1A1A1A" 
TEXT_COLOR = "#E2E2E2" 
SUBTLE_TEXT = "#9AA0A6" 
user_bubble_bg = "#303134" 
assistant_bubble_bg = "transparent"
border_color = "#303134" 
accent_color = "#8AB4F8" # Blue-ish Gemini accent

# --- CSS Injection ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600&family=Inter:wght@400;500&display=swap');

    /* Reset and Typography */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: {PRIMARY_BG} !important;
        color: {TEXT_COLOR} !important;
        font-family: 'Lexend', sans-serif !important;
    }}

    h1, h2, h3, h4 {{
        font-family: 'Lexend', sans-serif !important;
        font-weight: 500 !important;
    }}

    /* Hide Default UI Clutter but keep Header for sidebar toggle */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stHeader"] {{background-color: rgba(0,0,0,0);}}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {CARD_BG} !important;
        border-right: 1px solid {border_color};
        width: 320px !important;
    }}
    
    /* Center the main container */
    [data-testid="stVerticalBlock"] > div:has(div.chat-area) {{
        max-width: 850px;
        margin: 0 auto;
    }}

    /* LANDING GREETING */
    .greeting-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 25vh;
        animation: fadeIn 1s ease-out;
    }}
    .greeting-text {{
        font-size: 3.2rem;
        font-weight: 500;
        background: linear-gradient(135deg, {TEXT_COLOR} 0%, {accent_color} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }}
    .greeting-sub {{
        color: {SUBTLE_TEXT};
        font-size: 1.1rem;
        margin-top: 15px;
        font-weight: 300;
    }}

    /* CHAT BUBBLES - Alignment Logic */
    .chat-row {{
        display: flex;
        width: 100%;
        margin-bottom: 30px;
        animation: fadeIn 0.4s ease-out;
    }}
    .user-row {{
        justify-content: flex-end;
    }}
    .assistant-row {{
        justify-content: flex-start;
    }}

    .bubble {{
        max-width: 80%;
        padding: 16px 24px;
        border-radius: 24px;
        font-size: 1.05rem;
        line-height: 1.6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    
    .user-bubble {{
        background-color: {user_bubble_bg};
        color: {TEXT_COLOR};
        border-bottom-right-radius: 4px;
    }}
    
    .assistant-bubble {{
        background-color: {assistant_bubble_bg};
        color: {TEXT_COLOR};
        padding: 0; /* Minimalist look, no bubble background for assistant */
    }}

    .avatar-icon {{
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 12px;
        font-size: 1.2rem;
    }}

    /* INPUT BOX */
    div[data-testid="stChatInput"] {{
        background-color: transparent !important;
        border-top: 1px solid {border_color};
        padding: 20px 0 !important;
        max-width: 850px;
        margin: 0 auto;
    }}

    div[data-testid="stChatInput"] textarea {{
        background-color: {CARD_BG} !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid {border_color} !important;
        border-radius: 32px !important;
        padding: 14px 28px !important;
        font-size: 1.1rem !important;
        caret-color: {accent_color} !important;
        transition: border 0.3s ease, box-shadow 0.3s ease;
    }}

    div[data-testid="stChatInput"] textarea:focus {{
        border-color: {accent_color} !important;
        box-shadow: 0 0 12px rgba(26, 115, 232, 0.15) !important;
    }}

    /* Ensure Sidebar Toggle is visible */
    button[data-testid="stSidebarCollapseButton"] {{
        visibility: visible !important;
    }}

    /* SIDEBAR BUTTONS - 3D Effect */
    .stButton > button {{
        width: 100%;
        border-radius: 12px !important;
        border: 1px solid {border_color} !important;
        background-color: {user_bubble_bg} !important;
        color: {TEXT_COLOR} !important;
        font-weight: 500 !important;
        padding: 10px !important;
        transition: all 0.1s ease;
        /* 3D Effect */
        box-shadow: 0 4px 0 #000, 0 5px 10px rgba(0,0,0,0.3);
        margin-bottom: 4px;
        border: none !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(2px);
        box-shadow: 0 2px 0 #000, 0 3px 6px rgba(0,0,0,0.3);
    }}

    .stButton > button:hover {{
        border-color: {accent_color} !important;
        opacity: 0.9;
    }}

    /* ANIMATIONS */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Sources Box */
    .source-tag {{
        display: inline-block;
        background: {border_color};
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-top: 8px;
    }}

</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0;'>ENTERPRISE RAG</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Recent Conversations")
    if not st.session_state.messages:
        st.caption("No history yet.")
    else:
        # Display first query as a pill
        user_queries = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
        for q in user_queries[:3]:
            st.markdown(f"<div style='background:{user_bubble_bg}; padding:10px; border-radius:12px; margin-bottom:8px; font-size:0.9rem;'>{q[:40]}...</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main Page ---

# Chat Area Wrap
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

if not st.session_state.messages:
    # Minimalist Landing Section
    st.markdown(f"""
    <div class="greeting-container">
        <div class="greeting-text">Hello User</div>
        <div class="greeting-sub">Domain Knowledge Access: Finance, HR, Manufacturing</div>
    </div>
    """, unsafe_allow_html=True)
else:
    # Render messages with custom divs for bubble control
    for i, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            st.markdown(f"""
            <div class="chat-row user-row">
                <div class="bubble user-bubble">{content}</div>
                <div class="avatar-icon" style="background:{accent_color}; color:white;">👤</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant content
            st.markdown(f"""
            <div class="chat-row assistant-row">
                <div class="avatar-icon" style="background:transparent; border:1px solid {border_color};">🤖</div>
                <div class="bubble assistant-bubble">
                    <div style="margin-bottom:10px;">{content}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Sub-expander for sources if they exist
            if msg.get("sources"):
                with st.expander("📚 Sources & References"):
                    for s in msg["sources"]:
                        st.markdown(f"- **{s['department'].upper()}** — *{s['source_file']}*")

st.markdown('</div>', unsafe_allow_html=True)

# --- Space for floating input ---
st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)

# --- Chat Input ---
if prompt := st.chat_input("Ask your query"):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# --- Async Generation Handling ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    # 1. Prepare UI row for streaming
    st.markdown(f"""
    <div class="chat-row assistant-row">
        <div class="avatar-icon" style="background:transparent; border:1px solid {border_color};">🤖</div>
        <div class="bubble assistant-bubble" id="streaming-bubble">
    """, unsafe_allow_html=True)
    
    # 2. Use a placeholder for the actual text
    resp_container = st.empty()
    
    # 3. Stream from pipeline
    full_response = ""
    try:
        # Get stream from RAG pipeline
        stream_gen, sources_data, depts = query_rag_stream(user_query, top_k=3)
        
        for chunk in stream_gen:
            full_response += chunk
            resp_container.markdown(f"{full_response}▌")
            # time.sleep(0.01) # Small delay for smoother perception
            
        resp_container.markdown(full_response)
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources_data
        })
        
        # Closing the assistant row markup is handled by the next rerun but we can clear placeholder if needed
        st.rerun()
        
    except Exception as e:
        st.error(f"Generation Error: {e}")

st.markdown("</div></div>", unsafe_allow_html=True)
