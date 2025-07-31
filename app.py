# app.py

import streamlit as st
import google.generativeai as genai
import google.api_core.exceptions
import os
import tempfile
import time
from dotenv import load_dotenv
import re
import json
from docx import Document
import io

# --- Database & Authentication Imports ---
from database import SessionLocal
from models import User

# --- 1. INITIAL APP CONFIGURATION ---
st.set_page_config(
    page_title="Interactive Bengali Transcriber",
    page_icon="🎧",
    layout="wide"
)

# --- 2. LOAD SECRETS & CONFIGURE API ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("FATAL: GEMINI_API_KEY is not configured on the server!")
    st.stop()
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Google Gemini API: {e}")
    st.stop()

# --- 3. AUTHENTICATION & DATABASE HELPERS ---
def authenticate_user(username, password):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user and user.check_password(password): return user
        return None
    finally: db.close()

def add_new_user(username, password, role):
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first(): return False, "User already exists."
        new_user = User(username=username, role=role); new_user.set_password(password)
        db.add(new_user); db.commit()
        return True, f"User '{username}' created successfully."
    finally: db.close()

def get_all_users_from_db():
    db = SessionLocal()
    try: return db.query(User.username, User.role).all()
    finally: db.close()


# --- 4. CORE TRANSCRIPTION & AI FUNCTIONS ---
@st.cache_data(show_spinner="Transcribing audio... this may take a few moments.", persist=True)
def transcribe_audio_with_gemini(_file_content, model_name):
    tmp_file_path, gemini_file = None, None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(_file_content)
            tmp_file_path = tmp_file.name
        
        st.info("Uploading file to Google for processing...")
        gemini_file = genai.upload_file(path=tmp_file_path)
        model = genai.GenerativeModel(model_name=model_name)
        
        prompt = (
            "You are an expert audio transcriber. Transcribe the following Bengali audio file. "
            "The output transcription MUST be in the Bengali script (Unicode). "
            "This is a conversation. Identify and label each speaker using 'বক্তা ১:', 'বক্তা ২:', etc. "
            "CRITICAL: Precede each speaker's dialogue with a precise timestamp in the format [HH:MM:SS.mmm]. "
            "Example: [00:00:12.345] বক্তা ১: (text here)\n"
            "Ensure every spoken part has a timestamp. Do not use any markdown or HTML in your output."
        )
        response = model.generate_content([prompt, gemini_file], request_options={"timeout": 600})
        return re.sub(r'<[^>]+>', '', response.text)
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
        if gemini_file: genai.delete_file(gemini_file.name); st.info("Temporary files cleaned up.")

@st.cache_data(show_spinner="Analyzing text...")
def analyze_text_with_gemini(text_to_analyze, task):
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
    if task == "summarize":
        prompt = f"Please provide a concise summary of the following Bengali conversation:\n\n{text_to_analyze}"
    elif task == "topics":
        prompt = f"Please identify and list the key topics discussed in the following Bengali conversation:\n\n{text_to_analyze}"
    else: return "Unknown analysis task."
    response = model.generate_content(prompt); return response.text

# --- 5. HELPER FUNCTIONS FOR INTERACTIVITY & EXPORT ---
def parse_timestamped_transcription(raw_text):
    data = []; pattern = re.compile(r'\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s*(বক্তা\s*\d+:)\s*([\s\S]*?)(?=\[\d{2}:\d{2}:\d{2}\.\d{3}\]|\Z)')
    for i, match in enumerate(pattern.finditer(raw_text)):
        h, m, s, ms, speaker, text = match.groups()
        total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
        data.append({"id": f"segment_{i}", "time_sec": total_seconds, "timestamp": f"[{h}:{m}:{s}]", "speaker": speaker.strip(), "text": text.strip()})
    return data

def get_full_transcript_text(transcript_data):
    return "\n\n".join([f"{st.session_state.speaker_map.get(entry['speaker'], entry['speaker'])}\n{entry['text']}" for entry in transcript_data])

def create_docx_content(transcript_data):
    document = Document(); document.add_heading('Audio Transcription', level=1)
    for entry in transcript_data:
        p = document.add_paragraph()
        renamed_speaker = st.session_state.speaker_map.get(entry['speaker'], entry['speaker'])
        p.add_run(f"{renamed_speaker}\n").bold = True; p.add_run(entry['text'])
    bio = io.BytesIO(); document.save(bio); bio.seek(0); return bio.getvalue()

# --- MAIN APPLICATION UI & LOGIC ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False; st.session_state.username = None; st.session_state.role = None
    st.session_state.transcript_data = None; st.session_state.speaker_map = {}
    st.session_state.last_uploaded_file_id = None # [NEW] Initialize the tracker

# --- AUTHENTICATION GATE ---
if not st.session_state.authenticated:
    st.header("Bengali Transcriber Login")
    with st.form("login_form"):
        username = st.text_input("Username"); password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            user = authenticate_user(username, password)
            if user:
                st.session_state.authenticated = True; st.session_state.username = user.username; st.session_state.role = user.role
                st.rerun()
            else: st.error("Invalid username or password")
else:
    # --- AUTHENTICATED USER INTERFACE ---
    st.sidebar.success(f"Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
    
    st.sidebar.markdown("---"); st.sidebar.header("⚙️ Transcription Settings")
    selected_model = st.sidebar.selectbox("Choose a Gemini Model", ["models/gemini-1.5-pro-latest", "models/gemini-1.5-flash-latest"])

    if st.session_state.role == 'admin':
        with st.sidebar.expander("👑 Admin Panel"):
            st.subheader("Add New User")
            # ... (admin panel code is unchanged) ...

    st.title("Interactive Bengali Audio Transcription 🎧")
    uploaded_file = st.file_uploader("1. Upload an audio file", type=["wav", "mp3", "m4a"])

    # --- [NEW] AUTOMATIC STATE RESET LOGIC ---
    if uploaded_file is not None:
        # Check if the uploaded file is different from the last one we processed
        if uploaded_file.file_id != st.session_state.get('last_uploaded_file_id'):
            st.info("New audio file detected. Clearing previous results.")
            # Reset all relevant session state variables
            st.session_state.transcript_data = None
            st.session_state.speaker_map = {}
            # Update the tracker to the new file's ID
            st.session_state.last_uploaded_file_id = uploaded_file.file_id
            # IMPORTANT: Clear the function cache to force re-transcription
            st.cache_data.clear()
    
    if uploaded_file:
        st.audio(uploaded_file)

        if st.button("2. Transcribe Audio", type="primary"):
            # This button will now always process the currently uploaded file
            audio_bytes = uploaded_file.getvalue()
            try:
                raw_transcription = transcribe_audio_with_gemini(audio_bytes, selected_model)
                st.session_state.transcript_data = parse_timestamped_transcription(raw_transcription)
                if not st.session_state.transcript_data:
                    st.warning("Could not parse timestamps from the transcription. Displaying raw text.")
                    st.text_area("Raw Output", raw_transcription, height=300)
                else:
                    unique_speakers = sorted(list(set(d['speaker'] for d in st.session_state.transcript_data)))
                    st.session_state.speaker_map = {sp: sp for sp in unique_speakers}
                    st.success("Transcription complete!")
            except Exception as e: st.error(f"An error occurred during transcription: {e}")

    # The rest of the page only shows if transcript_data exists in the session state
    if st.session_state.get('transcript_data'):
        st.markdown("---"); st.header("3. Review and Edit Transcription")
        st.caption("Click on a timestamp to jump to the audio. Edit text directly in the boxes.")

        with st.expander("✏️ Rename Speakers"):
            # Using list() to avoid "dictionary changed size during iteration" error
            for original_speaker in list(st.session_state.speaker_map.keys()):
                new_name = st.text_input(
                    f"Rename {original_speaker}", 
                    value=st.session_state.speaker_map[original_speaker], 
                    key=f"rename_{original_speaker}"
                )
                st.session_state.speaker_map[original_speaker] = new_name

        for i, entry in enumerate(st.session_state.transcript_data):
            col1, col2 = st.columns([0.15, 0.85], gap="medium")
            with col1:
                js_code = f"document.querySelector('audio').currentTime = {entry['time_sec']};"
                st.button(entry['timestamp'], key=f"time_{i}", on_click=lambda js=js_code: st.components.v1.html(f"<script>{js}</script>"))
                renamed_speaker = st.session_state.speaker_map.get(entry['speaker'], entry['speaker'])
                st.write(f"**{renamed_speaker}**")
            with col2:
                edited_text = st.text_area("segment text", value=entry['text'], key=f"text_{i}", label_visibility="collapsed")
                st.session_state.transcript_data[i]['text'] = edited_text
        
        st.success("All edits are saved automatically in this session.")
        st.markdown("---"); st.header("4. Analyze and Export")
        
        full_text = get_full_transcript_text(st.session_state.transcript_data)
        with st.expander("🤖 AI-Powered Analysis"):
            if st.button("Generate Summary"):
                st.text_area("Conversation Summary", analyze_text_with_gemini(full_text, "summarize"), height=150)
            if st.button("Identify Key Topics"):
                st.text_area("Key Topics", analyze_text_with_gemini(full_text, "topics"), height=150)
        
        st.subheader("Export Options")
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            st.download_button(label="📥 Download as TXT", data=full_text.encode('utf-8'), file_name="transcription.txt", mime="text/plain")
        with col_export2:
            st.download_button(label="📥 Download as DOCX", data=create_docx_content(st.session_state.transcript_data), file_name="transcription.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")