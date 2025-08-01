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
import math
from datetime import datetime, timedelta

# Import pydub for audio processing
from pydub import AudioSegment

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
    db = SessionLocal(); user = db.query(User).filter(User.username == username).first(); db.close()
    if user and user.check_password(password): return user
    return None

def add_new_user(username, password, role):
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first(): return False, "User already exists."
        new_user = User(username=username, role=role); new_user.set_password(password)
        db.add(new_user); db.commit()
        return True, f"User '{username}' created."
    finally: db.close()

def get_all_users_from_db():
    db = SessionLocal(); users = db.query(User.username, User.role).all(); db.close()
    return users


# --- 4. CORE TRANSCRIPTION & AI FUNCTIONS ---

def _offset_timestamps(transcription_text, offset_seconds):
    """A helper function to add a time offset to all timestamps in a block of text."""
    
    def replacer(match):
        # Parse the timestamp matched by the regex
        h, m, s, ms = map(int, match.groups())
        
        # Create a timedelta object for the original timestamp
        original_time = timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)
        
        # Add the offset
        offset_delta = timedelta(seconds=offset_seconds)
        new_time = original_time + offset_delta
        
        # Format the new timestamp back into HH:MM:SS.mmm format
        # This is a bit complex to handle days and format correctly
        total_seconds = new_time.total_seconds()
        new_h = int(total_seconds // 3600)
        new_m = int((total_seconds % 3600) // 60)
        new_s = int(total_seconds % 60)
        new_ms = int(new_time.microseconds / 1000)
        
        return f"[{new_h:02d}:{new_m:02d}:{new_s:02d}.{new_ms:03d}]"

    # Regex to find all timestamps in the text
    pattern = re.compile(r'\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]')
    return pattern.sub(replacer, transcription_text)


@st.cache_data(show_spinner="Preparing audio...", persist=True)
def transcribe_audio_with_gemini(_file_content, file_type, model_name):
    """
    Transcribes audio by chunking it into smaller segments for better accuracy
    on large files.
    """
    CHUNK_LENGTH_MINUTES = 10
    CHUNK_LENGTH_MS = CHUNK_LENGTH_MINUTES * 60 * 1000

    st.info("Step 1: Normalizing and preparing audio...")
    audio = AudioSegment.from_file(io.BytesIO(_file_content), format=file_type)
    processed_audio = audio.set_frame_rate(16000).set_channels(1)
    processed_audio = processed_audio.apply_gain(-20.0 - processed_audio.dBFS)

    num_chunks = math.ceil(len(processed_audio) / CHUNK_LENGTH_MS)
    st.info(f"Audio is long. Splitting into {num_chunks} chunk(s) of ~{CHUNK_LENGTH_MINUTES} minutes each.")

    all_transcriptions = []
    model = genai.GenerativeModel(model_name=model_name)
    
    prompt = (
        "You are a highly accurate audio transcription service. Your task is to transcribe the provided Bengali audio file with maximum precision. "
        "Follow these rules strictly:\n"
        "1.  **Output Format:** The transcription must be in Bengali script (Unicode).\n"
        "2.  **Speaker Diarization:** Identify each speaker and label them as 'বক্তা ১:', 'বক্তা ২:', etc.\n"
        "3.  **Timestamp Precision:** This is the most critical rule. You MUST provide a precise start timestamp in `[HH:MM:SS.mmm]` format at the beginning of each speaker's turn. The timestamp must be relative to the start of the audio chunk I provide.\n"
        "4.  **Segmentation:** Create a new timestamped line after a significant pause (more than 2-3 seconds).\n"
        "5.  **No Extraneous Text:** Your output should only contain timestamps, speaker labels, and the transcribed text.\n"
    )

    for i in range(num_chunks):
        start_ms = i * CHUNK_LENGTH_MS
        end_ms = start_ms + CHUNK_LENGTH_MS
        audio_chunk = processed_audio[start_ms:end_ms]
        
        st.info(f"Processing chunk {i+1} of {num_chunks}...")
        
        tmp_file_path, gemini_file = None, None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                audio_chunk.export(tmp_file, format="wav")
                tmp_file_path = tmp_file.name

            gemini_file = genai.upload_file(path=tmp_file_path)
            
            response = model.generate_content([prompt, gemini_file], request_options={"timeout": 600})
            
            # Offset timestamps for chunks after the first one
            offset_seconds = start_ms / 1000
            chunk_transcription = _offset_timestamps(response.text, offset_seconds)
            all_transcriptions.append(chunk_transcription)

        finally:
            if tmp_file_path and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
            if gemini_file: genai.delete_file(gemini_file.name)
            
    st.success("All chunks processed successfully!")
    return "\n".join(all_transcriptions)


# --- (The rest of the app remains the same) ---
@st.cache_data(show_spinner="Analyzing text...")
def analyze_text_with_gemini(text_to_analyze, task):
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
    if task == "summarize":
        prompt = f"Please provide a concise summary of the following Bengali conversation:\n\n{text_to_analyze}"
    elif task == "topics":
        prompt = f"Please identify and list the key topics discussed in the following Bengali conversation:\n\n{text_to_analyze}"
    else: return "Unknown analysis task."
    response = model.generate_content(prompt); return response.text

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

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False; st.session_state.username = None; st.session_state.role = None
    st.session_state.transcript_data = None; st.session_state.speaker_map = {}; st.session_state.last_uploaded_file_id = None

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
    st.sidebar.success(f"Logged in as **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()
    
    st.sidebar.markdown("---"); st.sidebar.header("⚙️ Transcription Settings")
    selected_model = st.sidebar.selectbox("Choose a Gemini Model", ["models/gemini-1.5-pro-latest", "models/gemini-1.5-flash-latest"])

    if st.session_state.role == 'admin':
        with st.sidebar.expander("👑 Admin Panel"):
            st.subheader("Add New User")
            with st.form("add_user_form", clear_on_submit=True):
                new_user = st.text_input("New Username"); new_pass = st.text_input("New Password", type="password"); new_role = st.selectbox("Role", ["user", "admin"])
                if st.form_submit_button("Add User"):
                    if new_user and new_pass:
                        success, message = add_new_user(new_user, new_pass, new_role)
                        if success: st.success(message)
                        else: st.error(message)
            st.subheader("All Users"); st.dataframe(get_all_users_from_db(), use_container_width=True)

    st.title("Interactive Bengali Audio Transcription 🎧")
    st.markdown("For best results with long audio files (>15 minutes), please use the `gemini-1.5-pro-latest` model.")
    uploaded_file = st.file_uploader("1. Upload an audio file", type=["wav", "mp3", "m4a"])

    if uploaded_file:
        if uploaded_file.file_id != st.session_state.get('last_uploaded_file_id'):
            st.info("New audio file detected. Clearing previous results.")
            st.session_state.transcript_data = None; st.session_state.speaker_map = {}
            st.session_state.last_uploaded_file_id = uploaded_file.file_id
            st.cache_data.clear()
    
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("2. Transcribe Audio", type="primary"):
            audio_bytes = uploaded_file.getvalue()
            raw_format = uploaded_file.type.split('/')[1]
            format_map = {'x-m4a': 'm4a', 'mpeg': 'mp3'}
            normalized_format = format_map.get(raw_format, raw_format)
            try:
                raw_transcription = transcribe_audio_with_gemini(audio_bytes, normalized_format, selected_model)
                st.session_state.transcript_data = parse_timestamped_transcription(raw_transcription)
                if not st.session_state.transcript_data:
                    st.warning("Could not parse timestamps from the transcription. Displaying raw text.")
                    st.text_area("Raw Output", raw_transcription, height=300)
                else:
                    unique_speakers = sorted(list(set(d['speaker'] for d in st.session_state.transcript_data)))
                    st.session_state.speaker_map = {sp: sp for sp in unique_speakers}
            except Exception as e: 
                st.error(f"An error occurred during transcription: {e}")
                if "Decoding failed" in str(e) or "Encoding failed" in str(e):
                    st.info("💡 Hint: This error can sometimes happen with unusual audio file encodings. Please try converting the file to a standard MP3 or WAV format and re-uploading.")

    if st.session_state.get('transcript_data'):
        st.markdown("---"); st.header("3. Review and Edit Transcription")
        st.caption("Click on a timestamp to jump to the audio. Edit text directly in the boxes.")
        with st.expander("✏️ Rename Speakers"):
            for original_speaker in list(st.session_state.speaker_map.keys()):
                new_name = st.text_input(f"Rename {original_speaker}", value=st.session_state.speaker_map[original_speaker], key=f"rename_{original_speaker}")
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