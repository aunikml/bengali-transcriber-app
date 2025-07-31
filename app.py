# app.py

import streamlit as st
import google.generativeai as genai
import google.api_core.exceptions
import os
import tempfile
import time
from dotenv import load_dotenv
import re  # Import regular expressions for advanced splitting
import json # Import json to safely pass text to JavaScript

# --- New Imports for the Robust Database System ---
from database import SessionLocal
from models import User

# --- 1. INITIAL APP CONFIGURATION (MUST BE AT THE TOP) ---
st.set_page_config(
    page_title="Bengali Transcriber",
    page_icon="🗣️",
    layout="wide"
)

# --- 2. LOAD SECRETS & CONFIGURE API ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("FATAL: GEMINI_API_KEY is not configured on the server!")
    st.info("Please ensure a .env file with your GEMINI_API_KEY is present.")
    st.stop()
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Google Gemini API: {e}")
    st.stop()


# --- Authentication and Helper Functions ---
def authenticate_user(username, password):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user and user.check_password(password):
            return user
        return None
    finally:
        db.close()

def add_new_user(username, password, role):
    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            return False, "User already exists."
        new_user = User(username=username, role=role)
        new_user.set_password(password)
        db.add(new_user)
        db.commit()
        return True, f"User '{username}' created successfully."
    finally:
        db.close()

def get_all_users_from_db():
    db = SessionLocal()
    try:
        return db.query(User.username, User.role).all()
    finally:
        db.close()

def format_transcription_for_copying(text):
    """Creates a clean, plain-text version for the copy button."""
    return text.replace('বক্তা ', '\n\nবক্তা ').strip()

# --- NEW FUNCTION TO GENERATE STYLED HTML ---
def generate_html_transcription(raw_text):
    """
    Takes the raw transcription text and wraps it in styled HTML
    for a much better reading experience.
    """
    # Regex to split the text by speaker labels (e.g., "বক্তা ১:")
    # This keeps the speaker labels in the resulting list.
    parts = re.split(r'(বক্তা\s*\d+:)', raw_text)
    
    html_output = ""
    # Start from index 1 because the first element of the split is usually empty
    for i in range(1, len(parts), 2):
        speaker_label = parts[i]
        speech_text = parts[i+1].strip()
        
        # Create an HTML block for each speaker turn
        html_output += f"""
        <div class="speaker-block">
            <div class="speaker-label">{speaker_label}</div>
            <div class="speaker-text">{speech_text}</div>
        </div>
        """
    return f"<div class='transcription-container'>{html_output}</div>"


@st.cache_data(show_spinner="Transcribing audio... this may take a moment.", persist=True)
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
            "Please transcribe the following Bengali audio file. "
            "The output transcription MUST be in the Bengali script (Unicode). "
            "This is a conversation, so please identify and label each speaker. "
            "Use Bengali labels for the speakers, for example: 'বক্তা ১:', 'বক্তা ২:'."
        )
        response = model.generate_content([prompt, gemini_file], request_options={"timeout": 600})
        return response.text
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        if gemini_file:
            genai.delete_file(gemini_file.name)
            st.info("Temporary files cleaned up.")

# --- Main Application UI and Logic ---

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None

if not st.session_state.authenticated:
    st.header("Bengali Transcriber Login")
    # ... (Login form remains the same)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        if submitted:
            user = authenticate_user(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.username = user.username
                st.session_state.role = user.role
                st.rerun()
            else:
                st.error("Invalid username or password")
else:
    # --- Inject our Custom CSS ---
    st.markdown("""
        <style>
        .transcription-container {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            background-color: #fcfcfc;
            max-height: 600px;
            overflow-y: auto;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .speaker-block {
            background-color: #ffffff;
            border: 1px solid #e8e8e8;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .speaker-label {
            font-weight: bold;
            color: #007bff; /* A nice blue for the speaker label */
            margin-bottom: 5px;
            font-size: 1.05em;
        }
        .speaker-text {
            line-height: 1.6;
            color: #333;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Bengali Audio Transcription 🗣️")
    st.sidebar.success(f"Logged in as **{st.session_state.username}**")
    # ... (The rest of the sidebar remains the same)
    st.sidebar.caption(f"Role: {st.session_state.role}")
    if st.sidebar.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Transcription Settings")
    selected_model = st.sidebar.selectbox("Choose a Gemini Model", ["models/gemini-1.5-flash-latest", "models/gemini-1.5-pro-latest"])

    if st.session_state.role == 'admin':
        with st.sidebar.expander("👑 Admin Panel", expanded=False):
            # ... (Admin panel remains the same)
            st.subheader("Add New User")
            with st.form("add_user_form", clear_on_submit=True):
                new_user = st.text_input("New Username")
                new_pass = st.text_input("New Password", type="password")
                new_role = st.selectbox("Role", ["user", "admin"])
                if st.form_submit_button("Add User"):
                    if new_user and new_pass:
                        success, message = add_new_user(new_user, new_pass, new_role)
                        if success: st.success(message)
                        else: st.error(message)
                    else:
                        st.warning("Username and password cannot be empty.")
            st.subheader("All Users")
            st.dataframe(get_all_users_from_db(), use_container_width=True)

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        if "last_uploaded_file_id" not in st.session_state:
            st.session_state.last_uploaded_file_id = None
        
        if uploaded_file.file_id != st.session_state.last_uploaded_file_id:
            st.cache_data.clear()
            st.session_state.last_uploaded_file_id = uploaded_file.file_id
            st.info("New file detected. Cache has been cleared.")

    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("Transcribe Audio", type="primary"):
            audio_bytes = uploaded_file.getvalue()
            try:
                raw_transcription = transcribe_audio_with_gemini(audio_bytes, selected_model)
                st.success("Transcription Complete!")
                
                # --- START: THE NEW AND IMPROVED DISPLAY BLOCK ---
                st.subheader("Transcription Result")
                
                # 1. Generate the beautiful HTML view
                html_view = generate_html_transcription(raw_transcription)
                st.markdown(html_view, unsafe_allow_html=True)
                
                # 2. Prepare a clean, plain-text version for the copy button
                plain_text_for_copy = format_transcription_for_copying(raw_transcription)
                
                # 3. Use json.dumps to safely embed the plain text into the JavaScript
                text_to_copy_js = json.dumps(plain_text_for_copy)

                # 4. Display the self-contained "Copy to Clipboard" button
                st.components.v1.html(
                    f"""
                    <script>
                    function copyToClipboard() {{
                        // The text is directly embedded here, avoiding screen scraping
                        const textToCopy = {text_to_copy_js};
                        navigator.clipboard.writeText(textToCopy).then(() => {{
                            alert("Plain text transcription copied to clipboard!");
                        }}).catch(err => {{
                            console.error("Failed to copy text: ", err);
                        }});
                    }}
                    </script>
                    <br>
                    <button onclick="copyToClipboard()" style="width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #007bff; background-color: #007bff; color: white; cursor: pointer; font-size: 16px;">
                        📋 Copy Plain Text to Clipboard
                    </button>
                    """,
                    height=65
                )
                # --- END: THE NEW AND IMPROVED DISPLAY BLOCK ---

            except Exception as e:
                st.error(f"An error occurred during transcription: {e}")