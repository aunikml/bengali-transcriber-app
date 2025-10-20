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
import pandas as pd
import plotly.express as px
import logging
from pydub import AudioSegment
from database import SessionLocal, init_db
from models import User, Transcript, Tag, TagDefinition
from html import escape

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="AI Transcription Workbench", page_icon="📝", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
init_db()

# --- 2. API CONFIGURATION ---
load_dotenv(); GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY: st.error("FATAL: GEMINI_API_KEY is not configured!", icon="❌"); st.stop()
try: genai.configure(api_key=GEMINI_API_KEY)
except Exception as e: st.error(f"Failed to configure API: {e}", icon="❌"); st.stop()

# --- 3. DATABASE HELPER FUNCTIONS ---
def authenticate_user(username, password):
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username).first()
        if user and user.check_password(password): return user
        return None
def add_new_user(username, password, role):
    with SessionLocal() as db:
        if db.query(User).filter(User.username == username).first(): return False, "User already exists."
        new_user = User(username=username, role=role); new_user.set_password(password)
        db.add(new_user); db.commit()
        return True, f"User '{username}' created."
def get_all_users_from_db():
    with SessionLocal() as db: return db.query(User.username, User.role).all()
def save_transcript_to_db(user_id, title, filename, content):
    with SessionLocal() as db:
        try:
            new_transcript = Transcript(title=title, original_filename=filename, content=content, owner_id=user_id)
            db.add(new_transcript); db.commit(); db.refresh(new_transcript); return new_transcript.id
        except Exception as e: st.error(f"Database Error: {e}", icon="❌"); return None
def update_transcript_in_db(transcript_id, new_content):
    with SessionLocal() as db:
        transcript = db.query(Transcript).filter(Transcript.id == transcript_id).first()
        if transcript: transcript.content = new_content; db.commit(); return True
        return False
def get_user_transcripts(user_id):
    with SessionLocal() as db: return db.query(Transcript).filter(Transcript.owner_id == user_id).order_by(Transcript.created_at.desc()).all()
def load_transcript_from_db(transcript_id):
    with SessionLocal() as db:
        transcript = db.query(Transcript).filter(Transcript.id == transcript_id).first()
        if transcript and transcript.content:
            for entry in transcript.content:
                if 'text' in entry and 'sentences' not in entry:
                    sentences = re.split(r'(?<=[।!?])\s+', entry['text'].strip()); entry['sentences'] = [{"text": s.strip()} for s in sentences if s.strip()]; entry.pop('text', None)
        return transcript
def add_tag_to_db(transcript_id, segment_index, sentence_index, tag_text):
    with SessionLocal() as db:
        existing = db.query(Tag).filter_by(transcript_id=transcript_id, segment_index=segment_index, sentence_index=sentence_index, tag_text=tag_text).first()
        if not existing:
            new_tag = Tag(transcript_id=transcript_id, segment_index=segment_index, sentence_index=sentence_index, tag_text=tag_text)
            db.add(new_tag); db.commit()
def remove_tag_from_db(tag_id):
    with SessionLocal() as db:
        tag = db.query(Tag).filter(Tag.id == tag_id).first()
        if tag: db.delete(tag); db.commit()
def get_tags_for_transcript(transcript_id):
    if not transcript_id: return []
    with SessionLocal() as db: return db.query(Tag).filter(Tag.transcript_id == transcript_id).all()
def get_tag_definitions():
    with SessionLocal() as db: return db.query(TagDefinition).all()
def add_tag_definition(tag_name, color):
    with SessionLocal() as db:
        if db.query(TagDefinition).filter(TagDefinition.tag_name == tag_name).first(): return False, "Tag already exists."
        new_def = TagDefinition(tag_name=tag_name.strip(), color=color); db.add(new_def); db.commit()
        return True, f"Tag '{tag_name}' created."
def delete_tag_definition(tag_id):
    with SessionLocal() as db:
        tag_def = db.query(TagDefinition).filter(TagDefinition.id == tag_id).first()
        if tag_def: db.delete(tag_def); db.commit(); return True
        return False

# --- 4. AI & CORE PROCESSING FUNCTIONS ---

# CACHING TEMPORARILY DISABLED TO FIX PERSISTENT ERROR
# @st.cache_data(show_spinner="Applying AI speech enhancement...", persist=True)
def _enhance_speech(audio_bytes, original_filename):
    tmp_file_in, tmp_file_out = None, None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(original_filename)[1]) as tmp_in:
            tmp_in.write(audio_bytes); tmp_file_in = tmp_in.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            tmp_file_out = tmp_out.name
        st.info("Uploading for enhancement...")
        genai.files.enhance_speech(input_uri=tmp_file_in, output_uri=tmp_file_out)
        st.info("Enhancement complete. Downloading clean audio...")
        with open(tmp_file_out, 'rb') as f_out: enhanced_audio_bytes = f_out.read()
        return enhanced_audio_bytes
    finally:
        if tmp_file_in and os.path.exists(tmp_file_in): os.remove(tmp_file_in)
        if tmp_file_out and os.path.exists(tmp_file_out): os.remove(tmp_file_out)

# CACHING TEMPORARILY DISABLED TO FIX PERSISTENT ERROR
# @st.cache_data(show_spinner="Analyzing topics...", persist=True)
def _generate_topics_with_gemini(full_transcript_text):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    prompt = ("You are an expert conversation analyst. Analyze the following Bengali transcript and identify the main topics discussed. Create a set of 'smart chapters'.\n"
              "Rules:\n1. Identify 5-7 distinct, high-level topics.\n2. For each topic, provide a short `title` in English.\n3. For each topic, provide a one-sentence `summary` in English.\n4. For each topic, find the `timestamp` (e.g., `[HH:MM:SS]`) where the topic begins.\n5. Respond ONLY with a valid JSON array of objects.\n\n"
              f"Transcript:\n---\n{full_transcript_text}\n---")
    try:
        response = model.generate_content(prompt, request_options={"timeout": 600})
        topics = json.loads(re.sub(r'```json\s*|\s*```', '', response.text, flags=re.DOTALL).strip())
        if isinstance(topics, list) and all(isinstance(t, dict) for t in topics): return topics
        return []
    except Exception as e: st.error(f"Could not generate topics: {e}"); return []

# CACHING TEMPORARILY DISABLED TO FIX PERSISTENT ERROR
# @st.cache_data(show_spinner="Suggesting tags...", persist=True)
def suggest_tags_for_sentence(sentence_text, predefined_tags):
    if not sentence_text or not sentence_text.strip(): return []
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    tags_list_str = ", ".join([f'"{tag}"' for tag in predefined_tags])
    prompt = (f"Analyze this Bengali sentence. Suggest relevant tags ONLY from: [{tags_list_str}]. Prioritize 'Action Item', 'Decision', 'Question'. If none fit, return an empty JSON list. Respond ONLY with a valid JSON list of strings.\n\nSentence: \"{sentence_text}\"")
    try:
        response = model.generate_content(prompt, request_options={"timeout": 60})
        suggestions = json.loads(re.sub(r'```json\s*|\s*```', '', response.text, flags=re.DOTALL).strip())
        return [tag for tag in suggestions if tag in predefined_tags]
    except Exception: return []

def _offset_timestamps(transcription_text, offset_seconds):
    def replacer(match):
        h, m, s, ms_str = match.groups(); ms = int((ms_str or "0").ljust(3, '0')); original_time = timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=ms); offset_delta = timedelta(seconds=offset_seconds); new_time = original_time + offset_delta; total_seconds_val = new_time.total_seconds(); new_h = int(total_seconds_val // 3600); new_m = int((total_seconds_val % 3600) // 60); new_s = int(total_seconds_val % 60); new_ms = int(new_time.microseconds / 1000)
        return f"[{new_h:02d}:{new_m:02d}:{new_s:02d}.{new_ms:03d}]"
    pattern = re.compile(r'\[(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?\]')
    return pattern.sub(replacer, transcription_text)

# CACHING TEMPORARILY DISABLED TO FIX PERSISTENT ERROR
# @st.cache_data(show_spinner="Transcribing chunk...", persist=True)
def _transcribe_chunk(audio_chunk_data, model_name):
    tmp_file_path, gemini_file = None, None
    model = genai.GenerativeModel(model_name=model_name)
    prompt = ("our only task is to transcribe the provided Bengali audio file. You MUST follow these rules exactly:\n\n"
        "1.  **Timestamp is MANDATORY:** Every line of speech MUST begin with a precise start timestamp in `[HH:MM:SS.mmm]` format. There are no exceptions.\n"
        "2.  **Speaker Labels:** After the timestamp, add a speaker label like 'বক্তা ১:', 'বক্তা ২:', etc.\n"
        "3.  **Handling Uncertainty:** If you are completely unable to distinguish between speakers in the audio, you MUST label every segment as 'বক্তা ১:'. Do NOT omit the speaker label or the timestamp.\n"
        "4.  **Content:** After the speaker label, provide the transcribed Bengali text.\n"
        "5.  **No Extra Text:** Do NOT add any introductions, summaries, or any text that is not part of the required `[TIMESTAMP] SPEAKER: TEXT` format.\n\n"
    )
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file: tmp_file.write(audio_chunk_data); tmp_file_path = tmp_file.name
        gemini_file = genai.upload_file(path=tmp_file_path)
        response = model.generate_content([prompt, gemini_file], request_options={"timeout": 600})
        if not response.parts:
            reason = "Unknown";
            if response.prompt_feedback and response.prompt_feedback.block_reason: reason = response.prompt_feedback.block_reason.name
            return f"Error: Chunk transcription failed (Reason: {reason})."
        return response.parts[0].text
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
        if gemini_file: genai.delete_file(gemini_file.name)

def transcribe_audio_with_gemini(file_content, file_type, model_name):
    CHUNK_LENGTH_MINUTES = 10; CHUNK_LENGTH_MS = CHUNK_LENGTH_MINUTES * 60 * 1000; API_CALL_DELAY_SECONDS = 5
    st.info("Step 1: Normalizing audio...")
    audio = AudioSegment.from_file(io.BytesIO(file_content), format=file_type); processed_audio = audio.set_frame_rate(16000).set_channels(1).apply_gain(-20.0 - audio.dBFS); duration_ms = len(processed_audio)
    if duration_ms <= CHUNK_LENGTH_MS:
        st.info("Audio is short. Transcribing in a single request.")
        with io.BytesIO() as audio_io: processed_audio.export(audio_io, format="wav"); return _transcribe_chunk(audio_io.getvalue(), model_name)
    else:
        num_chunks = math.ceil(duration_ms / CHUNK_LENGTH_MS); st.info(f"Audio is long. Splitting into {num_chunks} chunks.")
        all_transcriptions = []; progress_bar = st.progress(0, text="Transcribing chunks...")
        for i in range(num_chunks):
            if i > 0: time.sleep(API_CALL_DELAY_SECONDS)
            start_ms, end_ms = i * CHUNK_LENGTH_MS, (i + 1) * CHUNK_LENGTH_MS
            audio_chunk = processed_audio[start_ms:end_ms]; st.info(f"Processing chunk {i+1}/{num_chunks}...")
            try:
                with io.BytesIO() as audio_io: audio_chunk.export(audio_io, format="wav"); chunk_data = audio_io.getvalue()
                raw_chunk_transcription = _transcribe_chunk(chunk_data, model_name)
                if not raw_chunk_transcription or "Error:" in raw_chunk_transcription: raise ValueError(f"Transcription failed for chunk {i+1}.")
                all_transcriptions.append(_offset_timestamps(raw_chunk_transcription, start_ms / 1000))
            except Exception as e:
                st.error(f"Error on chunk {i+1}: {e}"); time_obj = timedelta(seconds=start_ms / 1000); h, m, s = int(time_obj.total_seconds()//3600), int((time_obj.total_seconds()%3600)//60), int(time_obj.total_seconds()%60)
                all_transcriptions.append(f"[{h:02d}:{m:02d}:{s:02d}.000] বক্তা ?: [TRANSCRIPTION FAILED]")
            progress_bar.progress((i + 1) / num_chunks)
        st.success("All chunks processed!"); return "\n".join(all_transcriptions)

def parse_and_structure_transcription(raw_text):
    if not raw_text or not raw_text.strip():
        logger.warning("Parsing attempted on empty or whitespace-only raw_text."); return []
    logger.info("Starting robust transcription parsing...")
    data = []
    pattern = re.compile(r'^\s*\[(\d{1,2}):(\d{1,2}):(\d{1,2})\.(\d{1,3})\]\s*(বক্তা\s*\d+|\?+)\s*:\s*(.*)')
    lines = raw_text.strip().split('\n')
    for line in lines:
        line = line.strip();
        if not line: continue
        match = pattern.match(line)
        if match:
            h, m, s, ms_str, speaker, text = match.groups(); h, m, s, ms = int(h), int(m), int(s), int(ms_str.ljust(3, '0'))
            total_seconds = h * 3600 + m * 60 + s + ms / 1000.0
            sentences_raw = re.split(r'(?<=[।!?])\s+', text.strip()); sentences = [{"text": s.strip()} for s in sentences_raw if s.strip()]
            if not sentences and text: sentences = [{"text": text}]
            if not sentences: continue
            data.append({ "id": f"segment_{len(data)}", "time_sec": total_seconds, "timestamp": f"[{h:02d}:{m:02d}:{s:02d}]", "speaker": speaker.strip() + ":", "sentences": sentences })
        elif data:
            if data[-1]["sentences"]: data[-1]["sentences"][-1]["text"] += " " + line
            else: data[-1]["sentences"].append({"text": line})
        else:
            logger.warning(f"Ignoring pre-transcription line: '{line[:100]}...'")
    if not data:
        st.error("Could not parse timestamps from transcription.", icon="❌"); st.text_area("Raw AI Output for Debugging", raw_text, height=200)
    return data

def get_full_transcript_text(transcript_data, filtered_tags=None):
    segments = []; current_tags = st.session_state.get('tags', [])
    for seg_idx, entry in enumerate(transcript_data):
        speaker = st.session_state.speaker_map.get(entry['speaker'], entry['speaker']); sentences_to_include = []
        for sent_idx, sentence in enumerate(entry.get('sentences', [])):
            if filtered_tags:
                sentence_tags = {tag.tag_text for tag in current_tags if tag.segment_index == seg_idx and tag.sentence_index == sent_idx}
                if not sentence_tags.intersection(filtered_tags): continue
            sentences_to_include.append(sentence['text'])
        if sentences_to_include: segments.append(f"{speaker}\n" + " ".join(sentences_to_include))
    return "\n\n".join(segments)

def create_docx_content(transcript_data, tags, filtered_tags=None):
    doc = Document(); doc.add_heading(st.session_state.current_transcript_title, 1)
    for seg_idx, entry in enumerate(transcript_data):
        sentences_to_include = []
        for sent_idx, sentence in enumerate(entry.get('sentences', [])):
            sentence_tags = {tag.tag_text for tag in tags if tag.segment_index == seg_idx and tag.sentence_index == sent_idx}
            if filtered_tags and not sentence_tags.intersection(filtered_tags): continue
            sentences_to_include.append((sentence['text'], sentence_tags))
        if sentences_to_include:
            p = doc.add_paragraph(); p.add_run(f"{st.session_state.speaker_map.get(entry['speaker'], entry['speaker'])}\n").bold = True
            for text, tags_set in sentences_to_include: p.add_run(text + " ")
            all_tags = {tag for _, tags in sentences_to_include for tag in tags}
            if all_tags: p.add_run(f"\n[Tags: {', '.join(sorted(list(all_tags)))}]\n").italic = True
    bio = io.BytesIO(); doc.save(bio); bio.seek(0); return bio.getvalue()

# --- 5. UI COMPONENTS ---
def create_interactive_display(transcript_data, tags, tag_definitions, filtered_tags=None):
    st.components.v1.html(f"""<script>
        function selectSentence(segIdx, sentIdx) {{ const url = new URL(window.location); url.searchParams.set('ss_seg', segIdx); url.searchParams.set('ss_sent', sentIdx); window.parent.location.href = url.href; }}
        function seekAudio(time) {{ const audioPlayer = parent.document.querySelector("audio"); if (audioPlayer) {{ audioPlayer.currentTime = time; audioPlayer.play(); }} }}
    </script>""", height=0)
    tag_def_map = {td.tag_name: td.color for td in tag_definitions}; transcript_html = ""
    for i, entry in enumerate(transcript_data):
        sentences_html, has_visible_sentence = "", False
        for j, sentence in enumerate(entry.get('sentences', [])):
            current_tags = {tag.tag_text for tag in tags if tag.segment_index == i and tag.sentence_index == j}
            if filtered_tags and not current_tags.intersection(filtered_tags): continue
            has_visible_sentence = True
            tags_html = "".join([f'<span class="tag-badge" style="background-color: {escape(tag_def_map.get(tag, "#6c757d"))};">{escape(tag)}</span>' for tag in current_tags])
            is_selected = st.session_state.get('selected_segment_index') == i and st.session_state.get('selected_sentence_index') == j
            sentences_html += f"""<div class="sentence-container {'selected-sentence' if is_selected else ''}" onclick="selectSentence({i}, {j})" id="sentence-{i}-{j}"><span class="sentence-text">{escape(sentence['text'])}</span><div class="tags-container">{tags_html}</div></div>"""
        if has_visible_sentence:
            renamed_speaker = escape(st.session_state.speaker_map.get(entry['speaker'], entry['speaker']))
            transcript_html += f"""<div class="transcript-card" id="segment-{i}"><div class="card-header"><div class="speaker-label">{renamed_speaker}</div><div class="timestamp-btn" onclick="seekAudio({entry['time_sec']})"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16"><path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM6.79 5.093A.5.5 0 0 0 6 5.5v5a.5.5 0 0 0 .79.407l3.5-2.5a.5.5 0 0 0 0-.814l-3.5-2.5z"/></svg><span>{entry['timestamp']}</span></div></div><div class="card-body">{sentences_html}</div></div>"""
    st.markdown(f"""<style> .sentence-container {{ padding: 4px; border-radius: 4px; margin-bottom: 4px; cursor: pointer; transition: background-color 0.2s; }} .sentence-container:hover {{ background-color: #f0f2f6; }} .selected-sentence {{ background-color: #ddeaff; border: 1px solid #0d6efd; }} .tags-container {{ margin-top: 4px; }} .tag-badge {{ display: inline-block; background-color: #0d6efd; color: white; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 0.85em; }} .transcript-card {{...}} </style><div id="transcript-container">{transcript_html}</div><script>...</script>""", unsafe_allow_html=True)

def tagging_sidebar(transcript_id, tag_definitions):
    st.sidebar.markdown("---"); st.sidebar.header("🏷️ Tagging Workbench")
    if st.session_state.get('selected_segment_index') is None: st.sidebar.info("Click a sentence in the transcript to tag it."); return
    seg_idx, sent_idx = st.session_state.selected_segment_index, st.session_state.selected_sentence_index
    sentence_text = st.session_state.transcript_data[seg_idx]['sentences'][sent_idx]['text']
    st.sidebar.markdown("**Selected:**"); st.sidebar.markdown(f"> _{sentence_text}_")
    current_tags_obj = [tag for tag in st.session_state.tags if tag.segment_index == seg_idx and tag.sentence_index == sent_idx]
    current_tags_text = {tag.tag_text for tag in current_tags_obj}
    predefined_tags = [td.tag_name for td in tag_definitions]
    if "suggestions" not in st.session_state.transcript_data[seg_idx]['sentences'][sent_idx]:
        st.session_state.transcript_data[seg_idx]['sentences'][sent_idx]['suggestions'] = suggest_tags_for_sentence(sentence_text, predefined_tags)
    suggestions = st.session_state.transcript_data[seg_idx]['sentences'][sent_idx]['suggestions']
    if suggestions:
        st.sidebar.markdown("**AI Suggestions:**")
        for sug in suggestions:
            if sug not in current_tags_text and st.sidebar.button(f"Add '{sug}'", key=f"sug_{seg_idx}_{sent_idx}_{sug}"):
                add_tag_to_db(transcript_id, seg_idx, sent_idx, sug); st.rerun()
    options = [tag for tag in predefined_tags if tag not in current_tags_text]
    selected_tag = st.sidebar.selectbox("Add predefined tag:", [""] + options, key=f"sel_{seg_idx}_{sent_idx}")
    if selected_tag: add_tag_to_db(transcript_id, seg_idx, sent_idx, selected_tag); st.rerun()
    with st.sidebar.form(f"custom_tag_{seg_idx}_{sent_idx}", clear_on_submit=True):
        custom_tag = st.text_input("Or add a custom tag:")
        if st.form_submit_button("Add Custom Tag"):
            if custom_tag.strip(): add_tag_to_db(transcript_id, seg_idx, sent_idx, custom_tag.strip()); st.rerun()
    if current_tags_obj:
        st.sidebar.markdown("**Current Tags:**")
        for tag in current_tags_obj:
            col1, col2 = st.sidebar.columns([0.8, 0.2])
            col1.markdown(f"- {tag.tag_text}")
            if col2.button("✖️", key=f"del_{tag.id}", help=f"Remove '{tag.tag_text}'"):
                remove_tag_from_db(tag.id); st.rerun()

def create_analytics_dashboard(transcript_data, speaker_map):
    st.header("📊 Conversation Dashboard")
    if len(transcript_data) < 2: st.info("Analytics require at least two transcribed segments."); return
    try:
        df = pd.DataFrame(transcript_data); df['next_time_sec'] = df['time_sec'].shift(-1)
        if len(df) > 1: avg_dur = (df['next_time_sec'] - df['time_sec']).mean(); total_dur = df['time_sec'].iloc[-1] + avg_dur; df['next_time_sec'].fillna(total_dur, inplace=True)
        else: df['next_time_sec'].fillna(df['time_sec'] + 5, inplace=True)
        df['duration'] = df['next_time_sec'] - df['time_sec']; df['duration'] = df['duration'].apply(lambda x: max(x, 0)); df['renamed_speaker'] = df['speaker'].map(speaker_map)
        col1, col2 = st.columns(2)
        with col1:
            talk_time = df.groupby('renamed_speaker')['duration'].sum().reset_index()
            fig_pie = px.pie(talk_time, values='duration', names='renamed_speaker', title='<b>Speaker Contribution</b>', hole=0.3, color_discrete_sequence=px.colors.qualitative.Set2)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='Speaker: %{label}<br>Talk Time: %{value:.1f}s<br>Contribution: %{percent}')
            fig_pie.update_layout(showlegend=False, title_x=0.5, font=dict(family="Arial, sans-serif")); st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_timeline = px.bar(df, x='time_sec', y='duration', color='renamed_speaker', title='<b>Conversation Activity Timeline</b>', labels={'time_sec': 'Time (s)', 'duration': 'Duration (s)'}, hover_name='renamed_speaker', hover_data={'text': False})
            fig_timeline.update_layout(xaxis_title=None, yaxis_title="Speech Duration", title_x=0.5, font=dict(family="Arial, sans-serif")); st.plotly_chart(fig_timeline, use_container_width=True)
    except Exception as e: st.warning(f"Could not generate analytics dashboard. Error: {e}")

# --- MAIN APP ---
def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated=False; st.session_state.username=None; st.session_state.user_id=None; st.session_state.role=None; st.session_state.transcript_data=None; st.session_state.speaker_map={}; st.session_state.last_uploaded_file_id=None; st.session_state.current_transcript_id=None; st.session_state.selected_segment_index=None; st.session_state.selected_sentence_index=None; st.session_state.filter_tags=[]; st.session_state.current_transcript_title="Untitled"; st.session_state.tags=[]; st.session_state.tag_definitions=[]; st.session_state.selected_model = "gemini-1.5-flash-latest"; st.session_state.topics=None; st.session_state.translate_mode=False; st.session_state.translated_data=None

    if not st.session_state.authenticated:
        st.header("AI Transcription Workbench Login")
        with SessionLocal() as db: user_count = db.query(User).count()
        if user_count == 0:
            st.warning("No users found. Please create the first admin user.", icon="⚠️")
            with st.form("create_admin_form"):
                st.subheader("Create Admin User"); username = st.text_input("Admin Username"); password = st.text_input("Admin Password", type="password")
                if st.form_submit_button("Create Admin"):
                    if username and password:
                        success, msg = add_new_user(username, password, "admin")
                        if success: st.success(msg); st.rerun()
                        else: st.error(msg)
        with st.form("login_form"):
            username = st.text_input("Username"); password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                user = authenticate_user(username, password)
                if user: st.session_state.authenticated=True; st.session_state.username=user.username; st.session_state.user_id=user.id; st.session_state.role=user.role; st.rerun()
                else: st.error("Invalid credentials.")
    else:
        with st.sidebar:
            st.header(f"Welcome, {st.session_state.username}")
            if st.button("Logout"):
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()
            with st.expander("📄 My Transcripts", expanded=True):
                user_transcripts = get_user_transcripts(st.session_state.user_id)
                if not user_transcripts: st.info("No saved transcripts.")
                else:
                    for t in user_transcripts:
                        if st.button(t.title, key=f"load_{t.id}"):
                            transcript = load_transcript_from_db(t.id); st.session_state.transcript_data=transcript.content; st.session_state.current_transcript_id=t.id; st.session_state.current_transcript_title=t.title; st.session_state.original_filename=t.original_filename
                            unique_speakers = sorted(list(set(d['speaker'] for d in t.content if 'speaker' in d))); st.session_state.speaker_map={sp: sp for sp in unique_speakers}
                            st.session_state.selected_segment_index=None; st.session_state.selected_sentence_index=None; st.session_state.topics=None; st.session_state.translate_mode=False; st.session_state.translated_data=None
                            st.success(f"Loaded '{t.title}'"); st.rerun()
            if st.session_state.get('transcript_data'):
                tagging_sidebar(st.session_state.current_transcript_id, st.session_state.tag_definitions)
            st.markdown("---")
            with st.expander("⚙️ Settings & Admin"):
                st.subheader("Transcription Model")
                model_options = ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
                
                current_model = st.session_state.get('selected_model')
                default_index = model_options.index(current_model) if current_model in model_options else 0

                st.session_state.selected_model = st.selectbox(
                    "Choose Gemini Model", 
                    options=model_options, 
                    index=default_index, 
                    help="Flash is faster. Pro is more accurate."
                )
                
                if st.button("Clear App Cache", help="Click this if you experience persistent errors after fixing the code."):
                    st.cache_data.clear()
                    st.success("App cache cleared! Please try your action again.")
                
                if st.session_state.role == 'admin':
                    st.subheader("👑 Admin Panel")
                    with st.form("add_user_form", clear_on_submit=True):
                        st.write("**Add New User**"); new_user = st.text_input("New Username"); new_pass = st.text_input("New Password", type="password"); new_role = st.selectbox("Role", ["user", "admin"])
                        if st.form_submit_button("Add User"):
                            if new_user and new_pass:
                                success, msg = add_new_user(new_user, new_pass, new_role)
                                if success: st.success(msg)
                                else: st.error(msg)
                    st.write("**All Users**"); st.dataframe(get_all_users_from_db(), use_container_width=True)
                    st.write("**Manage Pre-defined Tags**")
                    tag_defs = get_tag_definitions()
                    for tag_def in tag_defs:
                        c1, c2, c3 = st.columns([0.5, 0.3, 0.2]); c1.markdown(f"- {tag_def.tag_name}"); c2.color_picker("Color", value=tag_def.color, key=f"color_{tag_def.id}", disabled=True)
                        if c3.button("Del", key=f"del_def_{tag_def.id}"): delete_tag_definition(tag_def.id); st.rerun()
                    with st.form("add_tag_def_form", clear_on_submit=True):
                        new_tag_name = st.text_input("New Tag Name"); new_tag_color = st.color_picker("Tag Color", "#0d6efd")
                        if st.form_submit_button("Add Tag Definition"):
                            if new_tag_name.strip(): add_tag_definition(new_tag_name, new_tag_color); st.rerun()

        st.title("AI Transcription Workbench 📝")
        uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])
        if uploaded_file:
            if uploaded_file.file_id != st.session_state.get('last_uploaded_file_id'):
                st.session_state.transcript_data=None; st.session_state.speaker_map={}; st.session_state.last_uploaded_file_id=uploaded_file.file_id; st.session_state.original_filename=uploaded_file.name; st.session_state.current_transcript_title=os.path.splitext(uploaded_file.name)[0]; st.session_state.current_transcript_id=None; st.session_state.selected_segment_index=None; st.session_state.selected_sentence_index=None; st.cache_data.clear(); st.session_state.topics=None; st.session_state.translate_mode=False; st.session_state.translated_data=None
            st.audio(uploaded_file.getvalue())
            enhance_audio = st.checkbox("✨ Enhance Speech Quality (for noisy audio)")
            if st.button("Transcribe Audio", type="primary"):
                with st.spinner("Processing audio..."):
                    audio_bytes = uploaded_file.getvalue()
                    if enhance_audio: audio_bytes = _enhance_speech(audio_bytes, uploaded_file.name)
                    raw_format = "wav" if enhance_audio else uploaded_file.type.split('/')[1]; format_map = {'x-m4a': 'm4a', 'mpeg': 'mp3'}; normalized_format = format_map.get(raw_format, raw_format)
                    raw_transcription = transcribe_audio_with_gemini(audio_bytes, normalized_format, st.session_state.selected_model)
                    if raw_transcription:
                        st.session_state.transcript_data = parse_and_structure_transcription(raw_transcription)
                        if st.session_state.transcript_data:
                            new_id = save_transcript_to_db(st.session_state.user_id, st.session_state.current_transcript_title, uploaded_file.name, st.session_state.transcript_data)
                            if new_id: st.session_state.current_transcript_id = new_id; st.success("Transcription complete!"); st.rerun()
        if st.session_state.get('transcript_data'):
            params = st.query_params
            if 'ss_seg' in params and 'ss_sent' in params:
                st.session_state.selected_segment_index = int(params.get('ss_seg')); st.session_state.selected_sentence_index = int(params.get('ss_sent')); st.query_params.clear()
            st.session_state.tags = get_tags_for_transcript(st.session_state.current_transcript_id)
            st.session_state.tag_definitions = get_tag_definitions()
            st.header(f"Reviewing: {st.session_state.current_transcript_title}")
            display_data = st.session_state.transcript_data
            
            with st.container(border=True):
                st.subheader("✨ Smart Chapters & Navigation")
                if st.button("Generate Conversation Topics"):
                    full_text = get_full_transcript_text(st.session_state.transcript_data) # Always analyze original
                    st.session_state.topics = _generate_topics_with_gemini(full_text)
                if st.session_state.get('topics'):
                    st.components.v1.html("""<script>function seekAudio(t){const e=parent.document.querySelector("audio");e&&(e.currentTime=t,e.play())}</script>""", height=0)
                    for i, topic in enumerate(st.session_state.topics):
                        cols = st.columns([0.6, 0.2, 0.2]);
                        with cols[0]: st.markdown(f"**{topic.get('title', 'Untitled')}**"); st.caption(topic.get('summary', ''))
                        with cols[1]:
                            ts_str = topic.get('timestamp', '[00:00:00]')
                            try:
                                h, m, s = map(int, re.findall(r'\d+', ts_str)); time_in_seconds = h*3600+m*60+s
                                if st.button(f"▶️ Go to {ts_str}", key=f"topic_{i}"): st.components.v1.html(f"<script>seekAudio({time_in_seconds});</script>", height=0)
                            except: st.text("Invalid TS")

            with st.container(border=True):
                st.subheader("📊 Conversation Dashboard")
                create_analytics_dashboard(st.session_state.transcript_data, st.session_state.speaker_map)
            
            with st.container(border=True):
                st.subheader("📝 Transcript Review & Tagging")
                st.session_state.translate_mode = st.toggle("Translate to English", value=st.session_state.translate_mode)
                if st.session_state.translate_mode:
                    if st.session_state.translated_data is None:
                        # You would need to implement this translation function
                        # st.session_state.translated_data = _translate_text_with_gemini(st.session_state.transcript_data)
                        st.warning("Translation function is not yet implemented.")
                        st.session_state.translated_data = st.session_state.transcript_data # Placeholder
                        st.rerun()
                    if st.session_state.translated_data:
                        display_data = st.session_state.translated_data
                all_tags_in_transcript = sorted(list(set(tag.tag_text for tag in st.session_state.tags)))
                st.session_state.filter_tags = st.multiselect("Filter by Tags:", all_tags_in_transcript)
                create_interactive_display(display_data, st.session_state.tags, st.session_state.tag_definitions, st.session_state.filter_tags)
            
            with st.container(border=True):
                st.subheader("💾 Export")
                full_text_to_export = get_full_transcript_text(display_data, st.session_state.filter_tags if st.session_state.filter_tags else None)
                st.download_button("Export TXT", full_text_to_export.encode('utf-8'), f"{st.session_state.current_transcript_title}.txt")
                st.download_button("Export DOCX", create_docx_content(display_data, st.session_state.tags, st.session_state.filter_tags), f"{st.session_state.current_transcript_title}.docx")

if __name__ == "__main__":
    main()