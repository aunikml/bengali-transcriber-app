# pages/2_My_Transcripts.py

import streamlit as st
import time
from database import SessionLocal
from models import Transcript

# --- Page Configuration ---
st.set_page_config(
    page_title="My Transcripts",
    page_icon="📚",
    layout="wide"
)

# --- Database Helper Functions (specific to this page) ---
def get_user_transcripts(user_id):
    db = SessionLocal()
    try:
        return db.query(Transcript).filter(Transcript.owner_id == user_id).order_by(Transcript.created_at.desc()).all()
    finally:
        db.close()

def delete_transcript_from_db(transcript_id):
    db = SessionLocal()
    try:
        transcript = db.query(Transcript).filter(Transcript.id == transcript_id).first()
        if transcript:
            db.delete(transcript)
            db.commit()
            return True
        return False
    finally:
        db.close()

# --- Main Page Logic ---

# [MODIFIED] Check for authentication at the very top.
if not st.session_state.get("authenticated"):
    st.warning("🔒 Please log in to view and manage your transcripts.")
    # Stop execution of the page if the user is not logged in.
    st.stop()

# If authenticated, proceed to render the page.
st.title("📚 My Saved Transcripts")

# Initialize state for delete confirmation
if 'delete_confirmation_id' not in st.session_state:
    st.session_state.delete_confirmation_id = None

user_id = st.session_state.user_id
transcripts = get_user_transcripts(user_id)

if not transcripts:
    st.info("You have not saved any transcripts yet. Transcribe an audio file on the main 'app' page and save it to see it here.")
else:
    # --- [NEW] Table-like Layout ---
    st.markdown(f"You have **{len(transcripts)}** saved transcript(s).")
    
    # Create the table header
    header_cols = st.columns([4, 2, 1, 1])
    header_cols[0].write("**Title**")
    header_cols[1].write("**Saved On**")
    header_cols[2].write("**Edit**")
    header_cols[3].write("**Delete**")
    
    st.markdown("---")

    # Loop through and display each transcript as a row
    for t in transcripts:
        row_cols = st.columns([4, 2, 1, 1])
        
        # Column 1: Title and Filename
        with row_cols[0]:
            st.subheader(t.title)
            st.caption(f"Original file: `{t.original_filename}`")

        # Column 2: Date Created
        with row_cols[1]:
            st.write("") # Spacer for vertical alignment
            st.write(t.created_at.strftime('%Y-%m-%d %H:%M'))

        # Column 3: Load/Edit Action
        with row_cols[2]:
            st.write("") # Spacer
            if st.button("Load", key=f"load_{t.id}", use_container_width=True, help="Load this transcript in the main editor"):
                st.session_state.transcript_data = t.content
                st.session_state.current_transcript_title = t.title
                st.session_state.original_filename = t.original_filename
                unique_speakers = sorted(list(set(d.get('speaker', 'Unknown') for d in t.content)))
                st.session_state.speaker_map = {sp: sp for sp in unique_speakers}
                st.success(f"Loaded '{t.title}'. Navigate to the 'app' page to view.")
                # Small delay to allow the user to read the message
                time.sleep(1)
                st.rerun()

        # Column 4: Delete Action with Confirmation
        with row_cols[3]:
            st.write("") # Spacer
            if st.button("🗑️", key=f"delete_{t.id}", use_container_width=True, help="Delete this transcript"):
                st.session_state.delete_confirmation_id = t.id
                st.rerun()

        # --- Confirmation Dialog Logic ---
        # This appears below the row if the delete button was clicked
        if st.session_state.delete_confirmation_id == t.id:
            st.warning(f"**Are you sure you want to permanently delete '{t.title}'?**")
            confirm_cols = st.columns(2)
            with confirm_cols[0]:
                if st.button("✅ Yes, Delete", key=f"confirm_delete_{t.id}", use_container_width=True, type="primary"):
                    if delete_transcript_from_db(t.id):
                        st.success(f"Successfully deleted '{t.title}'.")
                        st.session_state.delete_confirmation_id = None
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to delete the transcript.")
            with confirm_cols[1]:
                if st.button("❌ Cancel", key=f"cancel_delete_{t.id}", use_container_width=True):
                    st.session_state.delete_confirmation_id = None
                    st.rerun()