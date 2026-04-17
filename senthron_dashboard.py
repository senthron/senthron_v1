import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
from deepface import DeepFace
import mediapipe as mp
import numpy as np
from datetime import datetime
import time
import queue

# --- System Configuration ---
st.set_page_config(page_title="Senthron™: Visual Data Acquisition PoC", layout="wide")

# --- Senthron Premium CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700;800&family=Share+Tech+Mono&display=swap');
    .stApp { background-color: #05080f; color: #e2e8f0; }
    h1 { font-family: 'Space Grotesk', sans-serif !important; color: #ffffff !important; letter-spacing: 8px !important; text-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5); }
    h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #5ce5b4 !important; letter-spacing: 2px; font-weight: 400 !important; }
    div[data-testid="stMarkdownContainer"] p { font-family: 'Share Tech Mono', monospace !important; color: #94a3b8; }
    code { color: #5ce5b4 !important; background-color: rgba(92, 229, 180, 0.1) !important; border: 1px solid rgba(92, 229, 180, 0.2) !important; border-radius: 4px; padding: 2px 6px; }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'system_logs' not in st.session_state:
    st.session_state['system_logs'] = [f"[{datetime.now().strftime('%I:%M:%S %p')}] SYSTEM INITIALIZED"]

# Create a thread-safe Queue for communication
result_queue = queue.Queue()

# Global AI Engines
mp_face_mesh = mp.solutions.face_mesh
face_mesh_engine = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

class SenthronVisualEngine(VideoTransformerBase):
    def __init__(self):
        self.last_log_time = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh_engine.process(img_rgb)
        
        current_emotion = "Target Lost"
        face_detected = False

        if res.multi_face_landmarks:
            face_detected = True
            for face_landmarks in res.multi_face_landmarks:
                mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    None, mp_drawing.DrawingSpec(color=(92,229,180), thickness=1))
            
            try:
                # Fast Analysis
                small = cv2.resize(img, (0,0), fx=0.4, fy=0.4)
                analysis = DeepFace.analyze(small, actions=['emotion'], enforce_detection=False)
                current_emotion = analysis[0]['dominant_emotion'].capitalize()
            except:
                current_emotion = "Analyzing..."

        # Logic: If 3 seconds passed, send the current state to the Queue
        now = time.time()
        if now - self.last_log_time > 3.0:
            result_queue.put({
                "emotion": current_emotion,
                "status": "Active" if face_detected else "Inactive",
                "time": datetime.now().strftime("%I:%M:%S %p")
            })
            self.last_log_time = now

        cv2.putText(img, f"STATE: {current_emotion}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 229, 92), 2)
        return img

# --- UI Layout ---
st.title("S E N T H R O N")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Visual Network")
    webrtc_streamer(
        key="senthron-v1",
        video_transformer_factory=SenthronVisualEngine,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Command Center")
    st.info("Visual Data successfully bridging to Cloud.")
    st.markdown("---")
    st.subheader("System Terminal Log")
    log_placeholder = st.empty()

    # DASHBOARD LOOP: Pull data from the Queue and update the UI
    while True:
        try:
            # Check if there is a new data packet from the video engine
            data = result_queue.get_nowait()
            new_entry = f"[{data['time']}] Status: `{data['emotion']}` | Mesh: `{data['status']}`"
            
            # Update session state logs
            st.session_state['system_logs'].insert(0, new_entry)
            if len(st.session_state['system_logs']) > 15:
                st.session_state['system_logs'].pop()
            
            # Refresh the display
            log_placeholder.markdown("\n\n".join(st.session_state['system_logs']))
        except queue.Empty:
            # If no new data, just keep the current logs displayed
            log_placeholder.markdown("\n\n".join(st.session_state['system_logs']))
            break

    # Add a small manual refresh button for the logs if needed
    if st.button("Refresh Terminal Log"):
        st.rerun()
