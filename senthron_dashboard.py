import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
from deepface import DeepFace
import mediapipe as mp
import numpy as np
from datetime import datetime
import time

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
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'system_logs' not in st.session_state:
    st.session_state['system_logs'] = [f"[{datetime.now().strftime('%I:%M:%S %p')}] SYSTEM STANDBY"]

# Global AI Engines
mp_face_mesh = mp.solutions.face_mesh
face_mesh_engine = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

class SenthronVisualEngine(VideoTransformerBase):
    def __init__(self):
        self.last_log = time.time()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh_engine.process(img_rgb)
        
        state = "Target Lost"
        color = (0, 0, 255)

        if res.multi_face_landmarks:
            state = "Analyzing..."
            color = (180, 229, 92)
            for face_landmarks in res.multi_face_landmarks:
                mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    None, mp_drawing.DrawingSpec(color=(92,229,180), thickness=1, circle_radius=1))
            
            # Fast Emotion Extraction
            try:
                small = cv2.resize(img, (0,0), fx=0.4, fy=0.4)
                analysis = DeepFace.analyze(small, actions=['emotion'], enforce_detection=False)
                state = analysis[0]['dominant_emotion'].capitalize()
            except:
                pass

        cv2.putText(img, f"STATE: {state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

# --- UI Layout ---
st.title("S E N T H R O N")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Visual Network")
    # HARDENED RTC CONFIG: Using multiple STUN servers to punch through firewalls
    webrtc_streamer(
        key="senthron-web-engine",
        video_transformer_factory=SenthronVisualEngine,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Command Center")
    st.info("Visual Network ready. 1. Grant browser camera permission. 2. Click 'START' on the left.")
    st.markdown("---")
    st.subheader("System Terminal Log")
    log_placeholder = st.empty()
    log_placeholder.markdown("\n\n".join(st.session_state['system_logs']))
