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

# --- Senthron Premium CSS Injection ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700;800&family=Share+Tech+Mono&display=swap');
    .stApp { background-color: #05080f; color: #e2e8f0; }
    h1 { font-family: 'Space Grotesk', sans-serif !important; color: #ffffff !important; letter-spacing: 8px !important; text-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5); }
    h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #5ce5b4 !important; letter-spacing: 2px; font-weight: 400 !important; }
    .stButton>button { background: linear-gradient(90deg, #3b82f6 0%, #5ce5b4 100%) !important; border: none !important; border-radius: 6px !important; padding: 0.5rem 1rem !important; transition: all 0.3s ease !important; width: 100%; }
    .stButton>button p { color: #05080f !important; font-family: 'Space Grotesk', sans-serif !important; font-weight: 800 !important; letter-spacing: 1px !important; font-size: 16px !important; }
    div[data-testid="stMarkdownContainer"] p { font-family: 'Share Tech Mono', monospace !important; color: #94a3b8; }
    code { color: #5ce5b4 !important; background-color: rgba(92, 229, 180, 0.1) !important; border: 1px solid rgba(92, 229, 180, 0.2) !important; border-radius: 4px; padding: 2px 6px; }
    hr { border-color: rgba(92, 229, 180, 0.15); }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'system_logs' not in st.session_state:
    st.session_state['system_logs'] = []
if 'last_log_time' not in st.session_state:
    st.session_state['last_log_time'] = time.time()

# --- Global AI Assets ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh_engine = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_pose = mp.solutions.pose
pose_engine = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# --- The WebRTC Video Transformer ---
class SenthronVisualEngine(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process Visuals
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pose_res = pose_engine.process(img_rgb)
        face_res = face_mesh_engine.process(img_rgb)
        
        face_detected = False
        state = "Target Lost"
        hud_color = (0, 0, 255) # Red for lost

        # Draw Pose
        if pose_res.pose_landmarks:
            mp_drawing.draw_landmarks(img, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(92,229,180), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(59,130,246), thickness=2, circle_radius=2))

        # Draw Face Mesh
        if face_res.multi_face_landmarks:
            face_detected = True
            state = "Analyzing..."
            hud_color = (180, 229, 92) # Teal
            for face_landmarks in face_res.multi_face_landmarks:
                mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    None, mp_drawing.DrawingSpec(color=(92,229,180), thickness=1, circle_radius=1))

        # Emotion Extraction (DeepFace)
        if face_detected:
            try:
                # Process at 50% scale for cloud speed
                small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
                res = DeepFace.analyze(small, actions=['emotion'], enforce_detection=False)
                state = res[0]['dominant_emotion'].capitalize()
            except:
                pass

        # Update HUD
        cv2.putText(img, f"STATE: {state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, hud_color, 2)
        
        # Handle Session-Level Logging (This triggers a refresh of the log panel)
        curr_time = time.time()
        if curr_time - st.session_state['last_log_time'] > 3.0:
            timestamp = datetime.now().strftime("%I:%M:%S %p")
            mesh_status = "Active" if face_detected else "Inactive"
            new_log = f"[{timestamp}] Status: `{state}` | Mesh: `{mesh_status}`"
            st.session_state['system_logs'].insert(0, new_log)
            if len(st.session_state['system_logs']) > 10: st.session_state['system_logs'].pop()
            st.session_state['last_log_time'] = curr_time

        return img

# --- UI Layout ---
st.title("S E N T H R O N")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Visual Network")
    # THE CLOUD BRIDGE: replaces your previous cv2 loop
    webrtc_streamer(
        key="senthron-web-engine",
        video_transformer_factory=SenthronVisualEngine,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.subheader("Command Center")
    st.info("Visual Network ready. Click 'START' on the left to initialize.")
    st.markdown("---")
    st.subheader("System Terminal Log")
    log_placeholder = st.empty()
    
    # Render logs from session state
    if st.session_state['system_logs']:
        log_placeholder.markdown("\n\n".join(st.session_state['system_logs']))
    else:
        log_placeholder.write("Network: `STANDBY`")

# Auto-refresh log UI
if st.session_state['system_logs']:
    time.sleep(0.5)
    st.rerun()
