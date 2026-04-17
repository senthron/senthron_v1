import cv2
import streamlit as st
from deepface import DeepFace
import mediapipe as mp
from datetime import datetime
import time

# --- System Configuration ---
st.set_page_config(page_title="Senthron™: Visual Data Acquisition PoC", layout="wide")

# --- Senthron Premium CSS Injection ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700;800&family=Share+Tech+Mono&display=swap');

    .stApp {
        background-color: #05080f;
        color: #e2e8f0;
    }
    
    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #ffffff !important;
        letter-spacing: 8px !important;
        text-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    
    h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #5ce5b4 !important; 
        letter-spacing: 2px;
        font-weight: 400 !important;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #5ce5b4 100%) !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton>button p {
        color: #05080f !important; 
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important; 
        letter-spacing: 1px !important;
        font-size: 16px !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 4px 15px rgba(92, 229, 180, 0.4) !important;
    }

    div[data-testid="stMarkdownContainer"] p {
        font-family: 'Share Tech Mono', monospace !important;
        color: #94a3b8;
    }
    
    code {
        color: #5ce5b4 !important;
        background-color: rgba(92, 229, 180, 0.1) !important;
        border: 1px solid rgba(92, 229, 180, 0.2) !important;
        border-radius: 4px;
        padding: 2px 6px;
    }
    
    hr {
        border-color: rgba(92, 229, 180, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session states
if 'system_logs' not in st.session_state:
    st.session_state['system_logs'] = []

# --- Core AI Logic ---
def run_senthron():
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    mp_pose = mp.solutions.pose
    
    cap = cv2.VideoCapture(0)
    last_log_update = time.time()

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, \
         mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
         
         while cap.isOpened():
             ret, frame = cap.read()
             if not ret:
                 break
                 
             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             pose_results = pose.process(frame_rgb)
             face_results = face_mesh.process(frame_rgb)
             
             # --- The Target Lock Logic (Fixes the "Sad" glitch) ---
             face_detected = False
             
             if pose_results.pose_landmarks:
                 mp_drawing.draw_landmarks(
                     frame_rgb, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                     mp_drawing.DrawingSpec(color=(92,229,180), thickness=2, circle_radius=2),
                     mp_drawing.DrawingSpec(color=(59,130,246), thickness=2, circle_radius=2)
                 )
                 
             if face_results.multi_face_landmarks:
                 face_detected = True
                 for face_landmarks in face_results.multi_face_landmarks:
                     mp_drawing.draw_landmarks(
                         frame_rgb, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                         None,
                         mp_drawing.DrawingSpec(color=(92,229,180), thickness=1, circle_radius=1)
                     )
                     
             # Only analyze emotion IF a face is physically present on camera
             if face_detected:
                 try:
                     result = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
                     if isinstance(result, list):
                         result = result[0]
                     current_state = result['dominant_emotion'].capitalize()
                     hud_color = (180, 229, 92) # Teal
                 except Exception:
                     current_state = "Analyzing..."
                     hud_color = (180, 229, 92)
             else:
                 current_state = "Target Lost / Camera Blocked"
                 hud_color = (0, 0, 255) # Red to indicate lost tracking
                 
             # Draw the HUD
             cv2.putText(frame_rgb, f"State: {current_state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, hud_color, 2)
             
             # Push frame (use_container_width eliminates the green warning boxes)
             video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
             
             # Logging Logic
             if time.time() - last_log_update > 2.5:
                 timestamp = datetime.now().strftime("%I:%M:%S %p")
                 mesh_status = "Active" if face_detected else "Inactive"
                 
                 new_log = f"[{timestamp}] Status: `{current_state}` | Mesh: `{mesh_status}`"
                 st.session_state['system_logs'].insert(0, new_log)
                 
                 if len(st.session_state['system_logs']) > 12:
                     st.session_state['system_logs'].pop()
                 
                 log_placeholder.markdown("\n\n".join(st.session_state['system_logs']))
                 last_log_update = time.time()

    if cap.isOpened():
        cap.release()

# --- UI Layout ---
st.title("S E N T H R O N")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Visual Network")
    video_placeholder = st.empty()

with col2:
    st.subheader("Command Center")
    
    # 1. FIXED BUTTON LAYOUT (Locks them to the top of the column)
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_button = st.button("Initialize Network")
    with btn_col2:
        stop_button = st.button("Deactivate Network")
        
    st.markdown("---")
    
    # 2. LOGGING AREA (Pushed safely below the buttons)
    st.subheader("System Terminal Log")
    log_placeholder = st.empty()
    
    # 3. TRIGGER LOGIC
    if start_button:
        run_senthron()
        
    if stop_button:
        st.session_state['system_logs'].insert(0, f"**[{datetime.now().strftime('%I:%M:%S %p')}] SYSTEM DEACTIVATED**")
        log_placeholder.markdown("\n\n".join(st.session_state['system_logs']))
