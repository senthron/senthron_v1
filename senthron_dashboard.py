import cv2
import streamlit as st
from deepface import DeepFace
import mediapipe as mp
from datetime import datetime
import time

# --- System Configuration ---
st.set_page_config(page_title="Senthron™: Visual Data Acquisition PoC", layout="wide")
st.title("Senthron™: Visual Data Acquisition PoC")
st.markdown("---")

# --- UI Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Visual Network")
    video_placeholder = st.empty()

with col2:
    st.subheader("State Activity Log")
    log_placeholder = st.empty()

# Initialize session state for our text logs
if 'system_logs' not in st.session_state:
    st.session_state['system_logs'] = []

def run_senthron():
    # Initialize MediaPipe components
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
                 st.error("Camera feed lost.")
                 break
                 
             # Streamlit requires RGB color space
             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             
             # Process Data Trackers
             pose_results = pose.process(frame_rgb)
             face_results = face_mesh.process(frame_rgb)
             
             # 1. Draw Skeleton Trackers
             if pose_results.pose_landmarks:
                 mp_drawing.draw_landmarks(frame_rgb, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                 
             # 2. Draw Facial Mesh
             if face_results.multi_face_landmarks:
                 for face_landmarks in face_results.multi_face_landmarks:
                     mp_drawing.draw_landmarks(frame_rgb, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                     
             # 3. Extract Dominant Emotion
             current_state = "Analyzing..."
             try:
                 result = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
                 if isinstance(result, list):
                     result = result[0]
                 current_state = result['dominant_emotion'].capitalize()
             except Exception:
                 pass
                 
             # Draw the HUD directly on the frame
             cv2.putText(frame_rgb, f"State: {current_state}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
             
             # Push the frame to the Streamlit UI
             video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
             
             # 4. Update the Text Logs (Every 2.5 seconds to avoid overwhelming the feed)
             if time.time() - last_log_update > 2.5:
                 timestamp = datetime.now().strftime("%I:%M:%S %p")
                 new_log = f"**[{timestamp}]** State: `{current_state}` | Mesh: `Active`"
                 
                 # Insert new log at the top of the list
                 st.session_state['system_logs'].insert(0, new_log)
                 
                 # Keep the log clean (only show the last 12 entries)
                 if len(st.session_state['system_logs']) > 12:
                     st.session_state['system_logs'].pop()
                 
                 # Render the text log in the UI
                 log_placeholder.markdown("\n\n".join(st.session_state['system_logs']))
                 last_log_update = time.time()

# A button to safely start the camera loop
if st.button("Initialize Senthron Network"):
    run_senthron()
