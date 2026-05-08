import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import pandas as pd
import time
from collections import deque, Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from av import VideoFrame

# -------------------------------
# Load Model
# -------------------------------
with open('Human_action.pkl', 'rb') as f:
    model = pickle.load(f)

# -------------------------------
# MediaPipe Setup — GLOBALS (no instances here)
# -------------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("🧠 Human Action Recognition System")
st.caption("Real-time recognition using pose-based ML")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

col1, col2 = st.columns(2)
fps_placeholder = col1.empty()
latency_placeholder = col2.empty()
history_placeholder = st.empty()
confidence_bar = st.sidebar.empty()


# -------------------------------
# Video Processor
# -------------------------------
class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.prev_time = 0
        self.pred_buffer = deque(maxlen=10)
        self.history = deque(maxlen=5)
        # ✅ Holistic lives inside the instance — one per thread, thread-safe
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=0  # lightest model — important for free tier RAM
        )
        # ✅ Store metrics as instance variables instead of session_state
        self.fps = 0
        self.latency = 0
        self.prob = 0

    def recv(self, frame):
        start_time = time.time()

        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (640, 480))

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(image)  # ✅ self.holistic

        prediction = "..."
        prob = 0

        try:
            if results.pose_landmarks:
                pose = results.pose_landmarks.landmark

                row = []
                for lm in pose:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])

                X = pd.DataFrame([row])
                raw_pred = model.predict(X)[0]

                if hasattr(model, "predict_proba"):
                    prob = np.max(model.predict_proba(X))

                if prob < confidence_threshold:
                    raw_pred = "Unknown"

                self.pred_buffer.append(raw_pred)
                prediction = Counter(self.pred_buffer).most_common(1)[0][0]

                if len(self.history) == 0 or self.history[-1] != prediction:
                    self.history.append(prediction)

                mp_drawing.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )

        except Exception as e:
            print("Error:", e)

        # Overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        cv2.putText(img, f"ACTION: {prediction}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"CONF: {prob:.2f}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ✅ Store metrics on instance, not session_state
        self.latency = (time.time() - start_time) * 1000
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
        self.prev_time = current_time
        self.prob = prob

        return VideoFrame.from_ndarray(img, format="bgr24")


# -------------------------------
# Run Stream
# -------------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
ctx = webrtc_streamer(
    key="har-final",
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

# -------------------------------
# UI UPDATE — read from processor instance, not session_state
# -------------------------------
if ctx.video_processor:
    fps_placeholder.markdown(f"**FPS:** {ctx.video_processor.fps:.2f}")
    latency_placeholder.markdown(f"**Latency:** {ctx.video_processor.latency:.2f} ms")
    confidence_bar.progress(float(ctx.video_processor.prob))

    chips = "".join([
        f"<span style='background:#2ecc71;padding:6px;border-radius:10px;margin:3px;color:white'>{h}</span>"
        for h in ctx.video_processor.history
    ])
    history_placeholder.markdown(f"**Prediction History:**<br>{chips}", unsafe_allow_html=True)