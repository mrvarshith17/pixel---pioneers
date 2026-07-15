import os
import pickle
import time
import urllib.request
from datetime import datetime
from enum import Enum, auto

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# --------------------------------------------------------------------------
# CONFIG (same values as the desktop agent)
# --------------------------------------------------------------------------
class Config:
    SUSPECT_FRAMES = 8
    CONFIRM_SECONDS = 3.0
    TORSO_ANGLE_THRESHOLD = 55
    ASPECT_RATIO_THRESHOLD = 1.2
    ESCALATION_TIMEOUT = 8.0
    MODEL_PATH = "pose_landmarker_lite.task"
    MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
    CLASSIFIER_PATH = "fall_classifier.pkl"
    EXPECTED_N_FEATURES = 5


NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 0, 11, 12, 23, 24
FEATURE_NAMES = ["torso_angle", "aspect_ratio", "shoulder_hip_ratio",
                 "nose_hip_ratio", "hip_shoulder_ratio"]


def ensure_model_downloaded(cfg: Config):
    if not os.path.exists(cfg.MODEL_PATH):
        urllib.request.urlretrieve(cfg.MODEL_URL, cfg.MODEL_PATH)


@st.cache_resource
def load_detector():
    cfg = Config()
    ensure_model_downloaded(cfg)
    base_options = mp_python.BaseOptions(model_asset_path=cfg.MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


@st.cache_resource
def load_classifier():
    cfg = Config()
    if not os.path.exists(cfg.CLASSIFIER_PATH):
        return None
    with open(cfg.CLASSIFIER_PATH, "rb") as f:
        model = pickle.load(f)
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and n_features != cfg.EXPECTED_N_FEATURES:
        return None  # mismatched model, fall back to thresholds
    return model


def extract_features(landmarks, w, h):
    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    ls, rs = pt(L_SHOULDER), pt(R_SHOULDER)
    lh, rh = pt(L_HIP), pt(R_HIP)
    nose = pt(NOSE)

    shoulder_mid = (ls + rs) / 2
    hip_mid = (lh + rh) / 2

    vec = hip_mid - shoulder_mid
    vertical = np.array([0, 1])
    cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-6)
    torso_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    aspect_ratio = bw / bh if bh > 0 else 0

    shoulder_width = np.linalg.norm(ls - rs) + 1e-6
    shoulder_hip_dy = abs(shoulder_mid[1] - hip_mid[1])
    nose_hip_dy = abs(nose[1] - hip_mid[1])
    hip_width = np.linalg.norm(lh - rh)

    return [torso_angle, aspect_ratio,
            shoulder_hip_dy / shoulder_width,
            nose_hip_dy / shoulder_width,
            hip_width / shoulder_width]


def is_down_posture(features, cfg, classifier):
    torso_angle, aspect_ratio = features[0], features[1]
    if classifier is not None:
        return classifier.predict([features])[0] == "fall", torso_angle, aspect_ratio
    down = (torso_angle > cfg.TORSO_ANGLE_THRESHOLD) or (aspect_ratio > cfg.ASPECT_RATIO_THRESHOLD)
    return down, torso_angle, aspect_ratio


# --------------------------------------------------------------------------
# AGENT STATE MACHINE (same logic as the desktop agent)
# --------------------------------------------------------------------------
class AgentState(Enum):
    MONITORING = auto()
    SUSPECTED_FALL = auto()
    CONFIRMED_FALL = auto()
    ESCALATED = auto()
    RESOLVED = auto()


class FallAgent:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = AgentState.MONITORING
        self.down_frame_count = 0
        self.state_entered_at = None
        self.incidents = []  # in-memory log for the web session

    def perceive(self, is_down: bool):
        self.down_frame_count = self.down_frame_count + 1 if is_down else 0
        self._decide(is_down)

    def _decide(self, is_down: bool):
        now = time.time()
        if self.state == AgentState.MONITORING:
            if self.down_frame_count >= self.cfg.SUSPECT_FRAMES:
                self._transition(AgentState.SUSPECTED_FALL, now)
        elif self.state == AgentState.SUSPECTED_FALL:
            if not is_down:
                self._transition(AgentState.MONITORING, now)
            elif now - self.state_entered_at >= self.cfg.CONFIRM_SECONDS:
                self._transition(AgentState.CONFIRMED_FALL, now)
        elif self.state == AgentState.CONFIRMED_FALL:
            if not is_down:
                self._transition(AgentState.RESOLVED, now)
            elif now - self.state_entered_at >= self.cfg.ESCALATION_TIMEOUT:
                self._transition(AgentState.ESCALATED, now)
        elif self.state == AgentState.ESCALATED:
            if not is_down:
                self._transition(AgentState.RESOLVED, now)
        elif self.state == AgentState.RESOLVED:
            if now - self.state_entered_at >= 2.0:
                self._transition(AgentState.MONITORING, now)

    def _transition(self, new_state, now):
        old_state = self.state
        self.state = new_state
        self.state_entered_at = now
        if new_state == AgentState.MONITORING:
            self.down_frame_count = 0
        if new_state in (AgentState.SUSPECTED_FALL, AgentState.CONFIRMED_FALL,
                         AgentState.ESCALATED, AgentState.RESOLVED):
            self.incidents.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "event": new_state.name,
                "from": old_state.name,
            })

    def acknowledge(self):
        if self.state in (AgentState.CONFIRMED_FALL, AgentState.ESCALATED):
            self._transition(AgentState.RESOLVED, time.time())

    def status_color(self):
        return {
            AgentState.MONITORING: (0, 200, 0),
            AgentState.SUSPECTED_FALL: (0, 200, 255),
            AgentState.CONFIRMED_FALL: (0, 0, 255),
            AgentState.ESCALATED: (0, 0, 150),
            AgentState.RESOLVED: (255, 180, 0),
        }[self.state]


# --------------------------------------------------------------------------
# VIDEO PROCESSOR -- runs once per incoming WebRTC frame
# --------------------------------------------------------------------------
class FallDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.cfg = Config()
        self.detector = load_detector()
        self.classifier = load_classifier()
        self.agent = FallAgent(self.cfg)
        self.frame_timestamp_ms = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect_for_video(mp_image, self.frame_timestamp_ms)
        self.frame_timestamp_ms += 33  # approx 30fps step

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            features = extract_features(landmarks, w, h)
            down, angle, aspect = is_down_posture(features, self.cfg, self.classifier)
            self.agent.perceive(down)
            for lm in landmarks:
                cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
            info = f"angle={angle:.0f} deg  aspect={aspect:.2f}"
        else:
            self.agent.perceive(is_down=False)
            info = "no person detected"

        color = self.agent.status_color()
        cv2.rectangle(img, (0, 0), (w, 40), color, -1)
        cv2.putText(img, f"STATE: {self.agent.state.name}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, info, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --------------------------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Fall & Safety Monitoring Agent", layout="centered")
st.title("Fall & Safety Monitoring Agent")

# Pre-warm the pose model and classifier BEFORE starting WebRTC. Loading
# these inside the WebRTC worker's own startup can exceed its 10-second
# initialization timeout (especially on first run, when the model file
# still needs to download) -- warming them here first means the worker
# just grabs the already-cached objects instantly.
with st.spinner("Loading pose detection model (first run only takes a bit longer)..."):
    load_detector()
    load_classifier()
st.caption("AI Arena 3.0 -- AI Vision theme. Perceive -> Decide -> Act -> Escalate.")

st.markdown(
    "Click **Start**, allow camera access, then step back so your whole body "
    "is visible. Simulate a fall (lie down for a few seconds) to see the "
    "agent go from **Monitoring** -> **Suspected** -> **Confirmed**."
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="fall-detection",
    video_processor_factory=FallDetectionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
if ctx.video_processor:
    st.subheader("Incident Log")
    incidents = ctx.video_processor.agent.incidents
    if incidents:
        st.table(incidents[-10:])
    else:
        st.write("No events yet -- still monitoring.")
else:
    st.info("Start the camera above to begin monitoring.")

st.markdown("---")
st.caption(
    "Note: this demo runs pose detection on the server for every viewer, so "
    "on a free hosting tier, only use one active viewer at a time for best "
    "performance."
)