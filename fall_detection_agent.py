import argparse
import csv
import os
import pickle
import time
import urllib.request
from datetime import datetime
from enum import Enum, auto

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
class Config:
    SUSPECT_FRAMES = 8              # consecutive "down" frames before suspecting a fall
    CONFIRM_SECONDS = 3.0           # seconds "down" must persist to confirm
    TORSO_ANGLE_THRESHOLD = 55      # (fallback rule) degrees from vertical = "lying down"
    ASPECT_RATIO_THRESHOLD = 1.2    # (fallback rule) width/height ratio = horizontal
    ESCALATION_TIMEOUT = 8.0        # seconds to wait for ack before auto-escalating
    LOG_FILE = "incident_log.csv"
    MODEL_PATH = "pose_landmarker_lite.task"
    MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
    CLASSIFIER_PATH = "fall_classifier.pkl"   # trained model, optional
    EXPECTED_N_FEATURES = 5                    # must match the retrain script


# --------------------------------------------------------------------------
# AGENT STATE MACHINE (unchanged)
# --------------------------------------------------------------------------
class AgentState(Enum):
    MONITORING = auto()
    SUSPECTED_FALL = auto()
    CONFIRMED_FALL = auto()
    ESCALATED = auto()
    RESOLVED = auto()


class FallAgent:
    """The agentic decision layer: perceives posture over time, decides whether
    an event is real, acts by logging + alerting, and escalates autonomously
    if ignored."""

    def __init__(self, config: Config):
        self.cfg = config
        self.state = AgentState.MONITORING
        self.down_frame_count = 0
        self.state_entered_at = None
        self._ensure_log_file()

    def _ensure_log_file(self):
        if not os.path.exists(self.cfg.LOG_FILE):
            with open(self.cfg.LOG_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "event", "details"])

    def _log(self, event: str, details: str = ""):
        with open(self.cfg.LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), event, details])

    def perceive(self, is_down: bool):
        if is_down:
            self.down_frame_count += 1
        else:
            self.down_frame_count = 0
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

    def _transition(self, new_state: AgentState, now: float):
        old_state = self.state
        self.state = new_state
        self.state_entered_at = now
        if new_state == AgentState.MONITORING:
            self.down_frame_count = 0

        if new_state == AgentState.SUSPECTED_FALL:
            self._log("SUSPECTED_FALL", f"from {old_state.name}")
        elif new_state == AgentState.CONFIRMED_FALL:
            self._log("CONFIRMED_FALL", "alert raised")
            self._raise_alert()
        elif new_state == AgentState.ESCALATED:
            self._log("ESCALATED", "no response within timeout, escalating")
            self._escalate()
        elif new_state == AgentState.RESOLVED:
            self._log("RESOLVED", f"from {old_state.name}")

    def _raise_alert(self):
        print("\a")
        print("[ALERT] Fall confirmed! Notifying caregiver... (press 'r' in window to resolve)")

    def _escalate(self):
        print("[ESCALATION] No acknowledgement received -- escalating to emergency contact.")

    def acknowledge(self):
        if self.state in (AgentState.CONFIRMED_FALL, AgentState.ESCALATED):
            self._log("MANUAL_RESOLVE", "acknowledged by operator")
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
# POSE DETECTION (Tasks API) -- scale-invariant feature extraction
# --------------------------------------------------------------------------
NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 0, 11, 12, 23, 24

# NOTE: this order/name list MUST match the retrain script exactly.
FEATURE_NAMES = ["torso_angle", "aspect_ratio", "shoulder_hip_ratio",
                 "nose_hip_ratio", "hip_shoulder_ratio"]


def ensure_model_downloaded(cfg: Config):
    if not os.path.exists(cfg.MODEL_PATH):
        print("Downloading pose landmarker model (one-time, ~5MB)...")
        urllib.request.urlretrieve(cfg.MODEL_URL, cfg.MODEL_PATH)
        print("Downloaded.")


def create_detector(cfg: Config):
    base_options = mp_python.BaseOptions(model_asset_path=cfg.MODEL_PATH)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def extract_features(landmarks, w, h):
    """Scale-invariant features: torso_angle and aspect_ratio are already
    resolution-independent; the three distance-based measurements are
    normalized by shoulder_width so they mean the same thing regardless of
    camera resolution or distance-to-subject."""
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

    shoulder_width = np.linalg.norm(ls - rs) + 1e-6  # scale reference, avoid div-by-zero
    shoulder_hip_dy = abs(shoulder_mid[1] - hip_mid[1])
    nose_hip_dy = abs(nose[1] - hip_mid[1])
    hip_width = np.linalg.norm(lh - rh)

    shoulder_hip_ratio = shoulder_hip_dy / shoulder_width
    nose_hip_ratio = nose_hip_dy / shoulder_width
    hip_shoulder_ratio = hip_width / shoulder_width

    return [torso_angle, aspect_ratio, shoulder_hip_ratio, nose_hip_ratio, hip_shoulder_ratio]


def load_classifier(cfg: Config):
    """Load the trained model if present and its feature shape matches this
    script's current feature set. Returns None (fallback to thresholds)
    otherwise -- this prevents silently feeding a mismatched old model."""
    if not os.path.exists(cfg.CLASSIFIER_PATH):
        print(f"No {cfg.CLASSIFIER_PATH} found -- using threshold-based fallback rules.")
        return None

    with open(cfg.CLASSIFIER_PATH, "rb") as f:
        model = pickle.load(f)

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and n_features != cfg.EXPECTED_N_FEATURES:
        print(f"WARNING: {cfg.CLASSIFIER_PATH} expects {n_features} features but this "
              f"script now extracts {cfg.EXPECTED_N_FEATURES} (normalized ratios). "
              f"This looks like an OLD model trained on raw pixel distances -- "
              f"falling back to threshold rules instead of using it.")
        print("Retrain fall_classifier.pkl with the matching retrain script to use ML again.")
        return None

    print(f"Loaded trained classifier from {cfg.CLASSIFIER_PATH} -- using ML predictions.")
    return model


def is_down_posture(features, cfg: Config, classifier):
    torso_angle, aspect_ratio = features[0], features[1]
    if classifier is not None:
        pred = classifier.predict([features])[0]
        return pred == "fall", torso_angle, aspect_ratio
    down = (torso_angle > cfg.TORSO_ANGLE_THRESHOLD) or (aspect_ratio > cfg.ASPECT_RATIO_THRESHOLD)
    return down, torso_angle, aspect_ratio


# --------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fall & Safety Monitoring Agent")
    parser.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path to a video file")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cfg = Config()
    agent = FallAgent(cfg)

    ensure_model_downloaded(cfg)
    detector = create_detector(cfg)
    classifier = load_classifier(cfg)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: could not open video source '{args.source}'")
        return

    frame_timestamp_ms = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_duration_ms = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream / camera read failed.")
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += frame_duration_ms

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            features = extract_features(landmarks, w, h)
            down, angle, aspect = is_down_posture(features, cfg, classifier)
            agent.perceive(down)

            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

            info = f"angle={angle:.0f} deg  aspect={aspect:.2f}"
        else:
            agent.perceive(is_down=False)
            info = "no person detected"

        color = agent.status_color()
        cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
        cv2.putText(frame, f"STATE: {agent.state.name}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, info, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'r' to acknowledge/resolve, 'q' to quit", (10, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Fall & Safety Monitoring Agent", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            agent.acknowledge()

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Incident log saved to: {cfg.LOG_FILE}")


if __name__ == "__main__":
    main()