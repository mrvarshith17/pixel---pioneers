import argparse
import csv
import os

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

OUTPUT_FILE = "pose_training_data.csv"
FEATURE_NAMES = [
    "torso_angle", "aspect_ratio", "shoulder_hip_dy",
    "nose_hip_dy", "shoulder_width", "hip_width",
]


def extract_features(landmarks, w, h):
    """Turn raw pose landmarks into a fixed-size feature vector."""
    lm = landmarks
    P = mp_pose.PoseLandmark

    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    ls, rs = pt(P.LEFT_SHOULDER), pt(P.RIGHT_SHOULDER)
    lh, rh = pt(P.LEFT_HIP), pt(P.RIGHT_HIP)
    nose = pt(P.NOSE)

    shoulder_mid = (ls + rs) / 2
    hip_mid = (lh + rh) / 2

    vec = hip_mid - shoulder_mid
    vertical = np.array([0, 1])
    cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-6)
    torso_angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    xs = [l.x * w for l in lm]
    ys = [l.y * h for l in lm]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    aspect_ratio = bw / bh if bh > 0 else 0

    shoulder_hip_dy = abs(shoulder_mid[1] - hip_mid[1])
    nose_hip_dy = abs(nose[1] - hip_mid[1])
    shoulder_width = np.linalg.norm(ls - rs)
    hip_width = np.linalg.norm(lh - rh)

    return [torso_angle, aspect_ratio, shoulder_hip_dy, nose_hip_dy, shoulder_width, hip_width]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    file_exists = os.path.exists(OUTPUT_FILE)
    f = open(OUTPUT_FILE, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(FEATURE_NAMES + ["label"])

    cap = cv2.VideoCapture(source)
    counts = {"normal": 0, "fall": 0}
    mode = None  # None = paused, "normal" or "fall" = actively recording every frame

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            features = None
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                features = extract_features(results.pose_landmarks.landmark, w, h)

            # If a recording mode is active, auto-save this frame's features
            if mode is not None and features is not None:
                writer.writerow(features + [mode])
                counts[mode] += 1

            # ---- overlay UI ----
            mode_label = f"RECORDING: {mode.upper()}" if mode else "PAUSED"
            mode_color = (0, 0, 255) if mode == "fall" else (0, 200, 0) if mode == "normal" else (150, 150, 150)
            cv2.rectangle(frame, (0, 0), (w, 40), mode_color, -1)
            cv2.putText(frame, mode_label, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"normal saved: {counts['normal']}  fall saved: {counts['fall']}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, "'n'=start normal  'f'=start fall  's'=stop  'q'=quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow("Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("n"):
                mode = "normal"
            elif key == ord("f"):
                mode = "fall"
            elif key == ord("s"):
                mode = None

    cap.release()
    cv2.destroyAllWindows()
    f.close()
    print(f"Saved {counts['normal']} normal + {counts['fall']} fall samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()