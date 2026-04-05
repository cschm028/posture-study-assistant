import cv2
import mediapipe as mp
import numpy as np
from serial_send import send_score

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

baseline_head_drop = None
calibrated = False

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def get_landmarks(landmarks):
    left_ear       = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,       landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
    right_ear      = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,      landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
    left_shoulder  = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip       = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip      = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

    ear      = [(left_ear[0]+right_ear[0])/2,            (left_ear[1]+right_ear[1])/2]
    shoulder = [(left_shoulder[0]+right_shoulder[0])/2,   (left_shoulder[1]+right_shoulder[1])/2]
    hip      = [(left_hip[0]+right_hip[0])/2,             (left_hip[1]+right_hip[1])/2]
    return ear, shoulder, hip, left_shoulder, right_shoulder

def get_posture_score(landmarks, baseline):
    ear, shoulder, hip, left_shoulder, right_shoulder = get_landmarks(landmarks)

    neck_angle    = calculate_angle(ear, shoulder, hip)
    below_hip     = [hip[0], hip[1] + 0.1]
    torso_angle   = calculate_angle(shoulder, hip, below_hip)
    shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1]) * 100
    head_drop     = (ear[1] - shoulder[1]) * 100

    deviation  = head_drop - baseline
    head_score = max(0, 100 - abs(deviation) * 8)

    neck_score  = max(0, 100 - abs(neck_angle - 177) * 2)
    torso_score = max(0, 100 - abs(torso_angle - 179) * 3)
    tilt_score  = max(0, 100 - shoulder_tilt * 300)

    final_score = int((neck_score * 0.15) + (torso_score * 0.15) + (tilt_score * 0.1) + (head_score * 0.6))
    return final_score, neck_angle, torso_angle, head_drop, deviation

def get_status(score):
    if score >= 75:
        return "Good Posture", (0, 200, 0)
    elif score >= 50:
        return "Adjust Posture", (0, 165, 255)
    else:
        return "SIT UP!", (0, 0, 255)

# Load average baseline from previous calibrations
try:
    with open("calibrations.txt", "r") as f:
        values = [float(line.strip()) for line in f.readlines()]
    if values:
        baseline_head_drop = sum(values) / len(values)
        calibrated = False
        print(f"Loaded average from {len(values)} people: {baseline_head_drop:.1f} — Press C to calibrate your own")
except FileNotFoundError:
    print("No calibration data found. Press C to calibrate.")

cap = cv2.VideoCapture(0)

with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if not calibrated:
            cv2.putText(frame, "Sit up straight & press C to calibrate", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if calibrated:
                score, neck_angle, torso_angle, head_drop, deviation = get_posture_score(results.pose_landmarks.landmark, baseline_head_drop)
                print(f"Head drop: {head_drop:.1f} | Baseline: {baseline_head_drop:.1f} | Deviation: {deviation:.1f} | Score: {score}")
                status, color = get_status(score)
                send_score(score)

                cv2.putText(frame, f'Score: {score}',             (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 0, 0), 2)
                cv2.putText(frame, status,                         (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1,   (0, 0, 0), 2)
                cv2.putText(frame, f'Neck:  {neck_angle:.1f}',    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(frame, f'Torso: {torso_angle:.1f}',   (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                cv2.putText(frame, f'Deviation: {deviation:.1f}', (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        cv2.imshow('Posture Detection', frame)
        key = cv2.waitKey(100) & 0xFF

        if key == ord('c') and results.pose_landmarks:
            ear, shoulder, hip, _, _ = get_landmarks(results.pose_landmarks.landmark)
            baseline_head_drop = (ear[1] - shoulder[1]) * 100
            calibrated = True
            print(f"Calibrated! Baseline head drop: {baseline_head_drop:.1f}")

            with open("calibrations.txt", "a") as f:
                f.write(f"{baseline_head_drop}\n")
            print("Saved to calibrations.txt")

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()