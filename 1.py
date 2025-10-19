from platform import system
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import random
from collections import deque

# CONFIG
CAM_INDEX = 0
W, H = 640, 480

# Auto-Calibration: Keep mouth Closed during warmup
WARMUP_SEC = 2.0        # Seconds to learn closed-mouth baseline
DELTA = 0.10            # Threshold = baseLine + DELTA (tune 0.06-0.14)
ABS_MIN_THR = 0.28      # Safety floor for the threshold
CONSEC_FRAMES = 3       # Frames above threshold required to register "open"
SMOOTH_IN = 5           # Moving average to reduce jitter
HYSTERESIS = 0.02       # Gap around threshold to avoid chatter

# HUD / Demo rectangle
PLAYER_W, PLAYER_H = 50, 50     # Player rectangle size
PLAYER_DY = 30                  # Player rectangle vertical placement

OBSTACLE_W = 50
OBSTACLE_SPEED = 5
OBSTACLE_SPACING = 300
GAP_MIN = 140
GAP_MAX = 200
TOP_CLEAR_MIN = 80
BOTTOM_CLEAR_MIN = 80

Start = False

def gap_y():
    gap = random.randint(GAP_MIN, GAP_MAX)

    min_cy = TOP_CLEAR_MIN + gap // 2
    max_cy = H - BOTTOM_CLEAR_MIN - gap // 2
    cy = random.randint(min_cy, max_cy)
    return cy, gap

num_pairs = 5
tubes = []
for i in range(num_pairs):
    x = W + i * OBSTACLE_SPACING
    cy, gap = gap_y()
    tubes.append({"x": x, "cy": cy, "gap": gap})

# Mouth Landmarks
# Use robust standard indices: mouth corners 61 (left) and 291 (right)
MOUTH_CORNERS = (61, 291)
# Vertical pairs across the lips (outer and inner midlines)
MOUTH_VERTICAL_PAIRS = [(78, 308), (13, 14), (82, 312)]

# HELPERS
def eucliden_distance(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    return math.hypot(x1 - x0, y1 - y0)

def calculate_mar(landmarks_px):
    """
        MAR = (avg vertical lip gap) / (mouth width)
        landmarks_px: list of (x, y) pixel coords for all face landmarks
    """
    xL, yL = landmarks_px[MOUTH_CORNERS[0]]
    xR, yR = landmarks_px[MOUTH_CORNERS[1]]
    mouth_width = eucliden_distance((xL, yL), (xR, yR)) + 1e-6

    verts = []
    for a, b in MOUTH_VERTICAL_PAIRS:
        xa, ya = landmarks_px[a]
        xb, yb = landmarks_px[b]
        verts.append(eucliden_distance((xa, ya), (xb, yb)))
    mouth_open = sum(verts) / len(verts)
    return mouth_open / mouth_width

def aabb_intersect(ax, ay, aw, ah, bx, by, bw, bh):
    # Return True if rectangles intersect
    no_overlap = (ax + aw <= bx) or (bx + bw <= ax) or (ay + ah <= by) or (by + bh <= ay)
    return not no_overlap

def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    # Use standard frame size properties (XI_* are for Ximea cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    if not cap.isOpened():
        raise SystemExit("Could not open webcam!")

    SCORE = 0

    mp_face_mesh = mp.solutions.face_mesh
    # refine_landmarks if True gives extra precision for lips
    with mp_face_mesh.FaceMesh(
        static_image_mode = False,
        max_num_faces = 1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    ) as face_mesh:

        # State
        start_time = time.time()
        mar_hist = deque(maxlen = SMOOTH_IN)
        baseline_samples = []
        threshold = None
        frame_counter = 0
        mouth_open_counter = 0
        mouth_was_open = False

        # Player rectangles
        player_x = 50
        player_y = H // 2 - PLAYER_H // 2

        print("Press 'q' or ESC to quit, 'r' to recalibrate, '['/']' to adjust threshold")
        prev_time = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            collided = False
            frame = cv2.flip(frame, 1)
            rightmost_x = max(t["x"] for t in tubes)
            for t in tubes:
                if t["x"] < -OBSTACLE_W:
                    t["x"] = rightmost_x + OBSTACLE_SPACING
                    t["cy"], t["gap"] = gap_y()
                    rightmost_x = t["x"]
            for t in tubes:
                x = t["x"]
                cy = t["cy"]
                gap = t["gap"]
                top_h = cy - gap // 2
                bottom_y = cy + gap // 2

                # TOP TUBE
                cv2.rectangle(frame, (x, 0), (x + OBSTACLE_W, top_h), (200, 100, 50), -1)
                # BOTTOM TUBE
                cv2.rectangle(frame, (x, bottom_y), (x + OBSTACLE_W, H), (200, 100, 50), -1)

                # COLLISION WITH PLAYER
                px, py, pw, ph = player_x, player_y, PLAYER_W, PLAYER_H

                # TOP RECT
                if top_h > 0:
                    if aabb_intersect(px, py, pw, ph, x, 0, OBSTACLE_W, top_h):
                        collided = True

                # BOTTOM RECT
                if not collided and bottom_y < H:
                    if aabb_intersect(px, py, pw, ph, x, bottom_y, OBSTACLE_W, H - bottom_y):
                        collided = True

            h, w = frame.shape[:2]
            if collided is False:
                for t in tubes:
                    t["x"] -= OBSTACLE_SPEED
            if collided is True:
                for t in tubes:
                    t["x"] += 0

            if collided is False:
                player_y += 2
            if collided is False:
                SCORE += 1
                cv2.putText(frame, f"Score: {int(SCORE)}", (20, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)
            if collided is True:
                cv2.putText(frame, f"SCORE: {int(SCORE)}", (W // 2 - 150, H // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 5)

            # MediaPipe expects contiguous RGB uint8
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)

            results = face_mesh.process(rgb)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time

            #PLAYER
            cv2.rectangle(frame, (player_x, player_y),
                          (player_x + PLAYER_W, player_y + PLAYER_H),
                          (50, 220, 50), -1)
            # player_y = (player_y + PLAYER_DY) % (H - PLAYER_H)

            if results.multi_face_landmarks:
                fl = results.multi_face_landmarks[0].landmark
                # Convert all landmarks to pixel coords once
                landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in fl]

                mar = calculate_mar(landmarks_px)
                mar_hist.append(mar)
                smooth_mar = sum(mar_hist) / len(mar_hist)

                elapsed = time.time() - start_time
                if elapsed < WARMUP_SEC or threshold is None:
                    # Assume mouth closed during warmup
                    baseline_samples.append(smooth_mar)
                    cv2.putText(frame, "Calibrating... keep mouth CLOSED!",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Update threshold continuously until warmup done
                    if baseline_samples:
                        baseline = float(np.median(baseline_samples))
                        threshold = max(ABS_MIN_THR, baseline + DELTA)
                else:
                    open_thr = threshold + HYSTERESIS
                    close_thr = threshold - HYSTERESIS

                    if not mouth_was_open:
                        if smooth_mar > open_thr:
                            frame_counter += 1
                            # Register as open once consecutive frames threshold is met
                            if frame_counter >= CONSEC_FRAMES:
                                if collided is False:
                                    player_y -= 50
                                mouth_open_counter += 1
                                mouth_was_open = True
                                frame_counter = 0
                        else:
                            frame_counter = 0
                    else:
                        if smooth_mar < close_thr:
                            mouth_was_open = False
                            frame_counter = 0

                    # HUD
                    cv2.putText(frame, f"Mouth Opens: {mouth_open_counter}",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, f"MAR: {smooth_mar: .3f}  Thr: {threshold: .3f} R - Reset",
                                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    # RESET TEXT
                    # Visual status indicator (green when over threshold)
                    color = (0, 255, 0) if smooth_mar > open_thr else (0, 0, 255)
                    cv2.circle(frame, (w - 40, 40), 12, color, -1)

                # Always draw lip points when face detected (even during calibration)
                for idx in [61, 291, 78, 308, 13, 14, 82, 312]:
                    x, y = landmarks_px[idx]
                    cv2.circle(frame, (x, y), 2, (255, 100, 100), -1)
            else:
                frame_counter = 0
                mouth_was_open = False
                cv2.putText(frame, "Face not found!",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {int(fps)}", (W - 150, H - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Mouth Flappy Bird", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key in (ord('r'), ord('R')):
                    start_time = time.time()
                    mar_hist.clear()
                    baseline_samples = []
                    threshold = None
                    frame_counter = 0
                    mouth_open_counter = 0
                    mouth_was_open = False
                    player_y = H // 2 - PLAYER_H // 2
                    SCORE = 0
                    tubes.clear()
                    for i in range(num_pairs):
                        x = W + i * OBSTACLE_SPACING
                        cy, gap = gap_y()
                        tubes.append({"x": x, "cy": cy, "gap": gap})

            if key == ord(']') and threshold is not None:
                threshold = min(1.5, threshold + 0.01)
            if key == ord('[') and threshold is not None:
                threshold = max(0.05, threshold - 0.01)

    cap.release()
    cv2.destroyAllWindows()

main()
