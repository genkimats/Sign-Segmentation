import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from datetime import datetime
from collections import deque
from dataclasses import dataclass

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

@dataclass
class SegmentationFeatures:
    left_wrist_speed: float
    right_wrist_speed: float
    left_wrist_accel: float
    right_wrist_accel: float
    left_dir_change: float      
    right_dir_change: float
    inter_hand_distance: float
    left_dist_to_rest: float    
    right_dist_to_rest: float
    left_dist_to_face: float    
    right_dist_to_face: float
    left_hand_spread: float     
    right_hand_spread: float
    activation_ratio: float     

class MultiFeaturePlotter:
    """A high-speed grid oscilloscope plotting 14 independent graphs."""
    def __init__(self, max_frames=90, width=1000, panel_h=100):
        self.max_frames = max_frames
        self.width = width
        self.panel_h = panel_h
        self.cols = 2
        self.col_w = self.width // self.cols
        
        # Define the 14 individual features, display titles, and line colors
        self.features = [
            ("L Speed", "left_wrist_speed", (0, 255, 0)),
            ("R Speed", "right_wrist_speed", (0, 165, 255)),
            ("L Accel", "left_wrist_accel", (100, 255, 100)),
            ("R Accel", "right_wrist_accel", (100, 200, 255)),
            ("L Dir Change", "left_dir_change", (0, 255, 255)),
            ("R Dir Change", "right_dir_change", (0, 200, 255)),
            ("Inter-Hand Dist", "inter_hand_distance", (255, 255, 255)),
            ("L Dist -> Torso", "left_dist_to_rest", (255, 0, 255)),
            ("R Dist -> Torso", "right_dist_to_rest", (150, 0, 150)),
            ("L Dist -> Face", "left_dist_to_face", (255, 255, 0)),
            ("R Dist -> Face", "right_dist_to_face", (200, 200, 0)),
            ("L Hand Spread", "left_hand_spread", (0, 255, 100)),
            ("R Hand Spread", "right_hand_spread", (0, 165, 100)),
            ("Activation Ratio", "activation_ratio", (0, 0, 255))
        ]
        
        self.rows = (len(self.features) + self.cols - 1) // self.cols
        self.total_h = self.rows * self.panel_h
        
        # Independent history buffers for every feature
        self.history = {f[1]: deque([0.0]*max_frames, maxlen=max_frames) for f in self.features}
        
        # High-water mark tracker for the maximum values
        self.peak_max = {f[1]: -float('inf') for f in self.features}

    def update(self, features_dict):
        for f in self.features:
            k = f[1]
            if k in features_dict:
                self.history[k].append(features_dict[k])

    def draw(self):
        canvas = np.zeros((self.total_h, self.width, 3), dtype=np.uint8)
        
        for i, (title, key, color) in enumerate(self.features):
            # Calculate grid position (Left->Right, Top->Bottom)
            col = i % self.cols
            row = i // self.cols
            
            x_offset = col * self.col_w
            y_offset = row * self.panel_h
            
            # Draw panel border
            cv2.rectangle(canvas, (x_offset, y_offset), (x_offset + self.col_w, y_offset + self.panel_h), (40, 40, 40), 1)
            
            # Draw Title and Current Value
            current_val = self.history[key][-1]
            cv2.putText(canvas, title, (x_offset + 10, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(canvas, f"{current_val:.3f}", (x_offset + self.col_w - 60, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Auto-scaling logic per panel (with persistent maximum)
            local_min = min(self.history[key])
            local_max = max(self.history[key])
            
            # Update the high-water mark if the current window exceeds it
            if local_max > self.peak_max[key]:
                self.peak_max[key] = local_max
                
            max_v = self.peak_max[key]
            min_v = local_min
            
            rng = max_v - min_v if max_v > min_v else 1e-5
            
            pts = []
            for x_idx, val in enumerate(self.history[key]):
                x = x_offset + int((x_idx / self.max_frames) * self.col_w)
                norm = (val - min_v) / rng
                # Leave padding for text at the top (40px) and bottom margin (10px)
                y = y_offset + self.panel_h - 10 - int(norm * (self.panel_h - 40))
                pts.append([x, y])
                
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 1)
            
        return canvas

class SegmentationFeatureExtractor:
    def __init__(self, history_size=3):
        self.left_wrist_history = deque(maxlen=max(3, history_size))
        self.right_wrist_history = deque(maxlen=max(3, history_size))
        self.prev_l_speed = 0.0
        self.prev_r_speed = 0.0

    def _get_scale_factor(self, pose_landmarks):
        if not pose_landmarks: return 1.0
        lm = pose_landmarks.landmark
        left_ear = np.array([lm[7].x, lm[7].y, lm[7].z])
        right_ear = np.array([lm[8].x, lm[8].y, lm[8].z])
        face_size = np.linalg.norm(left_ear - right_ear)
        if face_size < 0.02:
            l_shoulder = np.array([lm[11].x, lm[11].y, lm[11].z])
            r_shoulder = np.array([lm[12].x, lm[12].y, lm[12].z])
            shoulder_width = np.linalg.norm(l_shoulder - r_shoulder)
            return float(max(shoulder_width / 3.0, 1e-5))
        return float(face_size)

    def _get_torso_reference(self, pose_landmarks):
        if not pose_landmarks: return np.zeros(3)
        lm = pose_landmarks.landmark
        return (np.array([lm[23].x, lm[23].y, lm[23].z]) + np.array([lm[24].x, lm[24].y, lm[24].z])) / 2.0

    def _get_face_reference(self, pose_landmarks):
        if not pose_landmarks: return np.zeros(3)
        lm = pose_landmarks.landmark[0]
        return np.array([lm.x, lm.y, lm.z])

    def _get_hand_spread(self, hand_landmarks):
        if not hand_landmarks: return 0.0
        lm = hand_landmarks.landmark
        wrist = np.array([lm[0].x, lm[0].y, lm[0].z])
        fingertip_indices = [4, 8, 12, 16, 20] 
        distances = [np.linalg.norm(np.array([lm[i].x, lm[i].y, lm[i].z]) - wrist) for i in fingertip_indices]
        return float(np.mean(distances))

    def _get_wrist_position(self, pose_landmarks, hand_landmarks, is_left=True):
        if hand_landmarks:
            lm = hand_landmarks.landmark[0]
            return np.array([lm.x, lm.y, lm.z])
        elif pose_landmarks:
            idx = 15 if is_left else 16
            lm = pose_landmarks.landmark[idx]
            if lm.visibility > 0.5: 
                return np.array([lm.x, lm.y, lm.z])
        return None

    def _calculate_kinematics(self, current_pos, history_buffer, prev_speed):
        if current_pos is None: return 0.0, 0.0, prev_speed 
        history_buffer.append(current_pos)
        if len(history_buffer) < 2: return 0.0, 0.0, 0.0
        speed = np.linalg.norm(history_buffer[-1] - history_buffer[0])
        accel = speed - prev_speed
        return float(speed), float(accel), speed

    def _calculate_directional_change(self, history_buffer, movement_threshold=0.15):
        if len(history_buffer) < 3: return 1.0 
        v1 = history_buffer[-2] - history_buffer[-3]
        v2 = history_buffer[-1] - history_buffer[-2]
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        
        if norm_v1 < movement_threshold or norm_v2 < movement_threshold: 
            return 1.0 
            
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return float(np.clip(cos_theta, -1.0, 1.0))

    def extract(self, results) -> SegmentationFeatures:
        scale = self._get_scale_factor(results.pose_landmarks)
        torso_pt = self._get_torso_reference(results.pose_landmarks)
        face_pt = self._get_face_reference(results.pose_landmarks)
        
        raw_l_wrist = self._get_wrist_position(results.pose_landmarks, results.left_hand_landmarks, is_left=True)
        raw_r_wrist = self._get_wrist_position(results.pose_landmarks, results.right_hand_landmarks, is_left=False)

        norm_face = (face_pt - torso_pt) / scale if face_pt is not None else None
        norm_l_wrist = (raw_l_wrist - torso_pt) / scale if raw_l_wrist is not None else None
        norm_r_wrist = (raw_r_wrist - torso_pt) / scale if raw_r_wrist is not None else None

        l_speed, l_accel, self.prev_l_speed = self._calculate_kinematics(norm_l_wrist, self.left_wrist_history, self.prev_l_speed)
        r_speed, r_accel, self.prev_r_speed = self._calculate_kinematics(norm_r_wrist, self.right_wrist_history, self.prev_r_speed)

        l_dir_change = self._calculate_directional_change(self.left_wrist_history)
        r_dir_change = self._calculate_directional_change(self.right_wrist_history)

        inter_hand_dist = float(np.linalg.norm(norm_l_wrist - norm_r_wrist)) if norm_l_wrist is not None and norm_r_wrist is not None else 0.0
        l_dist_torso = float(np.linalg.norm(norm_l_wrist)) if norm_l_wrist is not None else 0.0
        r_dist_torso = float(np.linalg.norm(norm_r_wrist)) if norm_r_wrist is not None else 0.0
        l_dist_face = float(np.linalg.norm(norm_l_wrist - norm_face)) if norm_l_wrist is not None and norm_face is not None else 0.0
        r_dist_face = float(np.linalg.norm(norm_r_wrist - norm_face)) if norm_r_wrist is not None and norm_face is not None else 0.0

        l_spread = self._get_hand_spread(results.left_hand_landmarks) / scale
        r_spread = self._get_hand_spread(results.right_hand_landmarks) / scale

        activation_ratio = r_speed / (l_speed + r_speed + 1e-5)

        return SegmentationFeatures(
            left_wrist_speed=l_speed, right_wrist_speed=r_speed,
            left_wrist_accel=l_accel, right_wrist_accel=r_accel,
            left_dir_change=l_dir_change, right_dir_change=r_dir_change,
            inter_hand_distance=inter_hand_dist,
            left_dist_to_rest=l_dist_torso, right_dist_to_rest=r_dist_torso,
            left_dist_to_face=l_dist_face, right_dist_to_face=r_dist_face,
            left_hand_spread=l_spread, right_hand_spread=r_spread,
            activation_ratio=activation_ratio
        )

    def flatten(self, features: SegmentationFeatures) -> np.ndarray:
        return np.array([
            features.left_wrist_speed, features.right_wrist_speed,
            features.left_wrist_accel, features.right_wrist_accel,
            features.left_dir_change, features.right_dir_change,
            features.inter_hand_distance,
            features.left_dist_to_rest, features.right_dist_to_rest,
            features.left_dist_to_face, features.right_dist_to_face,
            features.left_hand_spread, features.right_hand_spread,
            features.activation_ratio
        ], dtype=np.float32)

def draw_visual_aids(image, results):
    h, w, _ = image.shape
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    pts = {} 
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        pts['face'] = (int(lm[0].x * w), int(lm[0].y * h))
        cv2.circle(image, pts['face'], 8, (0, 255, 255), -1) 
        
        pts['torso'] = (int(((lm[23].x + lm[24].x) / 2) * w), int(((lm[23].y + lm[24].y) / 2) * h))
        cv2.circle(image, pts['torso'], 10, (255, 0, 255), -1) 

        pts['l_wrist'] = (int(lm[15].x * w), int(lm[15].y * h)) if lm[15].visibility > 0.5 else None
        pts['r_wrist'] = (int(lm[16].x * w), int(lm[16].y * h)) if lm[16].visibility > 0.5 else None

    if results.left_hand_landmarks:
        pts['l_wrist'] = (int(results.left_hand_landmarks.landmark[0].x * w), int(results.left_hand_landmarks.landmark[0].y * h))
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
    if results.right_hand_landmarks:
        pts['r_wrist'] = (int(results.right_hand_landmarks.landmark[0].x * w), int(results.right_hand_landmarks.landmark[0].y * h))
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    for wrist_key in ['l_wrist', 'r_wrist']:
        wrist_pt = pts.get(wrist_key)
        if wrist_pt:
            cv2.circle(image, wrist_pt, 8, (0, 0, 255), -1) 
            if 'torso' in pts:
                cv2.line(image, wrist_pt, pts['torso'], (255, 0, 255), 1)
            if 'face' in pts:
                cv2.line(image, wrist_pt, pts['face'], (0, 255, 255), 1)
                
    if pts.get('l_wrist') and pts.get('r_wrist'):
        cv2.line(image, pts['l_wrist'], pts['r_wrist'], (255, 255, 255), 2) 

def main():
    headers = [
        "frame_idx", "left_wrist_speed", "right_wrist_speed",
        "left_wrist_accel", "right_wrist_accel", "left_dir_change",
        "right_dir_change", "inter_hand_distance", "left_dist_to_rest",
        "right_dist_to_rest", "left_dist_to_face", "right_dist_to_face",
        "left_hand_spread", "right_hand_spread", "activation_ratio"
    ]

    extractor = SegmentationFeatureExtractor(history_size=3)
    
    # Grid width=1000, height per row=100 (7 rows total = 700px tall)
    plotter = MultiFeaturePlotter(max_frames=90, width=1000, panel_h=100)

    cap = cv2.VideoCapture(0)

    # Initialize camera window
    cv2.namedWindow("Segmentation Pipeline", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Segmentation Pipeline", 600, 600)
    cv2.moveWindow("Segmentation Pipeline", 0, 80)  # Top-Left corner

    # Initialize telemetry window
    cv2.namedWindow("Real-Time Telemetry", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time Telemetry", 870, 700)
    cv2.moveWindow("Real-Time Telemetry", 600, 80)  # Placed directly to the right of the camera

    is_recording = False
    is_counting_down = False
    countdown_start_time = 0
    frame_idx = 0
    csv_file = None
    csv_writer = None

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            key = cv2.waitKey(5) & 0xFF
            if key == 27: 
                break
            elif key == ord('r') and not is_recording and not is_counting_down:
                is_counting_down = True
                countdown_start_time = time.time()
                print("Countdown started...")

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            features = extractor.extract(results)
            feature_vector = extractor.flatten(features)
            
            # --- Update and Draw Grid Plots ---
            plotter.update(features.__dict__)
            cv2.imshow("Real-Time Telemetry", plotter.draw())
            
            # --- Drawing and Display ---
            draw_visual_aids(frame, results)
            display_img = cv2.flip(frame, 1) 

            if is_counting_down:
                elapsed_time = time.time() - countdown_start_time
                remaining_time = 3 - int(elapsed_time)
                
                if remaining_time > 0:
                    h, w, _ = display_img.shape
                    cv2.putText(display_img, str(remaining_time), (w//2 - 50, h//2 + 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)
                else:
                    is_counting_down = False
                    is_recording = True
                    frame_idx = 0
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                    csv_filename = f"segmentation_features_{timestamp}.csv"
                    csv_file = open(csv_filename, mode='w', newline='')
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(headers)
                    print(f"Recording started: {csv_filename}")

            if is_recording:
                cv2.putText(display_img, "[REC] RECORDING - Press ESC to Stop", (15, display_img.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                row_data = [frame_idx] + feature_vector.tolist()
                csv_writer.writerow(row_data)
                frame_idx += 1
                
            elif not is_counting_down:
                cv2.putText(display_img, "Press 'R' to Start Recording", (15, display_img.shape[0] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Segmentation Pipeline", display_img)

    cap.release()
    cv2.destroyAllWindows()
    if csv_file is not None:
        csv_file.close()
        print(f"Recording saved and closed.")

if __name__ == "__main__":
    main()