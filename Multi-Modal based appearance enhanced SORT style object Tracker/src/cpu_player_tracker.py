import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

class CPUPlayerTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 8
        self.similarity_threshold = 0.6
    def extract_comprehensive_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.zeros(128)
        roi_std = cv2.resize(roi, (64, 64))
        hsv = cv2.cvtColor(roi_std, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        color_features = np.concatenate([
            hist_h.flatten() / (hist_h.sum() + 1e-6),
            hist_s.flatten() / (hist_s.sum() + 1e-6),
            hist_v.flatten() / (hist_v.sum() + 1e-6)
        ])
        gray = cv2.cvtColor(roi_std, cv2.COLOR_BGR2GRAY)
        lbp = self.local_binary_pattern(gray)
        lbp_hist = cv2.calcHist([lbp], [0], None, [32], [0, 32])
        texture_features = lbp_hist.flatten() / (lbp_hist.sum() + 1e-6)
        height, width = roi_std.shape[:2]
        aspect_ratio = width / height if height > 0 else 0
        area_ratio = (roi_std.shape[0] * roi_std.shape[1]) / (64 * 64)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (64 * 64)
        colors_reshaped = roi_std.reshape(-1, 3)
        dominant_color = np.mean(colors_reshaped, axis=0) / 255.0
        shape_features = np.array([
            aspect_ratio, area_ratio, edge_density,
            dominant_color[0], dominant_color[1], dominant_color[2],
            np.std(colors_reshaped[:, 0]) / 255.0,
            np.std(colors_reshaped[:, 1]) / 255.0
        ])
        frame_h, frame_w = frame.shape[:2]
        center_x = (x1 + x2) / 2 / frame_w
        center_y = (y1 + y2) / 2 / frame_h
        bbox_w = (x2 - x1) / frame_w
        bbox_h = (y2 - y1) / frame_h
        position_features = np.array([center_x, center_y, bbox_w, bbox_h])
        combined = np.concatenate([
            color_features[:48],
            texture_features[:32], 
            shape_features[:8],
            position_features
        ])
        padded = np.zeros(128)
        padded[:len(combined)] = combined
        return padded
    def local_binary_pattern(self, image, radius=1, n_points=8):
        lbp = np.zeros_like(image, dtype=np.uint8)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                binary_string = ""
                angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
                for angle in angles:
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                        if image[x, y] >= center:
                            binary_string += "1"
                        else:
                            binary_string += "0"
                lbp[i, j] = int(binary_string, 2) if binary_string else 0
        return lbp
    def compute_similarity_matrix(self, new_features, track_features):
        if not track_features:
            return np.array([])
        new_feat_matrix = np.array(new_features)
        track_feat_matrix = np.array(track_features)
        similarity_matrix = cosine_similarity(new_feat_matrix, track_feat_matrix)
        return similarity_matrix
    def update_tracks(self, frame, detections):
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return self.get_active_tracks()
        new_features = []
        for det in detections:
            feat = self.extract_comprehensive_features(frame, det['bbox'])
            new_features.append(feat)
        if not self.tracks:
            for i, (det, feat) in enumerate(zip(detections, new_features)):
                self.tracks[self.next_id] = {
                    'bbox': det['bbox'],
                    'features': feat,
                    'feature_history': [feat],
                    'disappeared': 0,
                    'confidence': det.get('confidence', 1.0)
                }
                self.next_id += 1
            return self.get_active_tracks()
        track_ids = list(self.tracks.keys())
        track_features = []
        for track_id in track_ids:
            track = self.tracks[track_id]
            recent_features = track['feature_history'][-3:]
            avg_features = np.mean(recent_features, axis=0)
            track_features.append(avg_features)
        similarity_matrix = self.compute_similarity_matrix(new_features, track_features)
        if similarity_matrix.size > 0:
            cost_matrix = 1 - similarity_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_detections = set()
            matched_tracks = set()
            for row, col in zip(row_indices, col_indices):
                similarity = similarity_matrix[row, col]
                if similarity > self.similarity_threshold:
                    track_id = track_ids[col]
                    self.tracks[track_id]['bbox'] = detections[row]['bbox']
                    self.tracks[track_id]['features'] = new_features[row]
                    self.tracks[track_id]['feature_history'].append(new_features[row])
                    self.tracks[track_id]['disappeared'] = 0
                    self.tracks[track_id]['confidence'] = detections[row].get('confidence', 1.0)
                    if len(self.tracks[track_id]['feature_history']) > 5:
                        self.tracks[track_id]['feature_history'] = \
                            self.tracks[track_id]['feature_history'][-5:]
                    matched_detections.add(row)
                    matched_tracks.add(track_id)
            for i, (det, feat) in enumerate(zip(detections, new_features)):
                if i not in matched_detections:
                    self.tracks[self.next_id] = {
                        'bbox': det['bbox'],
                        'features': feat,
                        'feature_history': [feat],
                        'disappeared': 0,
                        'confidence': det.get('confidence', 1.0)
                    }
                    self.next_id += 1
            for i, track_id in enumerate(track_ids):
                if track_id not in matched_tracks:
                    self.tracks[track_id]['disappeared'] += 1
                    if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                        del self.tracks[track_id]
        return self.get_active_tracks()
    def get_active_tracks(self):
        active_tracks = []
        for track_id, track in self.tracks.items():
            if track['disappeared'] <= 2:
                x1, y1, x2, y2 = track['bbox']
                active_tracks.append({
                    'id': track_id,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': track['confidence']
                })
        return active_tracks

def run_cpu_tracking(video_path, model_path="best.pt", output_path="output/cpu_tracked_output.mp4", frame_skip=1, results_csv="output/tracker_results.csv"):
    from ultralytics import YOLO
    import os
    import csv
    model = YOLO(model_path)
    tracker = CPUPlayerTracker()
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    model.to('cpu')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_skip > 1 and frame_count % frame_skip != 0:
            frame_count += 1
            continue
        results_yolo = model(frame, conf=0.4, device='cpu', verbose=False)
        detections = []
        if results_yolo[0].boxes is not None:
            for box in results_yolo[0].boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append({
                    'bbox': bbox,
                    'confidence': conf
                })
        tracks = tracker.update_tracks(frame, detections)
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['id']
            conf = track['confidence']
            color = (0, 255, 0) if conf > 0.7 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {track_id} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            results.append([frame_count, track_id, x1, y1, x2, y2])
        out.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    cap.release()
    out.release()
    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "id", "x1", "y1", "x2", "y2"])
        writer.writerows(results)
    print(f"Saved output video to {output_path}")
    print(f"Saved tracking results to {results_csv}") 