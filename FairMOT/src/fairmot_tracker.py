import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
from collections import deque
import json

class DLANet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(32, 64, 3, stride=2)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        self.fusion = nn.Conv2d(512, 64, 1)
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fusion(x)
        return x

class FairMOTNetwork(nn.Module):
    def __init__(self, num_classes=1, reid_dim=128):
        super().__init__()
        self.backbone = DLANet34()
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 1)
        )
        self.reid_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, reid_dim, 1),
            nn.BatchNorm2d(reid_dim)
        )
    def forward(self, x):
        backbone_features = self.backbone(x)
        heatmap = torch.sigmoid(self.heatmap_head(backbone_features))
        offset = self.offset_head(backbone_features)
        size = self.size_head(backbone_features)
        reid_features = self.reid_head(backbone_features)
        reid_features = F.normalize(reid_features, dim=1)
        return {
            'heatmap': heatmap,
            'offset': offset,
            'size': size,
            'reid': reid_features
        }

class FairMOTDetector:
    def __init__(self, conf_thresh=0.4, nms_thresh=0.4):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
    def decode_detections(self, outputs, input_size):
        heatmap = outputs['heatmap']
        offset = outputs['offset']
        size = outputs['size']
        batch_size, _, feat_h, feat_w = heatmap.shape
        peaks = self.find_peaks(heatmap, pool_size=3)
        detections = []
        for b in range(batch_size):
            batch_peaks = peaks[b]
            for peak_y, peak_x in batch_peaks:
                conf = heatmap[b, 0, peak_y, peak_x].item()
                if conf < self.conf_thresh:
                    continue
                offset_x = offset[b, 0, peak_y, peak_x].item()
                offset_y = offset[b, 1, peak_y, peak_x].item()
                w = size[b, 0, peak_y, peak_x].item()
                h = size[b, 1, peak_y, peak_x].item()
                scale_x = input_size[1] / feat_w
                scale_y = input_size[0] / feat_h
                center_x = (peak_x + offset_x) * scale_x
                center_y = (peak_y + offset_y) * scale_y
                x1 = center_x - w/2
                y1 = center_y - h/2
                x2 = center_x + w/2
                y2 = center_y + h/2
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'center': (center_x, center_y),
                    'feat_pos': (peak_y, peak_x)
                })
        return detections
    def find_peaks(self, heatmap, pool_size=3):
        pooled = F.max_pool2d(heatmap, pool_size, stride=1, padding=pool_size//2)
        peaks = (heatmap == pooled) & (heatmap > self.conf_thresh)
        peak_coords = []
        for b in range(heatmap.shape[0]):
            coords = torch.nonzero(peaks[b, 0], as_tuple=False)
            peak_coords.append(coords.cpu().numpy())
        return peak_coords

class FairMOTFeatureExtractor:
    def __init__(self, reid_dim=128):
        self.reid_dim = reid_dim
    def extract_detection_features(self, reid_feature_map, detections):
        features = []
        for detection in detections:
            feat_y, feat_x = detection['feat_pos']
            feature_vector = reid_feature_map[0, :, feat_y, feat_x]
            neighborhood_features = self.extract_neighborhood_features(
                reid_feature_map, feat_y, feat_x, radius=2
            )
            combined_features = torch.cat([
                feature_vector,
                neighborhood_features.mean(dim=0)
            ])
            features.append(combined_features.cpu().numpy())
        return np.array(features)
    def extract_neighborhood_features(self, feature_map, y, x, radius=2):
        B, C, H, W = feature_map.shape
        y_min = max(0, y - radius)
        y_max = min(H, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(W, x + radius + 1)
        neighborhood = feature_map[0, :, y_min:y_max, x_min:x_max]
        return neighborhood.view(C, -1)

class LightReIDNet(nn.Module):
    def __init__(self, reid_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, reid_dim),
            nn.ReLU(),
            nn.Linear(reid_dim, reid_dim)
        )
    def forward(self, x):
        features = self.features(x)
        return F.normalize(features, dim=1)

class PlayerFairMOTTracker:
    def __init__(self, model_path=None, conf_thresh=0.4, reid_thresh=0.6, max_disappeared=30):
        if model_path and model_path != "best.pt":
            self.network = self.load_fairmot_model(model_path)
            self.use_fairmot = True
        else:
            self.network = None
            self.yolo = YOLO("models/best.pt")
            self.use_fairmot = False
        self.conf_thresh = conf_thresh
        self.reid_thresh = reid_thresh
        self.max_disappeared = max_disappeared
        self.tracks = {}
        self.disappeared_tracks = {}
        self.next_id = 1
        self.feature_extractor = LightReIDNet(reid_dim=128)
        self.feature_extractor.eval()
        self.player_gallery = {}
        self.gallery_initialized = False
        self.frame_count = 0
        self.processing_times = []
        self.tracking_results = []
    def load_fairmot_model(self, model_path):
        model = FairMOTNetwork()
        return model
    def process_frame(self, frame):
        start_time = time.time()
        if self.use_fairmot:
            tracks = self.process_frame_fairmot(frame)
        else:
            tracks = self.process_frame_yolo_hybrid(frame)
        self.tracking_results.append({
            'frame_id': self.frame_count,
            'tracks': tracks.copy(),
            'processing_time': time.time() - start_time
        })
        self.frame_count += 1
        return tracks
    def process_frame_fairmot(self, frame):
        input_tensor = self.preprocess_frame(frame)
        with torch.no_grad():
            outputs = self.network(input_tensor)
        detector = FairMOTDetector(self.conf_thresh)
        detections = detector.decode_detections(outputs, frame.shape[:2])
        feature_extractor = FairMOTFeatureExtractor()
        reid_features = feature_extractor.extract_detection_features(
            outputs['reid'], detections
        )
        tracks = self.update_tracks(detections, reid_features)
        return tracks
    def process_frame_yolo_hybrid(self, frame):
        results = self.yolo(frame, conf=self.conf_thresh, verbose=False)
        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'center': ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                })
        reid_features = self.extract_reid_features_hybrid(frame, detections)
        tracks = self.update_tracks(detections, reid_features)
        return tracks
    def extract_reid_features_hybrid(self, frame, detections):
        if not detections:
            return []
        features = []
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                features.append(np.zeros(128))
                continue
            roi_resized = cv2.resize(roi, (64, 64))
            roi_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
            roi_tensor = roi_tensor.unsqueeze(0)
            with torch.no_grad():
                feature = self.feature_extractor(roi_tensor)
                features.append(feature.squeeze(0).numpy())
        return np.array(features)
    def update_tracks(self, detections, reid_features):
        if not detections:
            self._age_tracks()
            return self.get_active_tracks()
        if not self.tracks and not self.gallery_initialized:
            self._initialize_tracks(detections, reid_features)
            return self.get_active_tracks()
        association_matrix = self._compute_association_matrix(
            detections, reid_features
        )
        self._solve_assignment(detections, reid_features, association_matrix)
        return self.get_active_tracks()
    def _compute_association_matrix(self, detections, reid_features):
        track_ids = list(self.tracks.keys())
        n_detections = len(detections)
        n_tracks = len(track_ids)
        if n_tracks == 0:
            return np.array([])
        cost_matrix = np.zeros((n_detections, n_tracks))
        for i, (det, feat) in enumerate(zip(detections, reid_features)):
            for j, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                reid_similarity = self._compute_reid_similarity(
                    feat, track['reid_features']
                )
                spatial_cost = self._compute_spatial_cost(
                    det['center'], track['predicted_center']
                )
                size_cost = self._compute_size_cost(
                    det['bbox'], track['bbox']
                )
                total_cost = (
                    0.6 * (1 - reid_similarity) +
                    0.3 * spatial_cost +
                    0.1 * size_cost
                )
                cost_matrix[i, j] = total_cost
        return cost_matrix
    def _compute_reid_similarity(self, feat1, track_features):
        if len(track_features) > 0:
            avg_track_features = np.mean(track_features[-3:], axis=0)
            similarity = cosine_similarity([feat1], [avg_track_features])[0, 0]
            return max(0, similarity)
        return 0
    def _compute_spatial_cost(self, det_center, predicted_center):
        if predicted_center is None:
            return 1.0
        distance = np.linalg.norm(np.array(det_center) - np.array(predicted_center))
        normalized_distance = distance / (720 * np.sqrt(2))
        return min(1.0, normalized_distance)
    def _compute_size_cost(self, det_bbox, track_bbox):
        det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
        track_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
        if track_area == 0:
            return 1.0
        size_ratio = abs(det_area - track_area) / track_area
        return min(1.0, size_ratio)
    def _solve_assignment(self, detections, reid_features, cost_matrix):
        if cost_matrix.size == 0:
            self._initialize_tracks(detections, reid_features)
            return
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        track_ids = list(self.tracks.keys())
        matched_detections = set()
        matched_tracks = set()
        for row, col in zip(row_indices, col_indices):
            cost = cost_matrix[row, col]
            if cost < (1 - self.reid_thresh):
                track_id = track_ids[col]
                self._update_track(track_id, detections[row], reid_features[row])
                matched_detections.add(row)
                matched_tracks.add(track_id)
        for i, (det, feat) in enumerate(zip(detections, reid_features)):
            if i not in matched_detections:
                self._create_new_track(det, feat)
        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id]['disappeared'] += 1
                self.tracks[track_id]['predicted_center'] = self._predict_next_position(track_id)
        self._age_tracks()
    def _initialize_tracks(self, detections, reid_features):
        for det, feat in zip(detections, reid_features):
            self._create_new_track(det, feat)
    def _create_new_track(self, detection, reid_feature):
        self.tracks[self.next_id] = {
            'bbox': detection['bbox'],
            'center': detection['center'],
            'predicted_center': detection['center'],
            'reid_features': [reid_feature],
            'disappeared': 0,
            'confidence': detection['confidence'],
            'positions_history': [detection['center']],
            'velocity': (0, 0),
            'first_seen': self.frame_count
        }
        self.next_id += 1
    def _update_track(self, track_id, detection, reid_feature):
        track = self.tracks[track_id]
        old_center = track['center']
        new_center = detection['center']
        track['bbox'] = detection['bbox']
        track['center'] = new_center
        track['predicted_center'] = new_center
        track['reid_features'].append(reid_feature)
        track['disappeared'] = 0
        track['confidence'] = detection['confidence']
        track['positions_history'].append(new_center)
        track['velocity'] = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )
        if len(track['reid_features']) > 10:
            track['reid_features'] = track['reid_features'][-10:]
        if len(track['positions_history']) > 10:
            track['positions_history'] = track['positions_history'][-10:]
    def _predict_next_position(self, track_id):
        track = self.tracks[track_id]
        current_pos = track['center']
        velocity = track['velocity']
        predicted_pos = (
            current_pos[0] + velocity[0],
            current_pos[1] + velocity[1]
        )
        return predicted_pos
    def _age_tracks(self):
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track['disappeared'] > self.max_disappeared:
                self.disappeared_tracks[track_id] = {
                    'reid_features': track['reid_features'][-5:],
                    'last_seen': self.frame_count,
                    'positions_history': track['positions_history'][-5:]
                }
                tracks_to_remove.append(track_id)
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    def get_active_tracks(self):
        active_tracks = []
        for track_id, track in self.tracks.items():
            if track['disappeared'] <= 2:
                x1, y1, x2, y2 = track['bbox']
                active_tracks.append({
                    'id': track_id,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': track['confidence'],
                    'center': track['center'],
                    'disappeared': track['disappeared']
                })
        return active_tracks
    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, (608, 608))
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor
    def save_results(self, output_path="fairmot_tracking_results.json"):
        results = {
            'metadata': {
                'total_frames': self.frame_count,
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                'max_tracks': max([len(frame['tracks']) for frame in self.tracking_results]) if self.tracking_results else 0
            },
            'frames': self.tracking_results
        }
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Tracking results saved to {output_path}")
    def get_performance_stats(self):
        if not self.tracking_times:
            return {}
        return {
            'avg_fps': 1.0 / np.mean(self.processing_times),
            'avg_processing_time': np.mean(self.processing_times),
            'total_tracks_created': self.next_id - 1,
            'current_active_tracks': len(self.tracks),
            'disappeared_tracks': len(self.disappeared_tracks)
        } 