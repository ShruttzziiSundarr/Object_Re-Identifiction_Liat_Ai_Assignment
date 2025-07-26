import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet18


resnet = resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet.eval()
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 64)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_deep_feature(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return None
    with torch.no_grad():
        tensor = transform(crop).unsqueeze(0)
        feature = resnet(tensor).squeeze().numpy()
    return feature

class Track:
    def __init__(self, bbox, track_id, appearance, deep_feature):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.x[:4] = bbox.reshape((4, 1))
        self.kf.F = np.eye(7)
        self.kf.H = np.eye(4, 7)
        self.time_since_update = 0
        self.id = track_id
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.bbox = bbox
        self.appearance = appearance  # color histogram
        self.deep_feature = deep_feature  # deep feature vector
        self.gallery = [appearance] if appearance is not None else []
        self.deep_gallery = [deep_feature] if deep_feature is not None else []

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:4].reshape((4,))

    def update(self, bbox, appearance, deep_feature):
        self.kf.update(bbox)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.bbox = bbox
        self.appearance = appearance
        self.deep_feature = deep_feature
        if appearance is not None:
            self.gallery.append(appearance)
        if deep_feature is not None:
            self.deep_gallery.append(deep_feature)

    def get_gallery_mean(self):
        if not self.gallery:
            return None
        return np.mean(self.gallery, axis=0)

    def get_deep_gallery_mean(self):
        if not self.deep_gallery:
            return None
        return np.mean(self.deep_gallery, axis=0)

class SimpleTracker:
    def __init__(self, max_age=10, iou_threshold=0.3, appearance_weight=0.3, deep_weight=0.7, gallery_frames=30):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        self.deep_weight = deep_weight
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
        self.gallery_frames = gallery_frames
        self.player_gallery = {}  # id: (mean_hist, mean_deep)

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o

    def appearance_distance(self, hist1, hist2):
        if hist1 is None or hist2 is None:
            return 1.0
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    def deep_distance(self, feat1, feat2):
        if feat1 is None or feat2 is None:
            return 1.0
        feat1 = feat1 / (np.linalg.norm(feat1) + 1e-6)
        feat2 = feat2 / (np.linalg.norm(feat2) + 1e-6)
        return 1.0 - np.dot(feat1, feat2)

    def associate_detections_to_tracks(self, detections, appearances, deep_features, tracks):
        if len(tracks) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        cost_matrix = np.zeros((len(detections), len(tracks)), dtype=np.float32)
        for d, (det, app, deep) in enumerate(zip(detections, appearances, deep_features)):
            for t, trk in enumerate(tracks):
                iou_score = self.iou(det, trk.bbox)
                app_dist = self.appearance_distance(app, trk.get_gallery_mean())
                deep_dist = self.deep_distance(deep, trk.get_deep_gallery_mean())
                cost = (1 - self.appearance_weight - self.deep_weight) * (1 - iou_score) \
                       + self.appearance_weight * app_dist \
                       + self.deep_weight * deep_dist
                cost_matrix[d, t] = cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches, unmatched_dets, unmatched_trks = [], [], []
        for d, det in enumerate(detections):
            if d not in row_ind:
                unmatched_dets.append(d)
        for t, trk in enumerate(tracks):
            if t not in col_ind:
                unmatched_trks.append(t)
        for r, c in zip(row_ind, col_ind):
            iou_score = self.iou(detections[r], tracks[c].bbox)
            app_dist = self.appearance_distance(appearances[r], tracks[c].get_gallery_mean())
            deep_dist = self.deep_distance(deep_features[r], tracks[c].get_deep_gallery_mean())
            if iou_score < self.iou_threshold and (app_dist > 0.5 or deep_dist > 0.5):
                unmatched_dets.append(r)
                unmatched_trks.append(c)
            else:
                matches.append((r, c))
        return matches, unmatched_dets, unmatched_trks

    def update(self, detections, appearances, deep_features):
        self.frame_count += 1
        if len(self.tracks) == 0:
            for i in range(len(detections)):
                self.tracks.append(Track(np.array(detections[i]), self.next_id, appearances[i], deep_features[i]))
                self.next_id += 1
            return [(trk.bbox, trk.id) for trk in self.tracks]
        for trk in self.tracks:
            trk.predict()
        matches, unmatched_dets, unmatched_trks = self.associate_detections_to_tracks(
            detections, appearances, deep_features, self.tracks)
        for m in matches:
            self.tracks[m[1]].update(np.array(detections[m[0]]), appearances[m[0]], deep_features[m[0]])
        for i in unmatched_dets:
            reid_id = None
            if self.frame_count > self.gallery_frames:
                min_dist = 1.0
                for pid, (gal_hist, gal_deep) in self.player_gallery.items():
                    app_dist = self.appearance_distance(appearances[i], gal_hist)
                    deep_dist = self.deep_distance(deep_features[i], gal_deep)
                    if app_dist < 0.4 and deep_dist < 0.4:
                        reid_id = pid
                        break
            if reid_id is not None:
                self.tracks.append(Track(np.array(detections[i]), reid_id, appearances[i], deep_features[i]))
            else:
                self.tracks.append(Track(np.array(detections[i]), self.next_id, appearances[i], deep_features[i]))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        if self.frame_count <= self.gallery_frames:
            for trk in self.tracks:
                if trk.id not in self.player_gallery and trk.get_gallery_mean() is not None and trk.get_deep_gallery_mean() is not None:
                    self.player_gallery[trk.id] = (trk.get_gallery_mean(), trk.get_deep_gallery_mean())
        return [(trk.bbox, trk.id) for trk in self.tracks]


def extract_histogram(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if crop.size == 0:
        return None
    hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist 