from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path="../models/best.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_players(self, frame):
        results = self.model(frame, conf=self.conf_threshold)
        
        return results 