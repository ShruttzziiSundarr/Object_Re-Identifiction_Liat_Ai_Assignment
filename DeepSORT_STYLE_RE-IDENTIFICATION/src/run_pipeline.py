from detector import PlayerDetector
from video_processor import extract_frames, write_video
from simple_tracker import SimpleTracker, extract_histogram, extract_deep_feature
import cv2
import numpy as np

if __name__ == "__main__":
    video_path = "15sec_input_720p.mp4"  
    output_path = "output/annotated_video.mp4"
    detector = PlayerDetector()
    tracker = SimpleTracker(max_age=10, iou_threshold=0.3, appearance_weight=0.3, deep_weight=0.7, gallery_frames=30)

    
    frames = extract_frames(video_path)
    annotated_frames = []

    for frame in frames:
        results = detector.detect_players(frame)
        detections = []
        appearances = []
        deep_features = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
            for box in boxes:
                detections.append(box)
                appearances.append(extract_histogram(frame, box))
                deep_features.append(extract_deep_feature(frame, box))
        
        tracked = tracker.update(detections, appearances, deep_features)
        
        for bbox, track_id in tracked:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        annotated_frames.append(frame)

    
    write_video(annotated_frames, output_path, fps=30)
    print(f"Annotated video with deep and color re-ID saved to {output_path}") 