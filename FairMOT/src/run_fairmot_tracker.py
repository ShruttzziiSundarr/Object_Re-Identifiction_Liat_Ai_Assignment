import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from fairmot_tracker import PlayerFairMOTTracker
import argparse

def create_color_palette(n_colors=20):
    """Create a consistent color palette for player IDs"""
    colors = []
    for i in range(n_colors):
        #different colors for all player
        hue = (i * 137.508) % 360  
        rgb = plt.cm.hsv(hue / 360.0)[:3]
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def visualize_tracks(frame, tracks, color_palette, frame_info=None):
    """Visualize tracking results on frame"""
    annotated_frame = frame.copy()
    
    
    for track in tracks:
        x1, y1, x2, y2 = track['bbox']
        track_id = track['id']
        confidence = track['confidence']
        
        color = color_palette[track_id % len(color_palette)]
        
        
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        
        label = f"Player {track_id} ({confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        
        cv2.rectangle(annotated_frame, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        
        center = track['center']
        cv2.circle(annotated_frame, (int(center[0]), int(center[1])), 3, color, -1)
        
        
        if 'positions_history' in track:
            positions = track['positions_history']
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                    pt2 = (int(positions[i][0]), int(positions[i][1]))
                    cv2.line(annotated_frame, pt1, pt2, color, 2)
    
    
    if frame_info:
        cv2.putText(annotated_frame, f"Frame: {frame_info['frame_count']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Active Players: {len(tracks)}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"FPS: {frame_info['fps']:.1f}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return annotated_frame

def run_fairmot_tracking(video_path="15sec_input_720p.mp4", 
                        output_path="fairmot_output.mp4",
                        conf_thresh=0.4, 
                        reid_thresh=0.6,
                        max_disappeared=30,
                        save_results=True,
                        show_video=False):
    """Run FairMOT tracking on video"""
    
    print(" Initializing FairMOT Player Re-Identification System...")
    print(f" Input video: {video_path}")
    print(f" Confidence threshold: {conf_thresh}")
    print(f" ReID threshold: {reid_thresh}")
    print(f" Max disappeared frames: {max_disappeared}")
    
    
    tracker = PlayerFairMOTTracker(
        conf_thresh=conf_thresh, 
        reid_thresh=reid_thresh,
        max_disappeared=max_disappeared
    )
    
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Error: Could not open video {video_path}")
        return
    
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f" Video properties: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    
   
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    color_palette = create_color_palette(20)

    frame_count = 0
    processing_times = []
    total_tracks_created = 0
    
    print("\n Starting FairMOT tracking...")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        
        tracks = tracker.process_frame(frame)
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        
        
        frame_info = {
            'frame_count': frame_count,
            'fps': current_fps,
            'processing_time': processing_time
        }
        
        
        annotated_frame = visualize_tracks(frame, tracks, color_palette, frame_info)
        
        
        out.write(annotated_frame)
        
        
        if show_video:
            cv2.imshow('FairMOT Player Tracking', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        
        if frame_count % 30 == 0:
            avg_fps = 1.0 / np.mean(processing_times[-30:]) if len(processing_times) >= 30 else 0
            print(f" Frame {frame_count}/{total_frames} | "
                  f"Active Players: {len(tracks)} | "
                  f"Avg FPS: {avg_fps:.1f} | "
                  f"Total Tracks: {tracker.next_id - 1}")
        
        frame_count += 1
    
    
    cap.release()
    out.release()
    if show_video:
        cv2.destroyAllWindows()
    avg_fps = 1.0 / np.mean(processing_times) if processing_times else 0
    total_time = sum(processing_times)
    
    print("\n" + "=" * 60)
    print("FairMOT Tracking Complete!")
    print("=" * 60)
    print(f" Performance Statistics:")
    print(f"   • Total frames processed: {frame_count}")
    print(f"   • Average FPS: {avg_fps:.1f}")
    print(f"   • Total processing time: {total_time:.2f}s")
    print(f"   • Total tracks created: {tracker.next_id - 1}")
    print(f"   • Final active tracks: {len(tracker.tracks)}")
    print(f"   • Disappeared tracks: {len(tracker.disappeared_tracks)}")
    print(f"Output video: {output_path}")
    
    
    if save_results:
        results_file = output_path.replace('.mp4', '_results.json')
        tracker.save_results(results_file)
        
        
        metrics = {
            'total_frames': frame_count,
            'avg_fps': avg_fps,
            'total_processing_time': total_time,
            'total_tracks_created': tracker.next_id - 1,
            'final_active_tracks': len(tracker.tracks),
            'disappeared_tracks': len(tracker.disappeared_tracks),
            'avg_processing_time_per_frame': np.mean(processing_times),
            'conf_thresh': conf_thresh,
            'reid_thresh': reid_thresh,
            'max_disappeared': max_disappeared
        }
        
        metrics_file = output_path.replace('.mp4', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Results saved: {results_file}")
        print(f" Metrics saved: {metrics_file}")
    
    return tracker

def analyze_tracking_results(results_file):
    """Analyze tracking results and generate insights"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    frames = results['frames']
    total_frames = len(frames)
    
    
    track_counts = [len(frame['tracks']) for frame in frames]
    processing_times = [frame['processing_time'] for frame in frames]
    
    
    track_ids = set()
    track_persistence = {}
    
    for frame in frames:
        for track in frame['tracks']:
            track_id = track['id']
            track_ids.add(track_id)
            
            if track_id not in track_persistence:
                track_persistence[track_id] = {'first_seen': frame['frame_id'], 'last_seen': frame['frame_id']}
            else:
                track_persistence[track_id]['last_seen'] = frame['frame_id']
    
    
    persistence_lengths = []
    for track_data in track_persistence.values():
        persistence = track_data['last_seen'] - track_data['first_seen'] + 1
        persistence_lengths.append(persistence)
    
    print("\nTracking Analysis:")
    print(f"   • Total unique players tracked: {len(track_ids)}")
    print(f"   • Average players per frame: {np.mean(track_counts):.1f}")
    print(f"   • Max players in single frame: {max(track_counts)}")
    print(f"   • Average track persistence: {np.mean(persistence_lengths):.1f} frames")
    print(f"   • Longest track persistence: {max(persistence_lengths)} frames")
    print(f"   • Average processing time: {np.mean(processing_times)*1000:.1f}ms")
    
    return {
        'total_players': len(track_ids),
        'avg_players_per_frame': np.mean(track_counts),
        'max_players': max(track_counts),
        'avg_persistence': np.mean(persistence_lengths),
        'max_persistence': max(persistence_lengths),
        'avg_processing_time': np.mean(processing_times)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FairMOT Player Re-Identification")
    parser.add_argument("--video", default="15sec_input_720p.mp4", help="Input video path")
    parser.add_argument("--output", default="fairmot_output.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--reid", type=float, default=0.6, help="ReID similarity threshold")
    parser.add_argument("--max-disappeared", type=int, default=30, help="Max frames before track removal")
    parser.add_argument("--show", action="store_true", help="Show video during processing")
    parser.add_argument("--analyze", action="store_true", help="Analyze results after processing")
    
    args = parser.parse_args()
    
    tracker = run_fairmot_tracking(
        video_path=args.video,
        output_path=args.output,
        conf_thresh=args.conf,
        reid_thresh=args.reid,
        max_disappeared=args.max_disappeared,
        show_video=args.show
    )
    
    if args.analyze:
        results_file = args.output.replace('.mp4', '_results.json')
        try:
            analysis = analyze_tracking_results(results_file)
            print(f"\n FairMOT Performance Summary:")
            print(f"   • Successfully tracked {analysis['total_players']} unique players")
            print(f"   • Average {analysis['avg_players_per_frame']:.1f} players per frame")
            print(f"   • Longest track persisted for {analysis['max_persistence']} frames")
            print(f"   • Average processing time: {analysis['avg_processing_time']*1000:.1f}ms per frame")
        except FileNotFoundError:
            print(" Results file not found for analysis") 