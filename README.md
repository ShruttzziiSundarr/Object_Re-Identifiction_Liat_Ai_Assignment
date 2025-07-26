# Multi-Modal Player Tracking and Re-Identification System

A comprehensive comparison framework for player tracking and re-identification using different state-of-the-art approaches including enhanced MULTI MODAL SORT, FairMOT, and DeepSORT-style implementations.

##  Project Overview

This project implements and compares three different approaches to player tracking and re-identification in video sequences:

1. **Multi-Modal Enhanced SORT**: CPU-based tracker with comprehensive appearance features
2. **FairMOT**: Joint detection and re-identification deep learning approach  
3. **DeepSORT-Style**: Classic DeepSORT implementation with deep features and appearance modeling

## Project Structure

```
├── Multi-Modal based appearance enhanced SORT style object Tracker/
│   ├── src/
│   │   ├── cpu_player_tracker.py          # Main CPU tracker implementation
│   │   └── run_cpu_tracker.py             # Script to run CPU tracker
│
├── FairMOT/
│   ├── src/
│   │   ├── fairmot_tracker.py             # Full FairMOT implementation
│   │   ├── run_fairmot_tracker.py         # FairMOT runner with analysis
│   │   └── simple_fairmot_demo.py         # Simplified FairMOT demo
│
├── DeepSORT_STYLE_RE-IDENTIFICATION/
│   ├── src/
│   │   ├── detector.py                    # YOLO-based detector
│   │   ├── simple_tracker.py              # DeepSORT-style tracker
│   │   ├── video_processor.py             # Video I/O utilities
│   │   ├── run_pipeline.py                # Main pipeline runner
│   │   └── __init__.py
│
├── models/
│   └── best.pt                            # YOLO model weights
│
├── output/                                # Output directory for results
└── 15sec_input_720p.mp4                  # Sample input video
```

##  Requirements

### Core Dependencies
```bash
pip install ultralytics opencv-python numpy scipy scikit-learn
pip install torch torchvision matplotlib filterpy
pip install papaparse sheetjs
```

### System Requirements
- Python 3.7+
- CUDA-compatible GPU (optional, for FairMOT)
- 8GB+ RAM recommended
- OpenCV with video codec support



### 1. Multi-Modal Enhanced SORT Tracker

The CPU-based tracker uses comprehensive appearance features including color histograms, texture patterns (LBP), shape features, and spatial information.

```python
from cpu_player_tracker import run_cpu_tracking

# Run CPU tracker
run_cpu_tracking(
    video_path="15sec_input_720p.mp4",
    model_path="models/best.pt",
    output_path="output/cpu_tracked_output.mp4",
    frame_skip=1,
    results_csv="output/tracker_results.csv"
)
```

**Features:**
-  CPU-optimized performance
-  Multi-modal feature extraction (color, texture, shape, position)
-  Local Binary Pattern (LBP) texture analysis
-  Cosine similarity matching
-  Hungarian algorithm for optimal assignment

### 2. FairMOT Implementation

Joint detection and re-identification using deep learning with DLA-34 backbone.

```python
from run_fairmot_tracker import run_fairmot_tracking

# Run FairMOT tracker
tracker = run_fairmot_tracking(
    video_path="15sec_input_720p.mp4",
    output_path="fairmot_output.mp4",
    conf_thresh=0.4,
    reid_thresh=0.6,
    max_disappeared=30,
    show_video=False
)
```

**Features:**
-  Joint detection and ReID optimization
-  DLA-34 backbone with ReID head
-  Real-time performance
-  Robust re-identification across occlusions
-  Comprehensive performance analytics

### 3. DeepSORT-Style Tracker

Classic DeepSORT approach with ResNet18 feature extraction and Kalman filtering.

```python
from run_pipeline import *

# Run DeepSORT-style tracker
python run_pipeline.py
```

**Features:**
-  ResNet18-based deep feature extraction
-  Kalman filter motion prediction
-  Gallery-based re-identification
-  Appearance and deep feature fusion
-  IoU-based data association

##  Feature Comparison

| Feature | Enhanced SORT | FairMOT | DeepSORT-Style |
|---------|---------------|---------|----------------|
| **Detection** | YOLO | Joint Detection along with YOLO | YOLO |
| **Re-ID Method** | Multi-modal features | Deep ReID head | ResNet18 features |
| **Motion Model** | Simple prediction | Velocity-based | Kalman Filter |
| **Performance** | CPU-optimized | GPU-accelerated | Balanced |
| **Robustness** | Medium | High | Medium-High |
| **Memory Usage** | Low | Medium-High | Medium |

##  Usage Examples

### Basic Tracking
```python
# For quick CPU-based tracking
from cpu_player_tracker import run_cpu_tracking
run_cpu_tracking("input_video.mp4", output_path="output.mp4")
```

### Advanced FairMOT with Analysis
```python
# For research and detailed analysis
from run_fairmot_tracker import run_fairmot_tracking, analyze_tracking_results

tracker = run_fairmot_tracking(
    video_path="input.mp4",
    conf_thresh=0.4,
    reid_thresh=0.6,
    save_results=True,
    show_video=True
)

# Analyze results
analysis = analyze_tracking_results("fairmot_output_results.json")
```

### Custom Configuration
```python
# Custom tracker configuration
tracker = PlayerFairMOTTracker(
    conf_thresh=0.3,        # Lower for more detections
    reid_thresh=0.7,        # Higher for stricter matching
    max_disappeared=50      # Keep tracks longer
)
```

##  Performance Metrics

All implementations provide comprehensive performance analytics:

- **Processing Speed (FPS)**
- **Track Persistence Analysis**
- **Re-identification Accuracy**
- **Memory Usage Statistics**
- **Detection Confidence Distributions**

### Sample Output
```
FairMOT Performance Summary:
   • Successfully tracked 12 unique players
   • Average 8.5 players per frame
   • Longest track persisted for 450 frames
   • Average processing time: 23.4ms per frame
   • Average FPS: 42.7
```

##  Configuration Options

### CPU Tracker Parameters
```python
tracker = CPUPlayerTracker()
tracker.max_disappeared = 8           # Frames before track deletion
tracker.similarity_threshold = 0.6    # Feature matching threshold
```

### FairMOT Parameters
```python
tracker = PlayerFairMOTTracker(
    conf_thresh=0.4,        # Detection confidence
    reid_thresh=0.6,        # ReID similarity threshold
    max_disappeared=30      # Track persistence
)
```

### DeepSORT Parameters
```python
tracker = SimpleTracker(
    max_age=10,             # Track lifetime
    iou_threshold=0.3,      # IoU matching threshold
    appearance_weight=0.3,  # Appearance feature weight
    deep_weight=0.7,        # Deep feature weight
    gallery_frames=30       # Gallery building period
)
```

##  Output Files

Each tracker generates different output formats:

### CPU Tracker
- `cpu_tracked_output.mp4` - Annotated video
- `tracker_results.csv` - Frame-by-frame tracking data

### FairMOT
- `fairmot_output.mp4` - Annotated video with trajectories
- `fairmot_output_results.json` - Detailed tracking results
- `fairmot_output_metrics.json` - Performance metrics

### DeepSORT
- `annotated_video.mp4` - Tracked video output

##  Visualization Features

- **Unique Color Coding**: Each tracked player gets a consistent color
- **Trajectory Visualization**: Historical path display
- **Real-time Statistics**: FPS, active players, track count
- **Confidence Indicators**: Visual confidence scoring
- **Re-identification Highlights**: Track recovery visualization

##  Troubleshooting

### Common Issues

1. **YOLO Model Loading Error**
   ```bash
   # Ensure model file exists
   ls models/best.pt
   # Download if missing from ultralytics model zoo
   ```

2. **Video Codec Issues**
   ```bash
   # Install additional codecs
   pip install opencv-python-headless
   ```

3. **CUDA Out of Memory (FairMOT)**
   ```python
   # Reduce batch size or use CPU mode
   model.to('cpu')
   ```

4. **Low FPS Performance**
   ```python
   # Use frame skipping
   run_cpu_tracking(frame_skip=2)  # Process every 2nd frame
   ```



##  Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-tracker`)
3. Commit changes (`git commit -am 'Add new tracker'`)
4. Push to branch (`git push origin feature/new-tracker`)
5. Create Pull Request



##  Acknowledgments

- **YOLO**: Object detection framework by Ultralytics
- **FairMOT**: Joint detection and re-identification approach
- **DeepSORT**: Multiple object tracking with deep association
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework

---

