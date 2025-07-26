from cpu_player_tracker import run_cpu_tracking

run_cpu_tracking(
    "15sec_input_720p.mp4",
    model_path="models/best.pt",
    output_path="output/cpu_tracked_output.mp4",
    frame_skip=1,
    results_csv="output/tracker_results.csv"
)
