from flask import Blueprint, render_template, current_app, jsonify
import cv2
import numpy as np
import os

tracker_bp = Blueprint(
    "tracker", __name__, template_folder="../templates", static_folder="../static"
)


@tracker_bp.route("/tracker")
def index():
    return render_template("tracker.html")


@tracker_bp.route("/tracker/sam2_data")
def sam2_data():
    """
    Parses the NPZ segmentation data and returns it as a JSON object
    containing simplified contours for each frame.
    """
    static_root = current_app.static_folder
    # Adjust paths based on your actual folder structure
    video_path = os.path.join(static_root, "tracking_sam2", "in24.mp4")
    npz_path = os.path.join(static_root, "tracking_sam2", "segmentation_data.npz")

    if not os.path.exists(npz_path):
        return jsonify({"error": "Segmentation data not found"}), 404

    try:
        # Load NPZ file
        data = np.load(npz_path)

        # Initialize output structure
        output = {
            "fps": 30.0,  # Default
            "frames": {},
        }

        # Attempt to get actual FPS from the video file
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                output["fps"] = cap.get(cv2.CAP_PROP_FPS)
                cap.release()

        # Process each frame in the NPZ
        # Filter for keys starting with "frame_"
        keys = [k for k in data.files if k.startswith("frame_")]

        for key in keys:
            frame_idx = int(key.split("_")[1])
            mask = data[key]

            # Squeeze dimensions if necessary (e.g., (1, H, W) -> (H, W))
            if len(mask.shape) > 2:
                mask = mask.squeeze()

            # Convert boolean mask to uint8 for findContours
            binary_mask = mask.astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Convert numpy contours to simple list of points [[x,y], [x,y]...]
            frame_contours = []
            for cnt in contours:
                # cnt is shape (N, 1, 2), reshape to (N, 2) and convert to list
                points = cnt.reshape(-1, 2).tolist()
                frame_contours.append(points)

            if frame_contours:
                output["frames"][frame_idx] = frame_contours

        return jsonify(output)

    except Exception as e:
        print(f"Error processing SAM2 data: {e}")
        return jsonify({"error": str(e)}), 500
