from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
import io
import base64
import os
import uuid
import shutil

segment_select_bp = Blueprint(
    "segment_select", __name__, template_folder="../templates"
)

# Reuse the temp directory structure from the other modules
TEMP_DIR = os.path.join(os.getcwd(), "temp_data")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)


def encode_image(img):
    """Encodes an OpenCV image to a base64 string."""
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def save_temp_image(img, session_id):
    """Saves the uploaded image to a session folder."""
    session_dir = os.path.join(TEMP_DIR, session_id)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)

    # We only store one active image for this tool for simplicity
    filepath = os.path.join(session_dir, "source.png")
    cv2.imwrite(filepath, img)
    return filepath


def get_temp_image(session_id):
    filepath = os.path.join(TEMP_DIR, session_id, "source.png")
    if os.path.exists(filepath):
        return cv2.imread(filepath)
    return None


def apply_sift_grabcut(img, click_x, click_y):
    """
    1. Detects SIFT keypoints.
    2. Uses Convex Hull of points to create a 'Probable Foreground' blob.
    3. Defines a large 'Probable Background' area to allow expansion.
    4. Runs GrabCut.
    """
    # 1. Initialize the ENTIRE mask as Sure Background (0).
    # We will carve out allowed regions from here.
    mask = np.full(img.shape[:2], cv2.GC_BGD, dtype=np.uint8)

    # Detect SIFT features
    sift = cv2.SIFT_create()
    keypoints = sift.detect(img, None)

    # Parameters
    search_radius = 150  # Pixels around click to look for SIFT points

    # Collect all points that will be part of our foreground "cloud"
    active_points = []

    # Always include the click itself
    active_points.append((int(click_x), int(click_y)))

    # Find SIFT points near the click
    for kp in keypoints:
        kx, ky = kp.pt
        dist = np.sqrt((kx - click_x) ** 2 + (ky - click_y) ** 2)

        if dist < search_radius:
            active_points.append((int(kx), int(ky)))

    pts = np.array(active_points, dtype=np.int32)

    # 2. Define a "Search Area" (Probable Background)
    # Instead of a tight box, we expand significantly (e.g. +150px or 2x size)
    # This addresses "cuts out too much" by giving GrabCut room to grow.
    x, y, w, h = cv2.boundingRect(pts)
    h_img, w_img = img.shape[:2]

    # Generous padding (flexible expansion)
    pad_w = max(100, int(w * 0.8))
    pad_h = max(100, int(h * 0.8))

    roi_x1 = max(0, x - pad_w)
    roi_y1 = max(0, y - pad_h)
    roi_x2 = min(w_img, x + w + pad_w)
    roi_y2 = min(h_img, y + h + pad_h)

    # Set the expanded search area to Probable Background (2)
    mask[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.GC_PR_BGD

    # 3. Create a "Shape Hint" (Probable Foreground)
    # Use the Convex Hull of the SIFT points.
    # This creates a solid polygon connecting the outer dots, giving
    # GrabCut a "good starting mask" that looks like a blob.
    if len(pts) > 2:
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, cv2.GC_PR_FGD)
    else:
        # Fallback if only 1-2 points found (draw a PR_FGD circle)
        for pt in pts:
            cv2.circle(mask, tuple(pt), 20, cv2.GC_PR_FGD, -1)

    # 4. Mark the specific SIFT points as Sure Foreground (1)
    # These are the hard anchors.
    for pt in active_points:
        # Draw small sure-fg circles at exact feature locations
        cv2.circle(mask, tuple(pt), 4, cv2.GC_FGD, -1)

    # Mark the exact click location strongly
    cv2.circle(mask, (int(click_x), int(click_y)), 6, cv2.GC_FGD, -1)

    # Visualization of the SIFT Mask (for debugging/display)
    # 0=Black (Sure BG), 1=White (Sure FG), 2=DarkGray (Prob BG), 3=LightGray (Prob FG)
    debug_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    debug_mask[mask == cv2.GC_BGD] = [0, 0, 0]  # Black
    debug_mask[mask == cv2.GC_PR_BGD] = [64, 64, 64]  # Dark Gray
    debug_mask[mask == cv2.GC_PR_FGD] = [160, 160, 160]  # Light Gray (Hull)
    debug_mask[mask == cv2.GC_FGD] = [255, 255, 255]  # White (Seeds)

    # --- Run GrabCut ---
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        # iterCount=5
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        print(f"GrabCut Error: {e}")
        return img, debug_mask, str(e)

    # Extract result
    # In the final mask:
    # 0 (GC_BGD) and 2 (GC_PR_BGD) -> Background
    # 1 (GC_FGD) and 3 (GC_PR_FGD) -> Foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # Segment the image
    segmented = img * mask2[:, :, np.newaxis]

    # Find bounding box to crop (optional, keeps it clean)
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        segmented = segmented[y1:y2, x1:x2]

    return segmented, debug_mask, None


def resize_image_if_needed(image, max_dim):
    """Resizes actual image data if it's too massive."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized_image


@segment_select_bp.route("/segment_select", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle Image Upload
        if "image" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Create Session
        session_id = str(uuid.uuid4())

        in_memory_file = io.BytesIO(file.read())
        file_bytes = np.frombuffer(in_memory_file.getvalue(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = resize_image_if_needed(img, 1920)

        # Save to disk
        save_temp_image(img, session_id)

        # Return base64 for canvas
        img_b64 = encode_image(img)
        return jsonify(
            {
                "session_id": session_id,
                "image_b64": img_b64,
                "width": img.shape[1],
                "height": img.shape[0],
            }
        )

    return render_template("segment_select.html")


@segment_select_bp.route("/segment_select/process", methods=["POST"])
def process():
    data = request.json
    session_id = data.get("session_id")
    click_x = data.get("x")
    click_y = data.get("y")

    if not session_id or click_x is None:
        return jsonify({"error": "Missing parameters"}), 400

    img = get_temp_image(session_id)
    if img is None:
        return jsonify({"error": "Session expired"}), 404

    # Perform Segmentation
    segmented_img, mask_debug, error = apply_sift_grabcut(
        img, float(click_x), float(click_y)
    )

    if error:
        return jsonify({"error": error}), 500

    return jsonify(
        {
            "segmented_b64": encode_image(segmented_img),
            "mask_b64": encode_image(mask_debug),
        }
    )
