from flask import Blueprint, render_template, request
import cv2
import numpy as np
import io
import base64
import math
import os

segment_bp = Blueprint("segment", __name__, template_folder="../templates")


def encode_image(img):
    """Encodes an OpenCV image to a base64 string."""
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def order_points_angular(points):
    """
    Orders a list of (x, y) points based on the angle relative to their centroid.
    """
    if len(points) < 3:
        return points

    pts = np.array(points)
    center = np.mean(pts, axis=0)
    cx, cy = center[0], center[1]

    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    sorted_indices = np.argsort(angles)
    sorted_points = pts[sorted_indices]

    return sorted_points


def smart_segmentation(img, markers_poly):
    """
    Uses GrabCut to snap the boundary to edges between markers.
    """
    height, width = img.shape[:2]
    max_dim = 1000
    scale = 1.0

    if max(height, width) > max_dim:
        scale = max_dim / max(height, width)
        img_small = cv2.resize(img, None, fx=scale, fy=scale)
        poly_small = (markers_poly * scale).astype(np.int32)
    else:
        img_small = img.copy()
        poly_small = markers_poly.astype(np.int32)

    mask = np.zeros(img_small.shape[:2], np.uint8)

    cv2.fillPoly(mask, [poly_small], cv2.GC_PR_FGD)

    for pt in poly_small:
        cv2.circle(mask, tuple(pt), 5, cv2.GC_FGD, -1)

    centroid = np.mean(poly_small, axis=0).astype(int)
    cv2.circle(mask, tuple(centroid), 10, cv2.GC_FGD, -1)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_small, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        print(f"GrabCut failed: {e}")
        return markers_poly.astype(np.int32)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return markers_poly.astype(np.int32)

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    final_contour = (approx_contour / scale).astype(np.int32)

    return final_contour


def detect_and_segment(img):
    """
    Main pipeline: Detect ArUco -> Order Points -> GrabCut Refinement -> Crop
    """
    vis_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()

    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )

    if not corners:
        return None, None, "No ArUco markers detected."

    if len(corners) < 3:
        cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
        return vis_img, None, f"Found {len(corners)} markers. Need at least 3."

    marker_centers = []
    for corner_set in corners:
        c = corner_set[0]
        center_x = int(np.mean(c[:, 0]))
        center_y = int(np.mean(c[:, 1]))
        marker_centers.append([center_x, center_y])

    ordered_poly_float = order_points_angular(marker_centers)

    # Use GrabCut to find exact boundary
    final_contour = smart_segmentation(img, ordered_poly_float)

    # Visualization
    cv2.aruco.drawDetectedMarkers(vis_img, corners, ids)
    cv2.polylines(
        vis_img,
        [ordered_poly_float.astype(np.int32)],
        True,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.drawContours(vis_img, [final_contour], -1, (0, 255, 0), 3)

    # Create Segmentation Mask & Crop
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [final_contour], -1, 255, -1)

    segmented_img = cv2.bitwise_and(img, img, mask=mask)

    x, y, w, h = cv2.boundingRect(final_contour)
    pad = 20
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)

    cropped_result = segmented_img[y1:y2, x1:x2]

    return vis_img, cropped_result, None


def run_benchmark():
    """
    Loads images from static/aruco_marked_seg, processes them,
    and finds matching SAM results in static/sam_seg.
    """
    base_static_path = os.path.join(os.path.dirname(__file__), "..", "static")
    aruco_dir = os.path.join(base_static_path, "aruco_marked_seg")
    sam_dir = os.path.join(base_static_path, "sam2_seg")

    results = []

    if not os.path.exists(aruco_dir):
        return [{"filename": "Error", "error": f"Directory not found: {aruco_dir}"}]

    # List all images
    files = sorted(
        [
            f
            for f in os.listdir(aruco_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    if not files:
        return [{"filename": "Error", "error": f"No images found in {aruco_dir}"}]

    for fname in files:
        # Load ArUco Image
        aruco_path = os.path.join(aruco_dir, fname)
        img = cv2.imread(aruco_path)

        if img is None:
            continue

        # Process using our algorithm
        vis, seg, error = detect_and_segment(img)

        res_entry = {"filename": fname, "error": error, "is_benchmark": True}

        if vis is not None:
            res_entry["vis_b64"] = encode_image(vis)
        if seg is not None:
            res_entry["seg_b64"] = encode_image(seg)

        # Look for matching SAM image
        sam_path = os.path.join(sam_dir, fname.rstrip("jpg") + "png")
        if os.path.exists(sam_path):
            sam_img = cv2.imread(sam_path)
            if sam_img is not None:
                res_entry["sam_b64"] = encode_image(sam_img)
        else:
            print("NOT FOUND SAM")

        results.append(res_entry)

    return results


@segment_bp.route("/segment", methods=["GET", "POST"])
def segment():
    if request.method == "POST":
        # Check if this is a benchmark request
        if request.form.get("action") == "benchmark":
            results = run_benchmark()
            return render_template("segment.html", results=results)

        # Standard File Upload
        uploaded_files = request.files.getlist("images")
        results = []

        if not uploaded_files or uploaded_files[0].filename == "":
            return render_template("segment.html", error="No files selected.")

        for file in uploaded_files:
            in_memory_file = io.BytesIO(file.read())
            file_bytes = np.frombuffer(in_memory_file.getvalue(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                continue

            vis, seg, error = detect_and_segment(img)

            res_entry = {"filename": file.filename, "error": error}
            if vis is not None:
                res_entry["vis_b64"] = encode_image(vis)
            if seg is not None:
                res_entry["seg_b64"] = encode_image(seg)

            results.append(res_entry)

        return render_template("segment.html", results=results)

    return render_template("segment.html")

