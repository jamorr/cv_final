from flask import Blueprint, render_template, request
import cv2
import numpy as np
import io
import base64

edges_corners_bp = Blueprint("edges_corners", __name__, template_folder="../templates")


def encode_image(img):
    """Encodes an OpenCV image to a base64 string."""
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def detect_edges(img):
    """
    Implements Canny Edge Detection (Classic CV).
    1. Converts to Grayscale.
    2. Applies Gaussian Blur (noise reduction).
    3. Finds intensity gradients using Canny.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 5x5 Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny Edge Detection
    # Threshold 1: Lower bound for hysteresis
    # Threshold 2: Upper bound for hysteresis
    edges = cv2.Canny(blurred, 50, 150)

    return edges


def detect_corners(img):
    """
    Implements Harris Corner Detection (Classic CV).
    1. Converts to Grayscale (Float32 required).
    2. Computes Harris response R.
    3. Thresholds R to find corners.
    4. Marks corners on the original image.
    """
    vis_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Harris requires float32 input
    gray = np.float32(gray)

    # cv2.cornerHarris(img, blockSize, ksize, k)
    # blockSize: Neighborhood size
    # ksize: Aperture parameter for Sobel
    # k: Harris detector free parameter
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Result is dilated for marking the corners, not important for the actual detection logic
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    # We define a corner as a pixel with a response > 1% of the max response.
    vis_img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark red

    return vis_img


@edges_corners_bp.route("/edges_corners", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return render_template("edges_corners.html", error="No file selected.")

        in_memory_file = io.BytesIO(file.read())
        file_bytes = np.frombuffer(in_memory_file.getvalue(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return render_template("edges_corners.html", error="Invalid image file.")

        # Run Algorithms
        edges = detect_edges(img)
        corners = detect_corners(img)

        return render_template(
            "edges_corners.html",
            original_b64=encode_image(img),
            edges_b64=encode_image(edges),
            corners_b64=encode_image(corners),
        )

    return render_template("edges_corners.html")
