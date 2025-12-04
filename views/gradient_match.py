# views/gradient_match.py
from flask import Blueprint, render_template, request, send_file, jsonify, current_app
import cv2
import numpy as np
import io
import os
import base64

gradient_match_bp = Blueprint(
    "gradient_match", __name__, template_folder="../templates"
)


def direction_to_color(angle_rad_matrix: np.ndarray) -> np.ndarray:
    """Converts a gradient direction matrix to a color image (HSV -> BGR)."""
    hue = ((angle_rad_matrix + np.pi) * (180 / (2 * np.pi))).astype(np.uint8)
    saturation = np.full(angle_rad_matrix.shape, 255, dtype=np.uint8)
    value = np.full(angle_rad_matrix.shape, 255, dtype=np.uint8)
    hsv_image = cv2.merge([hue, saturation, value])
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return bgr_image


def process_image_gradients(img_gray: np.ndarray) -> np.ndarray:
    """
    Computes Gradient Magnitude, Gradient Direction, and Laplacian of Gaussian.
    Returns a horizontally concatenated image of all three.
    """
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    mag, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=False)

    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag_vis = cv2.applyColorMap(mag_norm.astype(np.uint8), cv2.COLORMAP_BONE)

    dir_vis = direction_to_color(angle)

    gaussian = cv2.GaussianBlur(img_gray, (5, 5), 0)
    laplacian = cv2.Laplacian(gaussian, cv2.CV_64F)

    laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    laplacian_vis = cv2.applyColorMap(laplacian_norm.astype(np.uint8), cv2.COLORMAP_JET)

    combined = cv2.hconcat([mag_vis, dir_vis, laplacian_vis])

    separator = np.zeros((combined.shape[0], 5, 3), dtype=np.uint8)
    combined_sep = cv2.hconcat([mag_vis, separator, dir_vis, separator, laplacian_vis])

    return combined_sep


@gradient_match_bp.route("/gradient_match", methods=["GET", "POST"])
def gradient_match():
    if request.method == "POST":
        try:
            main_image_file = request.files.get("mainImage")

            if not main_image_file or not template_image_file:
                return jsonify(
                    {"error": "Please upload both a main image and a template image."}
                ), 400

            main_img_bytes = main_image_file.read()
            template_img_bytes = template_image_file.read()

            main_img_color = cv2.imdecode(
                np.frombuffer(main_img_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            h, w = main_img_color.shape[:-1]

            main_img_gray = cv2.cvtColor(main_img_color, cv2.COLOR_BGR2GRAY)

            result_img = process_image_gradients(main_img_gray)

            _, img_encoded = cv2.imencode(".png", result_img)
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype="image/png")

        except Exception as e:
            return jsonify({"error": f"An error occurred on the server: {str(e)}"}), 500

    return render_template("gradient_match.html")


@gradient_match_bp.route("/gradient_match/demo", methods=["POST"])
def gradient_demo():
    """Runs the gradient/LoG procedure on all files in static/perspective."""
    try:
        # Define path to static/perspective
        # Assuming app structure: /app/views/.. -> /app/static/perspective
        # Using os.getcwd() often points to root in these environments.
        demo_dir = os.path.join(os.getcwd(), "static", "perspective")

        if not os.path.exists(demo_dir):
            return jsonify({"error": f"Directory not found: {demo_dir}"}), 404

        files = [
            f
            for f in os.listdir(demo_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if not files:
            return jsonify({"error": "No images found in static/perspective."}), 404

        results = []

        for filename in files:
            filepath = os.path.join(demo_dir, filename)
            img = cv2.imread(filepath)

            if img is None:
                continue

            h, w = img.shape[:2]
            if w > 500:
                scale = 500 / w
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed_vis = process_image_gradients(gray)
            _, buffer = cv2.imencode(".png", processed_vis)
            b64_str = base64.b64encode(buffer).decode("utf-8")

            results.append({"filename": filename, "image": b64_str})

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
