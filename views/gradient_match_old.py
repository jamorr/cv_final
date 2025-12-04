# views/gradient_match.py
from flask import Blueprint, render_template, request, send_file, jsonify
import cv2
import numpy as np
import io

gradient_match_bp = Blueprint(
    "gradient_match", __name__, template_folder="../templates"
)


def direction_to_color(angle_rad_matrix: np.ndarray) -> np.ndarray:
    hue = ((angle_rad_matrix + np.pi) * (180 / (2 * np.pi))).astype(np.uint8)
    saturation = np.full(angle_rad_matrix.shape, 255, dtype=np.uint8)
    value = np.full(angle_rad_matrix.shape, 255, dtype=np.uint8)
    hsv_image = cv2.merge([hue, saturation, value])
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return bgr_image


@gradient_match_bp.route("/gradient_match", methods=["GET", "POST"])
def gradient_match():
    if request.method == "POST":
        try:
            main_image_file = request.files.get("mainImage")
            template_image_file = request.files.get("templateImage")

            if not main_image_file or not template_image_file:
                return jsonify(
                    {"error": "Please upload both a main image and a template image."}
                ), 400

            main_img_bytes = main_image_file.read()
            template_img_bytes = template_image_file.read()
            main_img_gray = cv2.imdecode(
                np.frombuffer(main_img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
            )
            template_img_gray = cv2.imdecode(
                np.frombuffer(template_img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE
            )

            if main_img_gray is None or template_img_gray is None:
                return jsonify({"error": "Could not decode images."}), 400

            # --- Calculate Gradients for Main Image and Template ---
            main_sobel_x = cv2.Sobel(main_img_gray, cv2.CV_64F, 1, 0, ksize=3)
            main_sobel_y = cv2.Sobel(main_img_gray, cv2.CV_64F, 0, 1, ksize=3)
            main_mag, main_dir = cv2.cartToPolar(
                main_sobel_x, main_sobel_y, angleInDegrees=False
            )

            template_sobel_x = cv2.Sobel(template_img_gray, cv2.CV_64F, 1, 0, ksize=3)
            template_sobel_y = cv2.Sobel(template_img_gray, cv2.CV_64F, 0, 1, ksize=3)
            template_mag, template_dir = cv2.cartToPolar(
                template_sobel_x, template_sobel_y, angleInDegrees=False
            )

            # --- Gradient-based Template Matching ---
            # This is complex. A simple approach is to match magnitudes.
            # Your original code matched colors, then computed gradients on the *correlation map*,
            # which is not standard gradient matching.
            #
            # Let's stick to your original implementation's flow:
            # 1. Correlate color images (decoding them again)
            main_img_color = cv2.imdecode(
                np.frombuffer(main_img_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            template_img_color = cv2.imdecode(
                np.frombuffer(template_img_bytes, np.uint8), cv2.IMREAD_COLOR
            )

            res = cv2.matchTemplate(
                main_img_color, template_img_color, cv2.TM_CCORR_NORMED
            )
            _, _, _, max_loc = cv2.minMaxLoc(res)

            # 2. Get gradients of the *correlation result map (res)*
            # This shows how "fast" the correlation changes, which is unusual.
            h, w = template_img_color.shape[:-1]

            # Ensure patch is valid
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # The correlation map `res` is smaller than the main image.
            # Its size is (W-w+1, H-h+1).
            # max_loc is just a single point (x,y) in this map.
            # Your original patch logic was flawed.

            # Let's compute gradients on the *matched patch* from the *main image*
            patch = main_img_gray[
                top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
            ]

            sobel_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
            direction = np.arctan2(sobel_y, sobel_x)

            cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX, -1)
            mag_heatmap = cv2.applyColorMap(mag.astype(np.uint8), cv2.COLORMAP_BONE)
            dir_color = direction_to_color(direction)

            if mag_heatmap.shape[0] == 0 or dir_color.shape[0] == 0:
                return jsonify({"error": "Match produced an empty patch."}), 400

            sobel_combined = cv2.hconcat((mag_heatmap, dir_color))

            _, img_encoded = cv2.imencode(".png", sobel_combined)
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype="image/png")

        except Exception as e:
            return jsonify({"error": f"An error occurred on the server: {str(e)}"}), 500

    return render_template("gradient_match.html")

