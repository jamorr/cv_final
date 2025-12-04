# views/gblur.py
from flask import Blueprint, render_template, request, send_file, jsonify
import cv2
import numpy as np
import io

gblur_bp = Blueprint("gblur", __name__, template_folder="../templates")


def direction_to_color(angle_rad_matrix: np.ndarray) -> np.ndarray:
    hue = ((angle_rad_matrix + np.pi) * (180 / (2 * np.pi))).astype(np.uint8)
    saturation = np.full(angle_rad_matrix.shape, 255, dtype=np.uint8)
    value = np.full(angle_rad_matrix.shape, 255, dtype=np.uint8)
    hsv_image = cv2.merge([hue, saturation, value])
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return bgr_image


def high_pass(blurred_img, gkern, k_factor):
    dft = np.fft.fft2(blurred_img)
    wiener_filter = np.conj(gkern) / (np.abs(gkern) ** 2 + k_factor)
    unblur_dft = dft * wiener_filter

    # Inverse FFT
    img_back = np.fft.ifft2(unblur_dft)
    img_back = np.fft.ifftshift(img_back)

    restored_img = np.clip(img_back, 0, 255).astype(np.uint8)
    return restored_img


@gblur_bp.route("/gblur", methods=["GET", "POST"])
def gblur():
    if request.method == "POST":
        try:
            k_factor = float(request.form.get("kFactor", 0.01))
            sigma = int(request.form.get("sigma", 5))
            k_h = k_w = int(request.form.get("size", 11))
            if k_h % 2 != 1:
                return jsonify({"error": "The kernel size must be odd."}), 400
            if sigma <= 0:
                return jsonify({"error": "Provide a positive variance."}), 400

            main_image_file = request.files.get("mainImage")

            if not main_image_file:
                return jsonify({"error": "Please upload a main image."}), 400

            main_img_bytes = main_image_file.read()
            main_img = cv2.imdecode(
                np.frombuffer(main_img_bytes, np.uint8), cv2.IMREAD_COLOR
            )

            if main_img is None:
                return jsonify(
                    {
                        "error": "Could not decode one of the images. Please ensure they are valid image files."
                    }
                ), 400
            h, w = main_img.shape[:-1]
            blurred_img = cv2.GaussianBlur((np.float32(main_img)), (k_h, k_w), sigma)
            kernel_1d = cv2.getGaussianKernel(k_h, sigma)
            kernel_2d = kernel_1d @ kernel_1d.T  # outer prod
            kernel_padded = np.zeros((h, w), dtype=kernel_2d.dtype)
            kernel_padded[:k_h, :k_w] = kernel_2d
            inv_gkern = np.fft.fft2(np.fft.ifftshift(kernel_padded))
            result = []
            for channel in cv2.split(blurred_img):
                result.append(high_pass(channel, inv_gkern, k_factor))

            img_back = cv2.merge(result)
            imgs_combined = cv2.vconcat(
                (main_img, np.uint8(blurred_img), np.uint8(img_back))
            )

            _, img_encoded = cv2.imencode(".png", imgs_combined)
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype="image/png")

        except Exception as e:
            return jsonify({"error": f"An error occurred on the server: {str(e)}"}), 500

    return render_template("gblur.html")
