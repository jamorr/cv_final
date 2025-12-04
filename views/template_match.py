# views/template_match.py
from flask import Blueprint, render_template, request, send_file, jsonify
import cv2
import numpy as np
import io

template_match_bp = Blueprint('template_match', __name__,
                              template_folder='../templates')

@template_match_bp.route("/template_match", methods=["GET", "POST"])
def template_match():
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
            main_img = cv2.imdecode(
                np.frombuffer(main_img_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            template_img = cv2.imdecode(
                np.frombuffer(template_img_bytes, np.uint8), cv2.IMREAD_COLOR
            )

            if main_img is None or template_img is None:
                return jsonify(
                    {
                        "error": "Could not decode one of the images. Please ensure they are valid image files."
                    }
                ), 400

            h, w = template_img.shape[:-1]
            res = cv2.matchTemplate(main_img, template_img, cv2.TM_CCORR_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(main_img, top_left, bottom_right, (0, 255, 0), 3)
            _, img_encoded = cv2.imencode(".png", main_img)
            
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype="image/png")

        except Exception as e:
            return jsonify({"error": f"An error occurred on the server: {str(e)}"}), 500

    return render_template("template_match.html")