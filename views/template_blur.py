# views/template_blur.py
from flask import Blueprint, render_template, request, send_file, jsonify
import cv2
import numpy as np
import io
from pathlib import Path
from functools import partial
from itertools import chain
from typing import List

template_blur_bp = Blueprint(
    "template_blur", __name__, template_folder="../templates", static_folder="../static"
)


def match_template(
    template_img: cv2.typing.MatLike,
    color: tuple[int, int, int],
    main_img: cv2.typing.MatLike,
    threshold: float,
) -> tuple[float, tuple[int, int], int, int, tuple[int, int, int]]:
    h, w = template_img.shape[:-1]
    res = cv2.matchTemplate(main_img, template_img, cv2.TM_CCORR_NORMED)
    _, max_corr, _, max_loc = cv2.minMaxLoc(res)
    if max_corr > threshold:
        return [(max_corr, max_loc, h, w, color)]
    else:
        return []


def blur_patch_and_draw_box(
    main_img: cv2.typing.MatLike,
    score: tuple[float, tuple[int, int], int, int, tuple[int, int, int]],
) -> None:
    top_left, h, w, color = score[1:]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    patch_slice = (
        slice(top_left[1], bottom_right[1]),
        slice(top_left[0], bottom_right[0]),
    )

    # Ensure patch is not empty
    if (
        patch_slice[0].start >= patch_slice[0].stop
        or patch_slice[1].start >= patch_slice[1].stop
    ):
        return

    patch = np.float32(main_img[patch_slice])

    # Kernel size must be odd and positive
    k_size = (11, 11)

    # Apply blur
    blurred_patch = cv2.GaussianBlur(patch, k_size, 4)
    main_img[patch_slice] = blurred_patch
    cv2.rectangle(main_img, top_left, bottom_right, color, 3)


def generate_distinct_colors(num_colors: int) -> List[tuple[int, int, int]]:
    if num_colors < 1:
        return []
    gray_values = np.linspace(0, 255, num_colors, dtype=np.uint8)
    gray_image = gray_values.reshape(-1, 1)
    color_image_bgr = cv2.applyColorMap(gray_image, cv2.COLORMAP_HSV)
    colors_bgr = [tuple(map(int, color)) for color in color_image_bgr.reshape(-1, 3)]
    return colors_bgr


@template_blur_bp.route("/template_blur", methods=["GET", "POST"])
def template_blur():
    # IMPORTANT: Adjust path
    TEMPLATE_PATH = Path(__file__).parent.parent / "static" / "template_matching_imgs"

    # Adjust path for relative_to
    tpaths = [
        p.relative_to(Path(__file__).parent.parent) for p in TEMPLATE_PATH.glob("*.png")
    ]

    template_colors = generate_distinct_colors(len(tpaths))
    hex_colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in template_colors]

    if request.method == "POST":
        try:
            main_image_file = request.files.get("mainImage")
            threshold = float(request.form.get("correlationThreshold"))
            blur_all = bool(request.form.get("blurAll"))

            if not main_image_file:
                return jsonify({"error": "Please upload an image"}), 400

            main_img_bytes = main_image_file.read()
            main_img = cv2.imdecode(
                np.frombuffer(main_img_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            templates = [
                cv2.imread(str(tpath.absolute()))
                for tpath in TEMPLATE_PATH.glob("*.png")
            ]
            if main_img is None:
                return jsonify({"error": "Could not decode one of the images."}), 400
            if not templates:
                return jsonify({"error": "Please generate templates on server."}), 400

            scores = list(
                chain.from_iterable(
                    map(
                        partial(match_template, main_img=main_img, threshold=threshold),
                        templates,
                        template_colors,
                    )
                )
            )

            if not len(scores):
                # Send back the original image if no matches are found
                _, img_encoded = cv2.imencode(".png", main_img)
                response = send_file(
                    io.BytesIO(img_encoded.tobytes()), mimetype="image/png"
                )
                # Add a custom header to signal the warning
                response.headers.add(
                    "X-App-Warning",
                    f"No matching locations found over correlation threshold {threshold}",
                )
                return response

            scores.sort(reverse=True, key=lambda x: x[0])

            if not blur_all:
                blur_patch_and_draw_box(main_img, scores[0])
            else:
                for score in scores:
                    blur_patch_and_draw_box(main_img, score)

            _, img_encoded = cv2.imencode(".png", main_img)
            return send_file(io.BytesIO(img_encoded.tobytes()), mimetype="image/png")

        except Exception as e:
            return jsonify({"error": f"An error occurred on the server: {str(e)}"}), 500

    return render_template("template_blur.html", colors=hex_colors, paths=tpaths)

