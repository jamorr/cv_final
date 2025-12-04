from os.path import islink
from flask import Blueprint, render_template, request, send_file, jsonify
import cv2
import numpy as np
import io
import base64
import os
import uuid
import shutil
import glob

stitch_bp = Blueprint("stitch", __name__, template_folder="../templates")

TEMP_DIR = os.path.join(os.getcwd(), "temp_data")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)
STATIC_DIR = os.path.join(
    os.path.dirname(__file__), "..", "static", "panorama_stitching"
)
MAX_DIMENSION = 1920


def scale_image_to_max_dim(img, max_dim):
    """
    Scales an image down so that its largest dimension (width or height)
    does not exceed max_dim.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img

    if w > h:
        scale_factor = max_dim / w
    else:
        scale_factor = max_dim / h

    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_images_to_disk(upload_id, images, demo_mode):
    """
    Saves a list of OpenCV images to disk under a specific upload_id
    after scaling them down.
    """
    session_dir = os.path.join(TEMP_DIR, upload_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)  # Clear old data if ID reused

    if demo_mode:
        # TODO: make a soft link to the STATIC_DIR as the session dir
        os.symlink(STATIC_DIR, session_dir)

    else:
        os.makedirs(session_dir)

        for i, img in enumerate(images):
            scaled_img = scale_image_to_max_dim(img, MAX_DIMENSION)
            cv2.imwrite(os.path.join(session_dir, f"image_{i:03d}.jpg"), scaled_img)


def load_images_from_disk(upload_id: str):
    """Loads images from disk for a specific upload_id."""
    session_dir = os.path.join(TEMP_DIR, upload_id)

    if not os.path.exists(session_dir):
        return None, False

    image_files = sorted(glob.glob(os.path.join(session_dir, "*.jpg")))
    images = []
    for f in image_files:
        img = cv2.imread(f)
        if img is not None:
            images.append(img)
    return images, os.path.islink(session_dir)


def cleanup_session(upload_id, is_demo):
    """Removes the temporary directory for a specific upload_id."""
    session_dir = os.path.join(TEMP_DIR, upload_id)
    if is_demo:
        os.unlink(session_dir)
        print(f"Cleaned up session link: {session_dir}")

    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
        print(f"Cleaned up session directory: {session_dir}")


def detect_and_match_features(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return kp1, kp2, []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return kp1, kp2, good_matches


def compute_homographies(images, center_idx):
    global_homographies = [None] * len(images)
    global_homographies[center_idx] = np.eye(3)

    # Right Side
    current_H = np.eye(3)
    for i in range(center_idx, len(images) - 1):
        img_dest, img_src = images[i], images[i + 1]
        kp_dest, kp_src, good = detect_and_match_features(img_dest, img_src)
        if len(good) < 4:
            return None, f"Not enough matches (Img {i + 1}-{i + 2})"
        src_pts = np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_dest[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        H_rel, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H_rel is None:
            return None, "Homography failed"
        current_H = np.matmul(current_H, H_rel)
        global_homographies[i + 1] = current_H

    # Left Side
    current_H = np.eye(3)
    for i in range(center_idx, 0, -1):
        img_dest, img_src = images[i], images[i - 1]
        kp_dest, kp_src, good = detect_and_match_features(img_dest, img_src)
        if len(good) < 4:
            return None, f"Not enough matches (Img {i}-{i + 1})"
        src_pts = np.float32([kp_src[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_dest[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        H_rel, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H_rel is None:
            return None, "Homography failed"
        current_H = np.matmul(current_H, H_rel)
        global_homographies[i - 1] = current_H

    return global_homographies, None


def create_weight_mask(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[1:-1, 1:-1] = 255
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    return dist


def stitch_all_images(images, manual_center_idx=None):
    if len(images) < 2:
        return images[0], [images[0]], None
    center_idx = (
        manual_center_idx if manual_center_idx is not None else len(images) // 2
    )

    global_homographies, error = compute_homographies(images, center_idx)
    if error:
        return None, None, error

    all_corners = []
    projected_areas = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, global_homographies[i])
        all_corners.append(warped_corners)
        projected_areas.append(cv2.contourArea(warped_corners))

    median_area = np.median(projected_areas)
    valid_indices = [
        i
        for i, a in enumerate(projected_areas)
        if a < 20 * median_area and a > 0.05 * median_area
    ]
    if center_idx not in valid_indices:
        valid_indices.append(center_idx)
    valid_indices.sort()

    if len(valid_indices) < 2:
        valid_indices = range(len(images))

    final_images = [images[i] for i in valid_indices]
    final_homographies = [global_homographies[i] for i in valid_indices]
    final_corners = [all_corners[i] for i in valid_indices]

    all_points = np.concatenate(final_corners, axis=0)
    [xmin, ymin] = np.int32(all_points.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_points.max(axis=0).ravel() + 0.5)

    full_w, full_h = xmax - xmin, ymax - ymin
    MAX_DIM_OUTPUT = 4000
    scale = (
        MAX_DIM_OUTPUT / max(full_w, full_h)
        if max(full_w, full_h) > MAX_DIM_OUTPUT
        else 1.0
    )
    canvas_w, canvas_h = int(full_w * scale), int(full_h * scale)

    T = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float32)
    M = np.matmul(S, T)

    accumulator = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_accumulator = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for i, img in enumerate(final_images):
        H_final = np.matmul(M, final_homographies[i])
        warped_img = cv2.warpPerspective(img, H_final, (canvas_w, canvas_h)).astype(
            np.float32
        )
        mask = create_weight_mask(img)
        warped_mask = cv2.warpPerspective(mask, H_final, (canvas_w, canvas_h))
        warped_mask_3ch = np.dstack([warped_mask] * 3)
        accumulator += warped_img * warped_mask_3ch
        weight_accumulator += warped_mask

    weight_accumulator[weight_accumulator == 0] = 1.0
    weight_accumulator_3ch = np.dstack([weight_accumulator] * 3)
    final_canvas = np.clip(accumulator / weight_accumulator_3ch, 0, 255).astype(
        np.uint8
    )

    display_images = [img.copy() for img in images]

    return final_canvas, display_images, None


def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@stitch_bp.route("/stitch", methods=["GET", "POST"])
def stitch():
    if request.method == "POST":
        # select base view then stitch
        if "base_index" in request.form and request.form["base_index"] != "":
            upload_id = request.form.get("upload_id")
            base_idx = int(request.form["base_index"])

            # Load from Disk
            images, is_demo = load_images_from_disk(upload_id)
            if not images:
                return render_template(
                    "stitch.html", error="Session expired. Please upload images again."
                )
            if is_demo:
                demo_image, _ = load_images_from_disk(
                    os.path.join(upload_id, "reference")
                )
            else:
                demo_image = None

            try:
                result_img, display_images, error_msg = stitch_all_images(
                    images, manual_center_idx=base_idx
                )
                source_b64_list = [encode_image(img) for img in display_images]

                if demo_image is not None:
                    demo_image_b64 = encode_image(
                        scale_image_to_max_dim(demo_image[0], 650)
                    )
                else:
                    demo_image_b64 = None

                cleanup_session(upload_id, is_demo)

                if error_msg:
                    return render_template(
                        "stitch.html",
                        error=error_msg,
                        # source_images=source_b64_list, # Can't show selection if cleanup occurred
                        upload_id=None,
                        selection_mode=False,
                    )
                result_b64 = encode_image(result_img)

                if demo_image is not None:
                    return render_template(
                        "stitch.html",
                        result_image=result_b64,
                        source_images=source_b64_list,
                        demo_image=demo_image_b64,
                        upload_id=None,  # Clear ID after cleanup
                        selection_mode=False,
                    )
                else:
                    return render_template(
                        "stitch.html",
                        result_image=result_b64,
                        source_images=source_b64_list,
                        upload_id=None,  # Clear ID after cleanup
                        selection_mode=False,
                    )
            except Exception as e:
                cleanup_session(upload_id, is_demo)
                return render_template(
                    "stitch.html", error=f"Stitching error: {str(e)}"
                )

        # upload to temp dir
        images = []
        uploaded_files = request.files.getlist("images")
        valid_uploads = [f for f in uploaded_files if f.filename != ""]

        if valid_uploads:
            for file in valid_uploads:
                in_memory_file = io.BytesIO(file.read())
                img = cv2.imdecode(
                    np.frombuffer(in_memory_file.getvalue(), np.uint8), cv2.IMREAD_COLOR
                )
                if img is not None:
                    images.append(img)

        if not images:  # demo fallback
            demo = True
            if os.path.exists(STATIC_DIR):
                filenames = sorted(
                    [f for f in os.listdir(STATIC_DIR) if f.lower().endswith(".jpg")]
                )
                for fname in filenames:
                    images.append(cv2.imread(os.path.join(STATIC_DIR, fname)))
        else:
            demo = False

        if len(images) < 2:
            return render_template(
                "stitch.html", error="Need at least 2 images to stitch."
            )

        new_upload_id = str(uuid.uuid4())
        save_images_to_disk(new_upload_id, images, demo)

        source_b64_list = [encode_image(img) for img in images]
        return render_template(
            "stitch.html",
            source_images=source_b64_list,
            upload_id=new_upload_id,
            selection_mode=True,
            prompt="Click an image below to set it as the center reference:",
        )

    return render_template("stitch.html")
