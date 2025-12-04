import base64
from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
import json
from pathlib import Path
import math

stereo_bp = Blueprint(
    "stereo", __name__, template_folder="../templates", static_folder="../static"
)


def get_default_calibration():
    calib_path = Path(__file__).parent.parent / "static" / "calibration"
    if not calib_path.exists():
        calib_path.mkdir(parents=True, exist_ok=True)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows, cols = 4, 3
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    images = list(calib_path.glob("*.jpg"))

    last_h, last_w = None, None
    for fname in images:
        img = cv2.imread(str(fname))
        if img is None:
            continue
        h, w = img.shape[0:2]
        if last_h is None:
            last_h, last_w = h, w
        if last_h != h:
            img = np.rot90(img)
        if last_h != h:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    if not objpoints or not imgpoints:
        return np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (last_w, last_h), None, None
    )
    return mtx


def compute_disparity(imgL, imgR, num_disp=96):
    max_dim = 1000
    h, w = imgL.shape[:2]
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        imgL = cv2.resize(imgL, None, fx=scale, fy=scale)
        imgR = cv2.resize(imgR, None, fx=scale, fy=scale)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=11,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
    )

    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    return disp, scale


def get_depth_at_point_raw(u, v, disparity):
    """Returns the raw disparity value at a point."""
    h, w = disparity.shape
    u, v = int(u), int(v)
    if u < 0 or u >= w or v < 0 or v >= h:
        return None

    window = disparity[max(0, v - 2) : min(h, v + 3), max(0, u - 2) : min(w, u + 3)]
    valid = window[window > 0]
    if len(valid) == 0:
        return None
    return np.median(valid)


def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@stereo_bp.route("/calibrated_stereo", methods=["GET", "POST"])
def index():
    load_path = Path(__file__).parent.parent / "static" / "calibration" / "default.json"
    try:
        with load_path.open() as f:
            DEFAULT_CALIBRATION = json.load(f)
    except:
        mtx = get_default_calibration()
        DEFAULT_CALIBRATION = {"0": mtx.tolist()}
        if not load_path.parent.exists():
            load_path.parent.mkdir(parents=True, exist_ok=True)
        with load_path.open("w") as f:
            json.dump(DEFAULT_CALIBRATION, f)

    if request.method == "POST":
        try:
            cam_matrix_str = request.form.get("cameraMatrix")
            baseline_str = request.form.get("baseline")
            hub_dist_str = request.form.get("hub_distance")
            points_data_str = request.form.get("points_data")

            try:
                num_disp_val = int(request.form.get("num_disp", 96))
                if num_disp_val % 16 != 0:
                    num_disp_val = (num_disp_val // 16) * 16
                if num_disp_val < 16:
                    num_disp_val = 16
            except:
                num_disp_val = 96

            fileL = request.files.get("imageL")
            fileR = request.files.get("imageR")

            # Basic validation: Need Matrix, Points, Images, AND (Baseline OR Hub Dist)
            if not all([cam_matrix_str, points_data_str, fileL, fileR]):
                return jsonify({"error": "Missing main parameters."})

            if not baseline_str and not hub_dist_str:
                return jsonify(
                    {"error": "You must provide either Baseline OR Distance to Hub."}
                )

            imgL = cv2.imdecode(np.frombuffer(fileL.read(), np.uint8), cv2.IMREAD_COLOR)
            imgR = cv2.imdecode(np.frombuffer(fileR.read(), np.uint8), cv2.IMREAD_COLOR)

            if imgL is None or imgR is None:
                return jsonify({"error": "Invalid images."})

            cam_matrix = np.array(json.loads(cam_matrix_str))
            fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
            cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]

            points_list = json.loads(points_data_str)
            if len(points_list) < 2:
                return jsonify({"error": "Select at least 2 points."})

            # 1. Compute Disparity
            disparity, scale = compute_disparity(imgL, imgR, num_disp=num_disp_val)

            # Scale intrinsics
            fx_s, fy_s, cx_s, cy_s = fx * scale, fy * scale, cx * scale, cy * scale

            # 2. Analyze Hub (First Point)
            p_hub = points_list[0]
            u_hub, v_hub = p_hub["x"] * scale, p_hub["y"] * scale

            d_hub = get_depth_at_point_raw(u_hub, v_hub, disparity)
            if d_hub is None or d_hub <= 0.1:
                return jsonify(
                    {
                        "error": "Could not determine disparity at Hub. Try a clearer point."
                    }
                )

            # 3. Determine Baseline and Depth (Z)
            # Geometry factors for Hub:
            # X = (u - cx) * Z / fx
            # Y = (v - cy) * Z / fy
            # Dist^2 = Z^2 * [ ((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1 ]

            geom_factor = math.sqrt(
                ((u_hub - cx_s) / fx_s) ** 2 + ((v_hub - cy_s) / fy_s) ** 2 + 1
            )

            if baseline_str:
                baseline_val = float(baseline_str)
                # Z = f * B / d
                Z_hub = (fx_s * baseline_val) / d_hub
                dist_l_hub = Z_hub * geom_factor
            else:
                # Educational Mode: Reverse Engineering Baseline
                dist_l_hub = float(hub_dist_str)
                Z_hub = dist_l_hub / geom_factor
                # B = Z * d / f
                baseline_val = (Z_hub * d_hub) / fx_s

            # Calculate Right Camera Distance for Educational Output
            # Right Camera is at (Baseline, 0, 0) relative to Left (0, 0, 0)
            X_hub_L = (u_hub - cx_s) * Z_hub / fx_s
            Y_hub_L = (v_hub - cy_s) * Z_hub / fy_s
            # Coords in Right Cam Frame: (X-B, Y, Z)
            dist_r_hub = math.sqrt(
                (X_hub_L - baseline_val) ** 2 + Y_hub_L**2 + Z_hub**2
            )

            # 4. Process Spokes
            measurements = []
            educational_edges = []

            for i, p_spoke in enumerate(points_list[1:], start=1):
                u_s, v_s = p_spoke["x"] * scale, p_spoke["y"] * scale
                d_s = get_depth_at_point_raw(u_s, v_s, disparity)

                if d_s is None or d_s <= 0.1:
                    measurements.append({"id": i, "error": "Depth undefined"})
                    educational_edges.append("")
                    continue

                Z_s = (fx_s * baseline_val) / d_s
                X_s = (u_s - cx_s) * Z_s / fx_s
                Y_s = (v_s - cy_s) * Z_s / fy_s

                dist = math.sqrt(
                    (X_hub_L - X_s) ** 2 + (Y_hub_L - Y_s) ** 2 + (Z_hub - Z_s) ** 2
                )

                measurements.append(
                    {
                        "id": i,
                        "length": f"{dist:.2f}",
                        "z_start": f"{Z_hub:.1f}",
                        "z_end": f"{Z_s:.1f}",
                    }
                )
                educational_edges.append(f"{dist:.2f}")

            # --- VISUALIZATION ---
            # 1. Disparity Map
            disp_vis = cv2.normalize(
                disparity,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
            disp_b64 = encode_image(cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET))

            # 2. Depth Map
            # Z = (f * B) / d
            # Avoid division by zero
            with np.errstate(divide="ignore"):
                depth_map = (fx_s * baseline_val) / disparity

            # Filter infinity or very large depths for better visualization
            depth_map[disparity <= 0] = 0
            depth_map[depth_map > 1000] = 1000  # Clip far depth for vis

            depth_vis = cv2.normalize(
                depth_map,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )
            # Invert so close = bright/red, far = dark/blue (standard convention usually varies, but JET is common)
            # Actually, standard depth maps are usually lighter = closer.
            # Let's stick to JET where Red=High Value (Far) and Blue=Low Value (Close)
            # BUT depth is usually visualized as inverse disparity.
            # Let's just map it to JET.
            depth_b64 = encode_image(cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET))

            return jsonify(
                {
                    "measurements": measurements,
                    "disparity_b64": disp_b64,
                    "depth_b64": depth_b64,
                    "used_num_disp": num_disp_val,
                    "educational": {
                        "baseline_used": f"{baseline_val:.2f}",
                        "dist_l_hub": f"{dist_l_hub:.2f}",
                        "dist_r_hub": f"{dist_r_hub:.2f}",
                        "edges": educational_edges,
                    },
                }
            )

        except Exception as e:
            return jsonify({"error": f"An error occurred: {str(e)}"})

    return render_template(
        "calibrated_stereo.html", calibration=DEFAULT_CALIBRATION["0"]
    )

