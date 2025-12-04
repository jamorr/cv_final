# views/persproj.py
from flask import Blueprint, render_template, request, jsonify
import cv2
import numpy as np
import json
from pathlib import Path
import math

# 1. Create the Blueprint object
persproj_bp = Blueprint(
    "persproj", __name__, template_folder="../templates", static_folder="../static"
)


def get_default_calibration():
    calib_path = Path(__file__).parent.parent / "static" / "calibration"

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows, cols = 4, 3
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    images = calib_path.glob("*.jpg")
    last_h, last_w = None, None
    for fname in images:
        img = cv2.imread(str(fname))  # Use str(fname) for cv2.imread
        h, w = img.shape[0:2]
        if last_h is None:
            last_h = h
            last_w = w
        if last_h != h:
            img = np.rot90(img)
        if last_h != h:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (rows, cols), None)
        if ret:
            print("Points found!")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        else:
            print("No points found")

    if not objpoints or not imgpoints:
        print("Error: No valid calibration points found. Returning default matrix.")
        # Return a plausible default if calibration fails
        return np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=float)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))
    print(mtx, rvecs, tvecs, sep="\n\n")
    return mtx


# 3. Change @app.route to @persproj_bp.route
@persproj_bp.route("/persproj", methods=["GET", "POST"])
def persproj():
    # 4. Adjust path here as well
    load_path = Path(__file__).parent.parent / "static" / "calibration" / "default.json"
    try:
        with load_path.open() as f:
            DEFAULT_CALIBRATION = json.load(f)
    except FileNotFoundError:
        DEFAULT_CALIBRATION = {"0": get_default_calibration().tolist()}
        with load_path.open("w") as f:  # Use 'w' instead of 'x+' for simplicity
            json.dump(DEFAULT_CALIBRATION, f)

    if request.method == "POST":
        try:
            cam_matrix_str = request.form.get("cameraMatrix")
            distance_cm = float(request.form.get("distance"))
            point1_str = request.form.get("point1")
            point2_str = request.form.get("point2")

            if not all([cam_matrix_str, distance_cm, point1_str, point2_str]):
                return jsonify(
                    {
                        "error": "Missing data. Please fill all fields and select two points."
                    }
                )

            cam_matrix = np.array(json.loads(cam_matrix_str))
            p1 = json.loads(point1_str)
            p2 = json.loads(point2_str)
            u1, v1 = p1["x"], p1["y"]
            u2, v2 = p2["x"], p2["y"]
            print(cam_matrix)
            fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
            delta_u = u2 - u1
            delta_v = v2 - v1
            Z = distance_cm

            if fx == 0:
                return jsonify({"error": "fx cannot be zero."})
            Sx = (delta_u * Z) / fx
            if fy == 0:
                print("Error: fy cannot be zero.")
                return jsonify({"error": "fy cannot be zero."})
            Sy = (delta_v * Z) / fy
            S_real = math.sqrt(Sx**2 + Sy**2)

            return jsonify({"size_cm": f"{S_real:.2f}"})

        except Exception as e:
            print(cam_matrix_str)
            raise e
            return jsonify({"error": f"An error occurred: {str(e)}"})

    return render_template("persproj.html", calibration=DEFAULT_CALIBRATION["0"])

