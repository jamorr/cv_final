from flask import Blueprint, render_template, request
import cv2
import numpy as np
import time
import base64
import io

sift_comp_bp = Blueprint("sift_comp", __name__, template_folder="../templates")

# ==========================================
# PART 1: IMPROVED FROM SCRATCH IMPLEMENTATION
# ==========================================


def compute_homography_dlt(src_pts, dst_pts):
    """
    Computes Homography using Direct Linear Transform (DLT) from 4 point correspondences.
    """
    A = []
    for i in range(4):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H


def ransac_homography_scratch(matches, kp1, kp2, threshold=5.0, max_iters=2000):
    """
    RANSAC optimization to find best Homography.
    """
    if len(matches) < 4:
        return None, []

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    best_H = None
    max_inliers = 0
    best_inliers_mask = []

    src_pts_h = np.hstack((src_pts, np.ones((len(src_pts), 1))))

    # Optimization: Early exit if high ratio of inliers found
    match_count = len(matches)

    for _ in range(max_iters):
        # 1. Randomly sample 4 points
        indices = np.random.choice(match_count, 4, replace=False)
        pts1_sample = src_pts[indices]
        pts2_sample = dst_pts[indices]

        # 2. Compute Homography
        try:
            H = compute_homography_dlt(pts1_sample, pts2_sample)
        except np.linalg.LinAlgError:
            continue

        # 3. Project
        dst_proj = (H @ src_pts_h.T).T

        with np.errstate(divide="ignore", invalid="ignore"):
            dst_proj = dst_proj / dst_proj[:, 2:3]

        # 4. Error
        diff = dst_proj[:, :2] - dst_pts
        errors = np.linalg.norm(diff, axis=1)

        # 5. Count inliers
        inliers_mask = errors < threshold
        num_inliers = np.sum(inliers_mask)

        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inliers_mask = inliers_mask

            # If 80% matches are inliers, stop early
            if num_inliers > match_count * 0.8:
                break

    inlier_matches = [matches[i] for i in range(len(matches)) if best_inliers_mask[i]]
    return best_H, inlier_matches


def generate_gaussian_pyramid(image, num_octaves=3, scales_per_octave=3):
    """
    Generates Gaussian Pyramid.
    Reduced octaves slightly for speed in Python.
    """
    pyr = []
    k = 2 ** (1 / scales_per_octave)

    current_img = image.copy()

    for _ in range(num_octaves):
        octave = [current_img]
        sigma = 1.6

        for _ in range(scales_per_octave + 2):
            sigma_prev = sigma
            sigma *= k
            sigma_diff = np.sqrt(sigma**2 - sigma_prev**2)
            blurred = cv2.GaussianBlur(octave[-1], (0, 0), sigma_diff)
            octave.append(blurred)

        pyr.append(octave)
        current_img = octave[-3][::2, ::2]

    return pyr


def generate_dog_pyramid(gaussian_pyr):
    dog_pyr = []
    for octave in gaussian_pyr:
        dog_octave = []
        for i in range(len(octave) - 1):
            dog = cv2.subtract(octave[i + 1], octave[i])
            dog_octave.append(dog)
        dog_pyr.append(dog_octave)
    return dog_pyr


def find_local_extrema(dog_pyr):
    """
    Finds keypoints in Scale Space.
    """
    keypoints = []
    # Lower threshold to find more points (OpenCV uses ~0.04, we relax to 0.03)
    contrast_threshold = 0.03 * 255

    for octave_idx, octave in enumerate(dog_pyr):
        for scale_idx in range(1, len(octave) - 1):
            img = octave[scale_idx]
            prev_img = octave[scale_idx - 1]
            next_img = octave[scale_idx + 1]

            # Vectorized neighbor check
            # We create a stack of the 3x3x3 neighborhood (27 pixels)
            # Center pixel is at index 13 (middle)

            # Shifts for 3x3 window in 3 images
            shifts = []
            for s_img in [prev_img, img, next_img]:
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        shifted = np.roll(s_img, (dy, dx), axis=(0, 1))
                        shifts.append(shifted)

            stack = np.stack(shifts, axis=0)
            center_pixel = stack[13]  # The pixel itself

            # Check if center is strictly max or strictly min
            # Create mask for "center is larger than all others" (excluding self)
            # We temporarily set self to min-1 to check max, etc.

            stack_no_center = np.delete(stack, 13, axis=0)

            is_max = np.all(center_pixel > stack_no_center, axis=0)
            is_min = np.all(center_pixel < stack_no_center, axis=0)

            extrema_mask = np.logical_or(is_max, is_min)
            contrast_mask = np.abs(center_pixel) > contrast_threshold

            # Edge Response Elimination (Harris corner check approximation)
            # Calculating Hessian at these points
            dx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
            dy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
            dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=1)
            dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=1)
            dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=1)

            tr = dxx + dyy
            det = dxx * dyy - dxy * dxy

            # r = 10 (standard)
            # (r+1)^2 / r = 12.1
            edge_threshold = 12.1
            # Avoid divide by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                curvature_ratio = (tr * tr) / det

            edge_mask = (det > 0) & (curvature_ratio < edge_threshold)

            final_mask = extrema_mask & contrast_mask & edge_mask

            # Mask borders
            border = 10
            final_mask[:border, :] = 0
            final_mask[-border:, :] = 0
            final_mask[:, :border] = 0
            final_mask[:, -border:] = 0

            ys, xs = np.where(final_mask)

            for i in range(len(xs)):
                # Map back to global coordinates
                scale_mult = 2**octave_idx
                # Size approximation
                size = 10 * scale_mult * (1.6 * (2 ** (scale_idx / 3)))
                pt = cv2.KeyPoint(
                    float(xs[i] * scale_mult), float(ys[i] * scale_mult), size
                )
                keypoints.append(pt)

    return keypoints


def compute_custom_descriptors(image, keypoints):
    """
    Computes 256-D SIFT Descriptors (Upright SIFT) using 16 orientation bins.
    1. Extract 16x16 patch around keypoint.
    2. Compute gradients.
    3. Gaussian weight the magnitudes.
    4. Divide into 4x4 subregions.
    5. Compute 16-bin orientation histogram for each subregion (updated from 8).
    6. Concatenate to 4x4x16 = 256D vector.
    7. Normalize.
    """
    descriptors = []
    valid_kps = []

    # Pre-compute image gradients
    # Use float32 for precision
    img_float = image.astype(np.float32)
    gx = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=1)

    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 360

    # Custom SIFT parameters - 16 bins
    num_bins = 16
    bins_per_degree = 360.0 / num_bins  # 22.5 degrees

    # Gaussian window for weighting
    # sigma = 0.5 * window_width (16) = 8
    # But usually sigma is relative to descriptor width.
    # Let's use a fixed gaussian mask for the 16x16 patch for speed
    y_grid, x_grid = np.mgrid[-8:8, -8:8]
    gaussian_weight = np.exp(-(x_grid**2 + y_grid**2) / (2 * (8**2)))

    for kp in keypoints:
        x_c, y_c = int(np.round(kp.pt[0])), int(np.round(kp.pt[1]))

        # Boundary check (16x16 patch needs 8 px margin)
        if x_c < 8 or x_c >= image.shape[1] - 8 or y_c < 8 or y_c >= image.shape[0] - 8:
            continue

        # Extract patch (16x16)
        patch_mag = magnitude[y_c - 8 : y_c + 8, x_c - 8 : x_c + 8]
        patch_ang = angle[y_c - 8 : y_c + 8, x_c - 8 : x_c + 8]

        # Apply Gaussian Weighting
        weighted_mag = patch_mag * gaussian_weight

        # Create 256D vector (flattened 4x4 grid of 16-bin histograms)
        descriptor_vec = np.zeros((4, 4, num_bins))

        # Iterate over 4x4 subregions (each subregion is 4x4 pixels)
        for i in range(4):  # Row of subregion
            for j in range(4):  # Col of subregion
                # Extract the 4x4 pixel block for this subregion
                start_y, end_y = i * 4, (i + 1) * 4
                start_x, end_x = j * 4, (j + 1) * 4

                sub_mag = weighted_mag[start_y:end_y, start_x:end_x]
                sub_ang = patch_ang[start_y:end_y, start_x:end_x]

                # Assign to bins
                # Quantize angles to 0-15
                bin_indices = (sub_ang // bins_per_degree).astype(int) % num_bins

                # Add magnitudes to appropriate bins
                # (Simple voting, real SIFT uses trilinear interpolation)
                for by in range(4):
                    for bx in range(4):
                        bin_idx = bin_indices[by, bx]
                        mag_val = sub_mag[by, bx]
                        descriptor_vec[i, j, bin_idx] += mag_val

        # Flatten
        desc_flat = descriptor_vec.flatten()

        # Normalization (L2)
        norm = np.linalg.norm(desc_flat)
        if norm > 1e-6:
            desc_flat /= norm

        # Threshold (clip at 0.2 to reduce non-linear illumination effects)
        desc_flat = np.clip(desc_flat, 0, 0.2)

        # Re-normalize
        norm = np.linalg.norm(desc_flat)
        if norm > 1e-6:
            desc_flat /= norm

        descriptors.append(desc_flat)
        valid_kps.append(kp)

    return valid_kps, np.array(descriptors, dtype=np.float32)


def sift_scratch_pipeline(img_bgr):
    """
    Main entry point for scratch SIFT.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Scale Space
    g_pyr = generate_gaussian_pyramid(gray)
    d_pyr = generate_dog_pyramid(g_pyr)

    # 2. Keypoint Detection
    kps = find_local_extrema(d_pyr)

    # 3. 256-D Descriptor Extraction
    kps, des = compute_custom_descriptors(gray, kps)

    return kps, des


# ==========================================
# FLASK ROUTES & HELPERS
# ==========================================


def encode_img(img):
    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


def resize_for_performance(img, max_width=400):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


@sift_comp_bp.route("/sift_compare", methods=["GET", "POST"])
def sift_compare():
    if request.method == "POST":
        if "image1" not in request.files or "image2" not in request.files:
            return render_template(
                "sift_comparison.html", error="Please upload two images."
            )

        file1 = request.files["image1"]
        file2 = request.files["image2"]

        if file1.filename == "" or file2.filename == "":
            return render_template("sift_comparison.html", error="Missing files.")

        in_memory_file1 = io.BytesIO(file1.read())
        in_memory_file2 = io.BytesIO(file2.read())

        img1_orig = cv2.imdecode(
            np.frombuffer(in_memory_file1.getvalue(), np.uint8), cv2.IMREAD_COLOR
        )
        img2_orig = cv2.imdecode(
            np.frombuffer(in_memory_file2.getvalue(), np.uint8), cv2.IMREAD_COLOR
        )

        # Resize for "Scratch" performance
        img1_small = img1_orig
        img2_small = img2_orig

        results = {}

        # -----------------------------
        # 1. RUN OPENCV SIFT (Reference)
        # -----------------------------
        start_cv = time.time()
        sift = cv2.SIFT_create()
        kp1_cv, des1_cv = sift.detectAndCompute(img1_small, None)
        kp2_cv, des2_cv = sift.detectAndCompute(img2_small, None)

        bf = cv2.BFMatcher()
        matches_cv = bf.knnMatch(des1_cv, des2_cv, k=2)
        good_cv = []
        for m, n in matches_cv:
            if m.distance < 0.75 * n.distance:
                good_cv.append(m)

        if len(good_cv) >= 4:
            src_pts = np.float32([kp1_cv[m.queryIdx].pt for m in good_cv]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([kp2_cv[m.trainIdx].pt for m in good_cv]).reshape(
                -1, 1, 2
            )
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers_cv = np.sum(mask) if mask is not None else 0
        else:
            inliers_cv = 0

        time_cv = time.time() - start_cv

        res_cv_img = cv2.drawMatches(
            img1_small,
            kp1_cv,
            img2_small,
            kp2_cv,
            good_cv,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        results["opencv"] = {
            "time": f"{time_cv:.4f} s",
            "kps1": len(kp1_cv),
            "kps2": len(kp2_cv),
            "matches": len(good_cv),
            "inliers": int(inliers_cv),
            "img": encode_img(res_cv_img),
        }

        # -----------------------------
        # 2. RUN IMPROVED SCRATCH SIFT
        # -----------------------------
        start_scratch = time.time()

        kp1_sc, des1_sc = sift_scratch_pipeline(img1_small)
        kp2_sc, des2_sc = sift_scratch_pipeline(img2_small)

        # Match Scratch (Euclidean on 128D)
        # Using Cross-Check + Ratio Test for higher quality matching
        good_sc = []
        if (
            des1_sc is not None
            and des2_sc is not None
            and len(des1_sc) > 0
            and len(des2_sc) > 0
        ):
            # We still use OpenCV BFMatcher for the O(N^2) distance calculation speed
            # but the input descriptors are our scratch-computed ones.
            bf_sc = cv2.BFMatcher()

            # Direction 1: 1 -> 2
            matches_12 = bf_sc.knnMatch(des1_sc, des2_sc, k=2)
            # Direction 2: 2 -> 1 (Cross Check)
            matches_21 = bf_sc.knnMatch(des2_sc, des1_sc, k=2)

            # Filter 1->2 with Ratio Test
            good_12 = []
            for m_tuple in matches_12:
                if len(m_tuple) == 2:
                    m, n = m_tuple
                    if m.distance < 0.75 * n.distance:
                        good_12.append(m)

            # Cross Check: Ensure mutual match
            # This dramatically cleans up the results
            for m in good_12:
                # m.queryIdx is index in kp1, m.trainIdx is index in kp2
                # Check if kp2 matches back to kp1
                # Find match in 2->1 where queryIdx is m.trainIdx
                back_match = None
                for m2_tuple in matches_21:
                    if len(m2_tuple) >= 1:
                        m2 = m2_tuple[0]
                        if m2.queryIdx == m.trainIdx:
                            back_match = m2
                            break

                if back_match and back_match.trainIdx == m.queryIdx:
                    good_sc.append(m)

        # RANSAC Scratch
        H_sc, inliers_matches_sc = ransac_homography_scratch(good_sc, kp1_sc, kp2_sc)

        time_scratch = time.time() - start_scratch

        res_sc_img = cv2.drawMatches(
            img1_small,
            kp1_sc,
            img2_small,
            kp2_sc,
            inliers_matches_sc,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        results["scratch"] = {
            "time": f"{time_scratch:.4f} s",
            "kps1": len(kp1_sc),
            "kps2": len(kp2_sc),
            "matches": len(good_sc),
            "inliers": len(inliers_matches_sc),
            "img": encode_img(res_sc_img),
        }

        return render_template("sift_comparison.html", results=results)

    return render_template("sift_comparison.html")
