# app.py
from flask import Flask, render_template
from views import (
    persproj_bp,
    template_match_bp,
    template_blur_bp,
    gblur_bp,
    gradient_match_bp,
    edges_corners_bp,
    stereo_bp,
    stitch_bp,
    sift_comp_bp,
    tracker_bp,
    segment_bp,
    segment_select_bp,
    pose_bp,
)

app = Flask(__name__)
app.jinja_env.filters["zip"] = zip

# --- Register Blueprints ---
app.register_blueprint(persproj_bp)
app.register_blueprint(template_match_bp)
app.register_blueprint(template_blur_bp)
app.register_blueprint(gblur_bp)
app.register_blueprint(gradient_match_bp)
app.register_blueprint(edges_corners_bp)
app.register_blueprint(stitch_bp)
app.register_blueprint(sift_comp_bp)
app.register_blueprint(tracker_bp)
app.register_blueprint(segment_bp)
app.register_blueprint(segment_select_bp)
app.register_blueprint(stereo_bp)
app.register_blueprint(pose_bp)

operations = [
    {"name": "Module 1"},
    {"url": "persproj.persproj", "name": "Perspective Projection"},
    {"name": "Module 2"},
    {"url": "template_match.template_match", "name": "Template Matching"},
    {"url": "gblur.gblur", "name": "Gaussian Blur and Recovery"},
    {"url": "template_blur.template_blur", "name": "Template Match then Blur"},
    {"name": "Module 3"},
    {
        "url": "gradient_match.gradient_match",
        "name": "Gradient vs Gaussian Blurred Lapacian",
    },
    {"url": "edges_corners.index", "name": "Edge and Corner Feature Detection"},
    {"url": "segment.segment", "name": "AruCo Segmentation"},
    {"url": "segment_select.index", "name": "Click Segmentation"},
    {"name": "Module 4"},
    {"url": "stitch.stitch", "name": "Panorama Stitching"},
    {"url": "sift_comp.sift_compare", "name": "SIFT from scratch with RANSAC"},
    {"name": "Module 5-6"},
    {
        "name": "Written Portion",
        "link": "https://drive.google.com/file/d/1yGtQyMoSCg3qVlzl-Sp_ppYcLWNrondO/view?usp=drive_link",
    },
    {"url": "tracker.index", "name": "Real Time object tracking"},
    {"name": "Module 7"},
    {"url": "stereo.index", "name": "Calibrated Stereo Size Estimation"},
    {"url": "pose.index", "name": "Realtime Pose Estimation"},
]


@app.route("/")
def index():
    return render_template("index.html", operations=operations)


if __name__ == "__main__":
    app.run(port=4000, debug=True)
