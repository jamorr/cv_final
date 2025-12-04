from flask import Blueprint, render_template

pose_bp = Blueprint("pose", __name__, template_folder="../templates")


@pose_bp.route("/pose", methods=["GET"])
def index():
    """
    Serves the client-side pose estimation page.
    Processing is now handled entirely in the browser via MediaPipe JS.
    """
    return render_template("pose.html")
