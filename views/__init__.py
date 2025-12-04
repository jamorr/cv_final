# views/__init__.py
from .persproj import persproj_bp
from .template_match import template_match_bp
from .template_blur import template_blur_bp
from .gblur import gblur_bp
from .gradient_match import gradient_match_bp
from .stitch import stitch_bp
from .sift import sift_comp_bp
from .tracker import tracker_bp
from .segment import segment_bp
from .segment_select import segment_select_bp
from .edges_corners import edges_corners_bp
from .stereo import stereo_bp
from .pose import pose_bp

__all__ = [
    "stereo_bp",
    "edges_corners_bp",
    "segment_select_bp",
    "segment_bp",
    "tracker_bp",
    "persproj_bp",
    "template_match_bp",
    "template_blur_bp",
    "gblur_bp",
    "gradient_match_bp",
    "stitch_bp",
    "sift_comp_bp",
]
