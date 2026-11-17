"""Load models once; generate or evaluate one Bengali couplet at a time."""

from .pipeline import PayarPipeline
from .semantic import SemanticClassifier
from .syntactic import analyze_couplet

__all__ = ["PayarPipeline", "SemanticClassifier", "analyze_couplet"]
