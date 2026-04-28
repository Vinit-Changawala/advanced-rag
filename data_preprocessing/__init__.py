# ============================================================
# data_preprocessing/__init__.py
#
# This file makes data_preprocessing a Python package.
# We export the main pipeline so users can do:
#   from data_preprocessing import PreprocessingPipeline
# instead of the longer:
#   from data_preprocessing.pipeline import PreprocessingPipeline
# ============================================================

from .pipeline import PreprocessingPipeline

__all__ = ["PreprocessingPipeline"]
