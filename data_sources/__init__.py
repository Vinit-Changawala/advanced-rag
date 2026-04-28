# ============================================================
# data_sources/__init__.py
# 
# What is __init__.py?
# Every folder that contains Python code needs this file.
# It tells Python "this folder is a package (module)".
# It also lets us control what gets exported when someone
# does: from data_sources import something
# ============================================================

# We import the main classes so users can do:
# from data_sources import DocumentLoader
# instead of:
# from data_sources.document_loader import DocumentLoader

from .document_loader import DocumentLoader
from .code_loader import CodeLoader
from .image_loader import ImageLoader
from .spreadsheet_loader import SpreadsheetLoader

# __all__ defines what gets exported when someone does:
# from data_sources import *
__all__ = [
    "DocumentLoader",
    "CodeLoader", 
    "ImageLoader",
    "SpreadsheetLoader"
]
