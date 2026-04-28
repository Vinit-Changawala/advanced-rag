# ============================================================
# data_sources/document_loader.py
#
# PURPOSE: Load text from PDF, DOCX, TXT, and Markdown files.
# Think of this as a "file reader" that handles many formats.
#
# BEGINNER CONCEPT - What is a "loader"?
# A loader takes a file from your hard drive and converts it
# into plain text that Python can work with.
# PDF files are binary (not readable as text directly),
# so we need special libraries to extract the text.
# ============================================================

import os                          # For file path operations
import logging                     # For printing log messages
from pathlib import Path           # Modern way to work with file paths
from typing import List, Dict, Any # Type hints - makes code easier to understand

# Third-party libraries (install via pip)
# pip install pypdf python-docx
import pypdf                       # Reads PDF files
from docx import Document          # Reads Microsoft Word files

# Set up logging for this module
# logging lets us print messages with timestamps and levels (INFO, ERROR, etc.)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads documents from various file formats and returns them as text.
    
    BEGINNER CONCEPT - What is a class?
    A class is like a blueprint. DocumentLoader is a blueprint for 
    creating "loader objects" that can load files.
    
    Usage:
        loader = DocumentLoader()
        result = loader.load("my_document.pdf")
        print(result["content"])  # Prints the extracted text
    """
    
    # Supported file extensions (what file types we can handle)
    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}
    
    def __init__(self):
        """
        __init__ is the constructor - it runs when you create a new DocumentLoader.
        Like setting up your workspace before starting work.
        """
        logger.info("DocumentLoader initialized")
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Main method: Load a single document and return its content.
        
        Args:
            file_path: The path to the file, e.g., "/documents/report.pdf"
            
        Returns:
            A dictionary with keys: content, metadata, source_type
            
        BEGINNER CONCEPT - What is a Dict[str, Any]?
        Dict means dictionary (like a Python {}).
        str means the keys are strings.
        Any means the values can be any type.
        So: {"content": "hello world", "pages": 5}
        """
        # Convert to Path object for easier manipulation
        path = Path(file_path)
        
        # Check if the file actually exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get the file extension (e.g., ".pdf", ".docx")
        extension = path.suffix.lower()
        
        # Check if we support this file type
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}. Supported: {self.SUPPORTED_EXTENSIONS}")
        
        logger.info(f"Loading document: {file_path}")
        
        # Route to the correct loading method based on file type
        # This is called a "dispatch" pattern
        if extension == ".pdf":
            content = self._load_pdf(path)
        elif extension == ".docx":
            content = self._load_docx(path)
        elif extension in {".txt", ".md", ".markdown"}:
            content = self._load_text(path)
        else:
            content = ""
        
        # Return a structured result dictionary
        return {
            "content": content,              # The extracted text
            "source": str(file_path),        # Where it came from
            "source_type": "document",       # Category for this source
            "file_type": extension,          # The file extension
            "file_name": path.name,          # Just the filename (no path)
            "file_size_bytes": path.stat().st_size,  # File size
        }
    
    def load_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Load ALL supported documents from a folder.
        
        Args:
            directory_path: Path to folder containing documents
            
        Returns:
            List of document dictionaries (one per file)
        """
        directory = Path(directory_path)
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        results = []
        
        # Walk through all files in the directory
        # rglob("*") means "search recursively through all subfolders"
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    doc = self.load(str(file_path))
                    results.append(doc)
                    logger.info(f"Loaded: {file_path.name}")
                except Exception as e:
                    # Don't crash if one file fails - log and continue
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Loaded {len(results)} documents from {directory_path}")
        return results
    
    def _load_pdf(self, path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Note: The underscore prefix (_load_pdf) means this is a "private" method.
        Private means it's only meant to be used inside this class, not by outside code.
        """
        text_parts = []
        
        # Open the PDF file
        with open(path, "rb") as file:  # "rb" = read binary mode
            reader = pypdf.PdfReader(file)
            
            # Loop through each page and extract text
            for page_number, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:  # Only add if there's actual text
                    text_parts.append(f"[Page {page_number + 1}]\n{page_text}")
        
        # Join all pages with a separator
        return "\n\n".join(text_parts)
    
    def _load_docx(self, path: Path) -> str:
        """Extract text from a Microsoft Word (.docx) file."""
        doc = Document(str(path))
        
        text_parts = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Skip empty paragraphs
                text_parts.append(paragraph.text)
        
        return "\n\n".join(text_parts)
    
    def _load_text(self, path: Path) -> str:
        """Load a plain text or Markdown file."""
        # Try UTF-8 first, fall back to latin-1 if that fails
        # (some files have special characters that need different encoding)
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")
