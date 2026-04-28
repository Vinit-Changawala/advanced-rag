# ============================================================
# data_sources/code_loader.py
#
# PURPOSE: Load source code files (.py, .js, .ts, .java, etc.)
# 
# WHY SPECIAL HANDLING FOR CODE?
# Code files need extra metadata like the programming language,
# function names, and class names so agents can search them better.
# We also preserve the structure (indentation matters in Python!)
# ============================================================

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class CodeLoader:
    """
    Loads source code files and extracts useful metadata.
    
    Usage:
        loader = CodeLoader()
        result = loader.load("my_script.py")
        print(result["language"])   # "python"
        print(result["content"])    # The actual code
    """
    
    # Map file extensions to programming language names
    # This is called a "lookup table" or "mapping"
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".sql": "sql",
        ".sh": "bash",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".html": "html",
        ".css": "css",
    }
    
    def __init__(self):
        self.supported_extensions = set(self.LANGUAGE_MAP.keys())
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load a code file and return content with metadata.
        
        The metadata includes the language, line count, etc.
        This helps the search system understand what kind of
        code is being stored.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Code file not found: {file_path}")
        
        extension = path.suffix.lower()
        language = self.LANGUAGE_MAP.get(extension, "unknown")
        
        # Read the raw code
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = path.read_text(encoding="latin-1")
        
        # Count lines (useful for chunking large files)
        lines = content.split("\n")
        line_count = len(lines)
        
        # Extract function/class names for better search metadata
        symbols = self._extract_symbols(content, language)
        
        logger.info(f"Loaded code file: {path.name} ({language}, {line_count} lines)")
        
        return {
            "content": content,
            "source": str(file_path),
            "source_type": "code",
            "file_type": extension,
            "file_name": path.name,
            "language": language,
            "line_count": line_count,
            "symbols": symbols,        # Function and class names
        }
    
    def load_directory(self, directory_path: str, 
                       exclude_dirs: List[str] = None) -> List[Dict[str, Any]]:
        """
        Load all code files from a directory.
        
        Args:
            directory_path: Root folder to search
            exclude_dirs: Folders to skip (e.g., ["node_modules", ".git", "venv"])
        """
        # Default directories to exclude (they contain non-project code)
        if exclude_dirs is None:
            exclude_dirs = ["node_modules", ".git", "venv", "__pycache__", 
                          ".env", "dist", "build", ".idea", ".vscode"]
        
        directory = Path(directory_path)
        results = []
        
        for file_path in directory.rglob("*"):
            # Skip excluded directories
            # any() returns True if ANY item in the list is true
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue
            
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    result = self.load(str(file_path))
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        return results
    
    def _extract_symbols(self, content: str, language: str) -> List[str]:
        """
        Extract function and class names from code.
        
        This is a simple approach using string matching.
        In production, you'd use a proper AST (Abstract Syntax Tree) parser.
        
        BEGINNER CONCEPT - What is string matching?
        We look for patterns like "def " (Python functions) or
        "class " to find where functions and classes are defined.
        """
        symbols = []
        lines = content.split("\n")
        
        for line in lines:
            stripped = line.strip()
            
            if language == "python":
                # Python functions start with "def "
                # Python classes start with "class "
                if stripped.startswith("def "):
                    # Extract just the name: "def my_function(args):" -> "my_function"
                    name = stripped[4:].split("(")[0].strip()
                    symbols.append(f"function:{name}")
                elif stripped.startswith("class "):
                    name = stripped[6:].split("(")[0].split(":")[0].strip()
                    symbols.append(f"class:{name}")
                    
            elif language in ("javascript", "typescript"):
                if "function " in stripped:
                    parts = stripped.split("function ")
                    if len(parts) > 1:
                        name = parts[1].split("(")[0].strip()
                        symbols.append(f"function:{name}")
                        
            elif language == "java":
                # Java methods/classes are more complex, simple detection
                if "public " in stripped or "private " in stripped:
                    if "class " in stripped:
                        name = stripped.split("class ")[1].split(" ")[0]
                        symbols.append(f"class:{name}")
        
        return symbols
