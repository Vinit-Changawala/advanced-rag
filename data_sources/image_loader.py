# ============================================================
# data_sources/image_loader.py
#
# PURPOSE: Extract text from images using OCR and Vision AI.
#
# BEGINNER CONCEPT - What is OCR?
# OCR = Optical Character Recognition
# It's the technology that reads text FROM images.
# Like when you take a photo of a sign and your phone reads it.
#
# BEGINNER CONCEPT - What is Vision AI?
# Vision AI (like GPT-4 Vision) can describe what's IN an image.
# So if you have a chart or diagram, it can explain what it shows.
# ============================================================

import base64           # Converts binary data (images) to text for APIs
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from pytesseract import pytesseract

logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Loads images and extracts text/descriptions using:
    1. OCR (pytesseract) for images with text
    2. Vision AI (OpenAI GPT-4 Vision) for understanding image content
    
    Usage:
        loader = ImageLoader(openai_client=client)
        result = loader.load("diagram.png")
        print(result["description"])  # AI description of the image
        print(result["ocr_text"])     # Any text found in the image
    """
    
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    
    def __init__(self, openai_client=None, mistral_client=None, use_ocr: bool = True):
        """
        Args:
            openai_client: An OpenAI client object for Vision AI
            mistral_client: A Mistral client object for Vision AI
            use_ocr: Whether to use OCR (requires pytesseract + tesseract installed)
        """
        self.openai_client = mistral_client or openai_client
        self.use_ocr = use_ocr
        
        # Try to import pytesseract (optional dependency)
        self._tesseract_available = False
        if use_ocr:
            try:
                import pytesseract
                self._tesseract_available = True
                logger.info("Tesseract OCR available")
            except ImportError:
                logger.warning("pytesseract not installed. OCR disabled. Install: pip install pytesseract")
    
    def load(self, file_path: str) -> Dict[str, Any]:
        """
        Load an image and extract all possible text/description.
        
        Returns a dict with both OCR text AND AI-generated description.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {file_path}")
        
        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {extension}")
        
        logger.info(f"Loading image: {path.name}")
        
        # Step 1: Read raw image bytes
        image_bytes = path.read_bytes()
        
        # Step 2: Try OCR (extract visible text from image)
        ocr_text = ""
        if self.use_ocr and self._tesseract_available:
            ocr_text = self._run_ocr(path)
        
        # Step 3: Try Vision AI (describe what the image contains)
        ai_description = ""
        if self.openai_client:
            ai_description = self._run_vision_ai(image_bytes, path.name)
        
        # Combine OCR text and AI description into content
        # The agents will use this combined text for search
        combined_content = ""
        if ocr_text:
            combined_content += f"[Text found in image via OCR]:\n{ocr_text}\n\n"
        if ai_description:
            combined_content += f"[AI description of image]:\n{ai_description}"
        
        if not combined_content:
            combined_content = f"Image file: {path.name} (no text extracted)"
        
        return {
            "content": combined_content,
            "source": str(file_path),
            "source_type": "image",
            "file_type": extension,
            "file_name": path.name,
            "ocr_text": ocr_text,
            "ai_description": ai_description,
            "file_size_bytes": path.stat().st_size,
        }
    
    def _run_ocr(self, path: Path) -> str:
        """
        Run Tesseract OCR to extract text from image.
        
        OCR works best with clear, high-contrast images.
        Blurry or low-resolution images may produce poor results.
        """
        try:
            import pytesseract
            from PIL import Image  # pip install Pillow
            
            # Windows: explicitly tell pytesseract where tesseract.exe is
            pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

            # Open image with Pillow
            img = Image.open(path)
            
            # Run OCR
            # lang="eng" means look for English text
            text = pytesseract.image_to_string(img, lang="eng")
            
            return text.strip()
        except Exception as e:
            logger.error(f"OCR failed for {path}: {e}")
            return ""
    
    # def _run_vision_ai(self, image_bytes: bytes, filename: str) -> str:
    #     """
    #     Use OpenAI GPT-4 Vision to describe the image.
        
    #     This converts the image to base64 (text format) and sends it
    #     to the API along with a prompt asking for a description.
        
    #     BEGINNER CONCEPT - What is base64?
    #     Computers store images as bytes (binary data).
    #     APIs only accept text. base64 converts binary -> text safely.
    #     """
    #     try:
    #         # Convert image bytes to base64 string
    #         # base64.b64encode() returns bytes, .decode() turns it to a string
    #         base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
    #         # Determine the MIME type from filename
    #         ext = Path(filename).suffix.lower()
    #         mime_types = {
    #             ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    #             ".png": "image/png", ".gif": "image/gif",
    #             ".webp": "image/webp"
    #         }
    #         mime_type = mime_types.get(ext, "image/png")
            
    #         # Call OpenAI Vision API
    #         response = self.openai_client.chat.completions.create(
    #             model="gpt-4o",
    #             messages=[
    #                 {
    #                     "role": "user",
    #                     "content": [
    #                         {
    #                             "type": "image_url",
    #                             "image_url": {
    #                                 # Data URL format: "data:image/png;base64,<data>"
    #                                 "url": f"data:{mime_type};base64,{base64_image}"
    #                             }
    #                         },
    #                         {
    #                             "type": "text",
    #                             "text": (
    #                                 "Describe this image in detail. "
    #                                 "If it contains charts, tables, or diagrams, "
    #                                 "explain what data or information they show. "
    #                                 "If there is text in the image, transcribe it."
    #                             )
    #                         }
    #                     ]
    #                 }
    #             ],
    #             max_tokens=500
    #         )
            
    #         return response.choices[0].message.content
            
    #     except Exception as e:
    #         logger.error(f"Vision AI failed for {filename}: {e}")
    #         return ""
    def _run_vision_ai(self, image_bytes: bytes, filename: str) -> str:
        """
        Use Mistral's vision model to describe the image.
        Mistral Pixtral supports image input via base64.
        No OpenAI key needed — uses your existing MISTRAL_API_KEY.
        """
        if not self.openai_client:
            return ""

        try:
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            ext = Path(filename).suffix.lower()
            mime_types = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp"
            }
            mime_type = mime_types.get(ext, "image/png")

            # Use Mistral's vision-capable model
            # pixtral-12b-2409 is free tier and supports image input
            response = self.openai_client.chat.completions.create(
                model="pixtral-12b-2409",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Extract ALL information from this image. Do the following:\n"
                                    "1. Transcribe every word of text exactly as written\n"
                                    "2. For any table: write each row as 'Column: Value, Column: Value'\n"
                                    "3. For any chart/graph: describe the title, all axis labels, and every data value\n"
                                    "4. For any diagram/infographic: describe each section and its content\n"
                                    "5. Preserve numbers, percentages, currency values exactly\n"
                                    "Output only the extracted content, no commentary."
                                )
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Vision AI failed for {filename}: {e}")
            return ""
