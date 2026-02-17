"""
NuMarkdown OCR Extractor.
Uses NuMarkdown-8B-Thinking model for advanced document OCR and markdown conversion.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union
import base64
import io

from loguru import logger
from PIL import Image

try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config.settings import settings
from extractors.base import BaseExtractor


class NuMarkdownExtractor(BaseExtractor):
    """
    OCR extractor using NuMarkdown-8B-Thinking model.
    
    NuMarkdown is a vision-language model that converts document images
    to well-structured markdown format.
    """
    
    SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
    
    _model = None
    _processor = None
    _initialized = False
    
    def __init__(self, lazy_load: bool = True):
        """
        Initialize the NuMarkdown extractor.
        
        Args:
            lazy_load: If True, model is loaded on first use. If False, load immediately.
        """
        self.model_name = settings.model_name
        self.device = settings.device
        self.lazy_load = lazy_load
        
        if not lazy_load:
            self._load_model()
    
    def _load_model(self):
        """Load the NuMarkdown model and processor."""
        if self._initialized:
            return
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for NuMarkdown. "
                "Install with: pip install torch transformers accelerate"
            )
        
        logger.info(f"Loading NuMarkdown model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        
        try:
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=settings.hf_token
            )
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "token": settings.hf_token,
            }
            
            if self.device == "cuda" and torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
                logger.info("Using CUDA with float16 precision")
            else:
                self.device = "cpu"
                model_kwargs["torch_dtype"] = torch.float32
                logger.info("Using CPU with float32 precision")
            
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "cpu":
                self._model = self._model.to("cpu")
            
            self._initialized = True
            logger.success("NuMarkdown model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load NuMarkdown model: {e}")
            raise
    
    def supports_format(self, file_extension: str) -> bool:
        """Check if image format is supported."""
        return file_extension.lower() in self.SUPPORTED_FORMATS
    
    def extract(self, file_path: Path) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extract text from image using NuMarkdown OCR.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Tuple of (raw_text, markdown_content, metadata)
        """
        metadata = self.get_file_info(file_path)
        metadata["extraction_method"] = "numarkdown"
        metadata["model"] = self.model_name
        
        # Load and process image
        image = Image.open(file_path).convert("RGB")
        metadata["image_size"] = image.size
        
        # Extract using NuMarkdown
        markdown_content = self.process_image(image)
        
        # Extract raw text from markdown (remove markdown syntax)
        raw_text = self._markdown_to_plain_text(markdown_content)
        
        return raw_text, markdown_content, metadata
    
    def process_image(self, image: Union[Image.Image, Path, str]) -> str:
        """
        Process a single image and convert to markdown.
        
        Args:
            image: PIL Image, file path, or base64 string
            
        Returns:
            Markdown formatted text
        """
        # Ensure model is loaded
        if not self._initialized:
            self._load_model()
        
        # Handle different input types
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        # Prepare prompt for document conversion
        prompt = "Convert this document image to well-structured markdown format. Extract all text content, preserve the layout structure, and identify sections like headers, paragraphs, lists, and tables."
        
        try:
            # Process with the model
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move to appropriate device
            if self.device == "cuda":
                inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
            
            # Generate output
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.9,
                )
            
            # Decode output
            generated_text = self._processor.decode(
                outputs[0], 
                skip_special_tokens=True
            )
            
            # Clean up the output (remove the prompt if echoed)
            if prompt in generated_text:
                generated_text = generated_text.split(prompt)[-1].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error processing image with NuMarkdown: {e}")
            raise
    
    def process_images(self, images: List[Image.Image]) -> str:
        """
        Process multiple images (e.g., PDF pages) and combine results.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Combined markdown content
        """
        markdown_parts = []
        
        for i, image in enumerate(images):
            logger.info(f"Processing page {i + 1}/{len(images)}")
            try:
                page_markdown = self.process_image(image)
                markdown_parts.append(f"<!-- Page {i + 1} -->\n\n{page_markdown}")
            except Exception as e:
                logger.error(f"Error processing page {i + 1}: {e}")
                markdown_parts.append(f"<!-- Page {i + 1} - Error: {str(e)} -->")
        
        return "\n\n---\n\n".join(markdown_parts)
    
    def _markdown_to_plain_text(self, markdown: str) -> str:
        """
        Convert markdown to plain text by removing markdown syntax.
        
        Args:
            markdown: Markdown formatted text
            
        Returns:
            Plain text
        """
        import re
        
        text = markdown
        
        # Remove markdown headers
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        # Remove links but keep text
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        
        # Remove horizontal rules
        text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\*{3,}$', '', text, flags=re.MULTILINE)
        
        # Remove list markers
        text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


class FallbackOCRExtractor(BaseExtractor):
    """
    Fallback OCR extractor using Tesseract.
    Used when NuMarkdown is not available or fails.
    """
    
    SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]
    
    def __init__(self):
        try:
            import pytesseract
            self.pytesseract = pytesseract
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("pytesseract not available for fallback OCR")
    
    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() in self.SUPPORTED_FORMATS and self.available
    
    def extract(self, file_path: Path) -> Tuple[str, str, Dict[str, Any]]:
        """Extract text using Tesseract OCR."""
        metadata = self.get_file_info(file_path)
        metadata["extraction_method"] = "tesseract"
        
        if not self.available:
            raise ImportError("pytesseract is not installed")
        
        image = Image.open(file_path)
        raw_text = self.pytesseract.image_to_string(image)
        
        # Basic markdown conversion
        markdown_content = self._text_to_markdown(raw_text)
        
        return raw_text, markdown_content, metadata
    
    def process_image(self, image: Image.Image) -> str:
        """Process a single image with Tesseract."""
        if not self.available:
            raise ImportError("pytesseract is not installed")
        return self.pytesseract.image_to_string(image)
    
    def _text_to_markdown(self, text: str) -> str:
        """Convert plain text to basic markdown."""
        lines = text.split("\n")
        markdown_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                markdown_lines.append("")
            elif stripped.isupper() and len(stripped) < 50:
                markdown_lines.append(f"## {stripped.title()}")
            else:
                markdown_lines.append(stripped)
        
        return "\n".join(markdown_lines)
