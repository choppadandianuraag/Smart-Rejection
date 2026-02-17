"""
Base extractor interface for document processing.
Defines the abstract base class that all extractors must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any


class BaseExtractor(ABC):
    """Abstract base class for document extractors."""
    
    @abstractmethod
    def extract(self, file_path: Path) -> Tuple[str, str, Dict[str, Any]]:
        """
        Extract content from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple containing:
                - raw_text: Plain text extracted from document
                - markdown_content: Structured markdown representation
                - metadata: Additional extraction metadata
        """
        pass
    
    @abstractmethod
    def supports_format(self, file_extension: str) -> bool:
        """
        Check if this extractor supports the given file format.
        
        Args:
            file_extension: File extension (e.g., '.pdf', '.docx')
            
        Returns:
            True if format is supported, False otherwise
        """
        pass
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get basic file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        stat = file_path.stat()
        return {
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "file_path": str(file_path.absolute())
        }
