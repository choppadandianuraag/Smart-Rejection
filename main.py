"""
Smart Rejection - Main Application Entry Point
Resume extraction and storage system.
"""

import sys
from pathlib import Path
from typing import List, Optional
import argparse

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/smart_rejection.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)

from config.settings import settings
from extractors.resume_processor import ResumeProcessor, create_processor
from database.models import ResumeCreate
from database.supabase_client import get_resume_repository


class SmartRejectionApp:
    """Main application class for Smart Rejection system."""
    
    def __init__(self, use_numarkdown: bool = True):
        """
        Initialize the application.
        
        Args:
            use_numarkdown: Whether to use NuMarkdown model for OCR
        """
        self.processor = create_processor(use_numarkdown=use_numarkdown)
        self.repository = get_resume_repository()
        
        # Ensure upload directory exists
        settings.resume_upload_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Smart Rejection application initialized")
    
    def process_resume(self, file_path: str | Path) -> dict:
        """
        Process a single resume file.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Processed resume data with database ID
        """
        file_path = Path(file_path)
        logger.info(f"Processing resume: {file_path.name}")
        
        # Extract data from resume
        extracted = self.processor.process_file(file_path)
        
        # Create database entry
        resume_create = ResumeCreate(
            filename=extracted["filename"],
            file_type=extracted["file_type"],
            file_size_bytes=extracted["file_size_bytes"],
            raw_text=extracted["raw_text"],
            markdown_content=extracted["markdown_content"],
            extracted_data=extracted["extracted_data"],
            metadata=extracted["metadata"]
        )
        
        # Store in Supabase
        resume = self.repository.create_sync(resume_create)
        
        logger.success(f"Resume stored with ID: {resume.id}")
        
        return {
            "id": resume.id,
            "filename": resume.filename,
            "status": "completed",
            "extracted_data": resume.extracted_data,
            "word_count": extracted["extracted_data"].get("word_count", 0)
        }
    
    def process_directory(self, directory: str | Path) -> List[dict]:
        """
        Process all resume files in a directory.
        
        Args:
            directory: Path to directory containing resumes
            
        Returns:
            List of processed resume results
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all supported files
        supported_extensions = [".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg"]
        files = []
        
        for ext in supported_extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(files)} resume files in {directory}")
        
        results = []
        for file_path in files:
            try:
                result = self.process_resume(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                results.append({
                    "filename": file_path.name,
                    "status": "failed",
                    "error": str(e)
                })
        
        return results
    
    def get_resume(self, resume_id: str) -> Optional[dict]:
        """
        Retrieve a resume by ID.
        
        Args:
            resume_id: UUID of the resume
            
        Returns:
            Resume data or None if not found
        """
        resume = self.repository.get_by_id(resume_id)
        
        if resume:
            return {
                "id": resume.id,
                "filename": resume.filename,
                "file_type": resume.file_type,
                "raw_text": resume.raw_text,
                "markdown_content": resume.markdown_content,
                "extracted_data": resume.extracted_data,
                "created_at": resume.created_at.isoformat()
            }
        
        return None
    
    def list_resumes(self, limit: int = 50) -> List[dict]:
        """
        List all stored resumes.
        
        Args:
            limit: Maximum number of resumes to return
            
        Returns:
            List of resume summaries
        """
        resumes = self.repository.get_all(limit=limit)
        
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "file_type": r.file_type,
                "status": r.processing_status,
                "created_at": r.created_at.isoformat() if r.created_at else None
            }
            for r in resumes
        ]


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Smart Rejection - Resume Extraction System"
    )
    
    parser.add_argument(
        "action",
        choices=["process", "process-dir", "list", "get"],
        help="Action to perform"
    )
    
    parser.add_argument(
        "--file", "-f",
        help="Path to resume file (for 'process' action)"
    )
    
    parser.add_argument(
        "--directory", "-d",
        help="Path to directory with resumes (for 'process-dir' action)"
    )
    
    parser.add_argument(
        "--id",
        help="Resume ID (for 'get' action)"
    )
    
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=50,
        help="Limit for list action"
    )
    
    parser.add_argument(
        "--no-numarkdown",
        action="store_true",
        help="Disable NuMarkdown and use Tesseract fallback"
    )
    
    args = parser.parse_args()
    
    # Initialize application
    app = SmartRejectionApp(use_numarkdown=not args.no_numarkdown)
    
    try:
        if args.action == "process":
            if not args.file:
                print("Error: --file is required for 'process' action")
                sys.exit(1)
            result = app.process_resume(args.file)
            print(f"✅ Resume processed successfully!")
            print(f"   ID: {result['id']}")
            print(f"   Filename: {result['filename']}")
            print(f"   Word Count: {result['word_count']}")
            
        elif args.action == "process-dir":
            if not args.directory:
                print("Error: --directory is required for 'process-dir' action")
                sys.exit(1)
            results = app.process_directory(args.directory)
            
            successful = sum(1 for r in results if r.get("status") == "completed")
            failed = len(results) - successful
            
            print(f"\n📊 Processing Complete:")
            print(f"   ✅ Successful: {successful}")
            print(f"   ❌ Failed: {failed}")
            
        elif args.action == "list":
            resumes = app.list_resumes(limit=args.limit)
            print(f"\n📄 Stored Resumes ({len(resumes)} total):")
            for resume in resumes:
                print(f"   • {resume['filename']} (ID: {resume['id'][:8]}...)")
                
        elif args.action == "get":
            if not args.id:
                print("Error: --id is required for 'get' action")
                sys.exit(1)
            resume = app.get_resume(args.id)
            if resume:
                print(f"\n📄 Resume Details:")
                print(f"   Filename: {resume['filename']}")
                print(f"   Type: {resume['file_type']}")
                print(f"   Created: {resume['created_at']}")
                print(f"\n📝 Extracted Text (first 500 chars):")
                print(resume['raw_text'][:500])
            else:
                print(f"❌ Resume not found: {args.id}")
                
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
