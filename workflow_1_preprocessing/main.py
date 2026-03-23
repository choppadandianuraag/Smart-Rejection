#!/usr/bin/env python3
"""
Workflow 1 - Resume Preprocessing Pipeline
==========================================

Main entry point for processing resumes through the complete pipeline:
1. Extract text from PDF/DOCX/Image files
2. Segment into sections (contact, skills, experience, education, etc.)
3. Generate BERT embeddings for each section
4. Store in Supabase database (applicant_profiles + applicant_embeddings)

Usage:
    # Process a single resume
    python main.py /path/to/resume.pdf

    # Process multiple resumes
    python main.py /path/to/resumes/*.pdf

    # Process directory of resumes
    python main.py --dir /path/to/resumes/

    # With explicit applicant info
    python main.py resume.pdf --name "John Doe" --email "john@example.com"

    # Disable OCR (faster, for digital PDFs only)
    python main.py resume.pdf --no-ocr
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent and shared directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from loguru import logger

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True
)
logger.add(
    Path(__file__).parent.parent / "logs" / "workflow1_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)

from ingestion_pipeline import SectionAwareResumeProcessor, create_section_aware_processor


def process_single_resume(
    file_path: Path,
    processor: SectionAwareResumeProcessor,
    name: Optional[str] = None,
    email: Optional[str] = None,
    contact_number: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single resume file through the complete pipeline.

    Args:
        file_path: Path to the resume file
        processor: Initialized SectionAwareResumeProcessor
        name: Applicant name (optional, extracted if not provided)
        email: Applicant email (optional, extracted if not provided)
        contact_number: Contact number (optional, extracted if not provided)

    Returns:
        Processing result dictionary
    """
    logger.info(f"Processing: {file_path.name}")

    try:
        result = processor.process_resume(
            file_path=file_path,
            name=name,
            email=email,
            contact_number=contact_number
        )

        logger.success(
            f"✓ {result['name']} ({result['email']}) - "
            f"{result['section_count']} sections, "
            f"confidence: {result['avg_confidence']:.2f}"
        )

        if result.get('needs_review'):
            logger.warning(f"  ⚠ Needs review: {result.get('review_reason')}")

        return result

    except Exception as e:
        logger.error(f"✗ Failed to process {file_path.name}: {e}")
        return {
            "status": "error",
            "filename": file_path.name,
            "error": str(e)
        }


def process_batch(
    files: List[Path],
    use_ocr: bool = True,
    use_numarkdown: bool = False
) -> Dict[str, Any]:
    """
    Process multiple resume files.

    Args:
        files: List of file paths to process
        use_ocr: Enable OCR for scanned documents
        use_numarkdown: Use NuMarkdown model (more accurate but slower)

    Returns:
        Batch processing summary
    """
    logger.info(f"Processing batch of {len(files)} resumes...")
    logger.info(f"OCR: {'enabled' if use_ocr else 'disabled'}, "
                f"Model: {'NuMarkdown' if use_numarkdown else 'Tesseract'}")

    # Initialize processor once for all files
    processor = create_section_aware_processor(
        use_ocr=use_ocr,
        use_numarkdown=use_numarkdown
    )

    results = {
        "total": len(files),
        "successful": 0,
        "failed": 0,
        "needs_review": 0,
        "processed": []
    }

    for i, file_path in enumerate(files, 1):
        logger.info(f"\n[{i}/{len(files)}] Processing {file_path.name}")

        result = process_single_resume(file_path, processor)
        results["processed"].append(result)

        if result.get("status") == "success":
            results["successful"] += 1
            if result.get("needs_review"):
                results["needs_review"] += 1
        else:
            results["failed"] += 1

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total files:      {results['total']}")
    logger.success(f"Successful:       {results['successful']}")
    if results['failed'] > 0:
        logger.error(f"Failed:           {results['failed']}")
    if results['needs_review'] > 0:
        logger.warning(f"Needs review:     {results['needs_review']}")

    return results


def get_resume_files(
    paths: List[str],
    directory: Optional[str] = None
) -> List[Path]:
    """
    Collect resume files from paths and/or directory.

    Args:
        paths: List of file paths or glob patterns
        directory: Directory to scan for resume files

    Returns:
        List of Path objects for resume files
    """
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg', '.webp'}
    files = []

    # Collect from explicit paths
    for path_str in paths:
        path = Path(path_str)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
        elif '*' in path_str:
            # Handle glob patterns
            parent = path.parent if path.parent != path else Path('.')
            pattern = path.name
            files.extend(
                f for f in parent.glob(pattern)
                if f.suffix.lower() in SUPPORTED_EXTENSIONS
            )

    # Collect from directory
    if directory:
        dir_path = Path(directory)
        if dir_path.is_dir():
            for ext in SUPPORTED_EXTENSIONS:
                files.extend(dir_path.glob(f'*{ext}'))
                files.extend(dir_path.glob(f'*{ext.upper()}'))

    # Remove duplicates and sort
    unique_files = sorted(set(files), key=lambda f: f.name)

    return unique_files


def main():
    """Main entry point for workflow 1."""
    parser = argparse.ArgumentParser(
        description="Resume Preprocessing Pipeline - Extract, Segment, Embed, Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py resume.pdf
  python main.py *.pdf --no-ocr
  python main.py --dir ./test_resumes/
  python main.py resume.pdf --name "John Doe" --email "john@example.com"
        """
    )

    # Input files
    parser.add_argument(
        'files',
        nargs='*',
        help='Resume file path(s) to process (PDF, DOCX, PNG, JPG)'
    )
    parser.add_argument(
        '--dir', '-d',
        dest='directory',
        help='Directory containing resume files to process'
    )

    # Applicant info (for single file processing)
    parser.add_argument(
        '--name', '-n',
        help='Applicant name (extracted automatically if not provided)'
    )
    parser.add_argument(
        '--email', '-e',
        help='Applicant email (extracted automatically if not provided)'
    )
    parser.add_argument(
        '--phone', '-p',
        help='Applicant phone number (extracted automatically if not provided)'
    )

    # Processing options
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR (faster, for digital PDFs only)'
    )
    parser.add_argument(
        '--numarkdown',
        action='store_true',
        help='Use NuMarkdown model for OCR (more accurate, slower, requires GPU)'
    )

    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-essential output'
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    elif args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    # Collect files to process
    files = get_resume_files(args.files or [], args.directory)

    if not files:
        logger.error("No resume files found to process!")
        logger.info("Provide file paths or use --dir to specify a directory")
        parser.print_help()
        sys.exit(1)

    logger.info(f"Found {len(files)} resume file(s) to process")

    # Process files
    if len(files) == 1:
        # Single file - can use explicit applicant info
        processor = create_section_aware_processor(
            use_ocr=not args.no_ocr,
            use_numarkdown=args.numarkdown
        )

        result = process_single_resume(
            file_path=files[0],
            processor=processor,
            name=args.name,
            email=args.email,
            contact_number=args.phone
        )

        if result.get("status") == "success":
            print(f"\n✓ Successfully processed: {result['name']}")
            print(f"  Applicant ID: {result['applicant_id']}")
            print(f"  Email: {result['email']}")
            print(f"  Sections: {result['section_count']}")
            print(f"  Confidence: {result['avg_confidence']:.2f}")
        else:
            print(f"\n✗ Processing failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    else:
        # Batch processing
        results = process_batch(
            files=files,
            use_ocr=not args.no_ocr,
            use_numarkdown=args.numarkdown
        )

        if results['failed'] > 0:
            sys.exit(1)

    logger.info("Workflow 1 complete!")


if __name__ == "__main__":
    main()
