"""
Process and store a resume in Supabase.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from extractors.resume_processor import create_processor
from database.supabase_client import get_resume_repository
from database.models import ResumeCreate


def main():
    # Resume file path
    pdf_path = Path("uploads/resumes/Anuraag_Choppadhandi_19Jan26 .pdf")
    
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Process resume
    print("Processing resume...")
    processor = create_processor(use_numarkdown=False)
    result = processor.process_file(pdf_path)
    
    print("\n=== RESUME PROCESSING COMPLETE ===")
    print(f"Filename: {result['filename']}")
    print(f"Word Count: {result['extracted_data'].get('word_count', 0)}")
    print(f"Skills Found: {result['extracted_data'].get('skills', [])}")
    print(f"Sections: {result['extracted_data'].get('sections', [])}")
    
    contact = result['extracted_data'].get('contact_info', {})
    print("\n=== CONTACT INFO ===")
    print(f"Email: {contact.get('email')}")
    print(f"Phone: {contact.get('phone')}")
    print(f"LinkedIn: {contact.get('linkedin')}")
    print(f"GitHub: {contact.get('github')}")
    
    # Store in Supabase
    print("\n=== STORING IN SUPABASE ===")
    try:
        repo = get_resume_repository()
        resume_create = ResumeCreate(
            filename=result['filename'],
            file_type=result['file_type'],
            file_size_bytes=result['file_size_bytes'],
            raw_text=result['raw_text'],
            markdown_content=result['markdown_content'],
            extracted_data=result['extracted_data'],
            metadata=result['metadata']
        )
        resume = repo.create_sync(resume_create)
        print(f"SUCCESS! Resume stored with ID: {resume.id}")
    except Exception as e:
        print(f"Error storing in database: {e}")
        print("\nMake sure you've run the SQL schema in Supabase SQL Editor!")


if __name__ == "__main__":
    main()
