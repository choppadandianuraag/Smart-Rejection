"""
Example script demonstrating resume extraction and storage.
Run this to test the system with sample resumes.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def test_extraction_only():
    """
    Test extraction without database storage.
    Useful for testing when Supabase is not configured.
    """
    from extractors.resume_processor import create_processor
    
    # Create processor (set use_numarkdown=False if no GPU)
    processor = create_processor(use_numarkdown=False)
    
    # Test with a sample file (update path to your test file)
    test_file = Path("uploads/resumes/sample_resume.pdf")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        print("\nTo test, place a resume file at: uploads/resumes/sample_resume.pdf")
        print("Or update the test_file path in this script.")
        return
    
    print(f"📄 Processing: {test_file.name}")
    
    try:
        result = processor.process_file(test_file)
        
        print("\n✅ Extraction successful!")
        print(f"\n📋 File Info:")
        print(f"   Filename: {result['filename']}")
        print(f"   Type: {result['file_type']}")
        print(f"   Size: {result['file_size_bytes']} bytes")
        
        print(f"\n📊 Extracted Data:")
        print(f"   Word Count: {result['extracted_data'].get('word_count', 'N/A')}")
        print(f"   Sections Found: {result['extracted_data'].get('sections', [])}")
        print(f"   Skills Detected: {result['extracted_data'].get('skills', [])}")
        
        contact = result['extracted_data'].get('contact_info', {})
        print(f"\n👤 Contact Info:")
        print(f"   Email: {contact.get('email', 'Not found')}")
        print(f"   Phone: {contact.get('phone', 'Not found')}")
        print(f"   LinkedIn: {contact.get('linkedin', 'Not found')}")
        print(f"   GitHub: {contact.get('github', 'Not found')}")
        
        print(f"\n📝 Raw Text (first 300 chars):")
        print("-" * 50)
        print(result['raw_text'][:300])
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Extraction failed")


def test_full_pipeline():
    """
    Test the full pipeline with database storage.
    Requires Supabase to be configured.
    """
    from main import SmartRejectionApp
    
    print("🚀 Testing Full Pipeline with Supabase Storage")
    print("-" * 50)
    
    try:
        # Initialize app (set use_numarkdown=False if no GPU)
        app = SmartRejectionApp(use_numarkdown=False)
        
        # Test with a sample file
        test_file = Path("uploads/resumes/sample_resume.pdf")
        
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            return
        
        # Process and store
        result = app.process_resume(test_file)
        
        print("\n✅ Resume processed and stored!")
        print(f"   ID: {result['id']}")
        print(f"   Filename: {result['filename']}")
        print(f"   Word Count: {result['word_count']}")
        
        # Retrieve from database
        print("\n📥 Retrieving from database...")
        stored = app.get_resume(result['id'])
        
        if stored:
            print("✅ Successfully retrieved from Supabase!")
            print(f"   Created: {stored['created_at']}")
        else:
            print("❌ Failed to retrieve from database")
        
        # List all resumes
        print("\n📋 All stored resumes:")
        resumes = app.list_resumes(limit=5)
        for r in resumes:
            print(f"   • {r['filename']} ({r['status']})")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Pipeline failed")


def create_sample_resume():
    """Create a sample text file for testing."""
    sample_content = """
JOHN DOE
Software Engineer

Email: john.doe@email.com
Phone: (555) 123-4567
LinkedIn: linkedin.com/in/johndoe
GitHub: github.com/johndoe

SUMMARY
Experienced software engineer with 5+ years of experience in Python, 
JavaScript, and cloud technologies. Passionate about building scalable 
applications and solving complex problems.

EXPERIENCE

Senior Software Engineer | TechCorp Inc.
2020 - Present
- Led development of microservices architecture using Python and Docker
- Implemented CI/CD pipelines reducing deployment time by 50%
- Mentored junior developers and conducted code reviews

Software Developer | StartupXYZ
2018 - 2020
- Built RESTful APIs using Django and Flask
- Developed frontend applications with React and TypeScript
- Managed AWS infrastructure including EC2, S3, and Lambda

EDUCATION

Bachelor of Science in Computer Science
State University | 2014 - 2018
GPA: 3.8/4.0

SKILLS

Technical: Python, JavaScript, TypeScript, React, Django, Flask, Docker, 
Kubernetes, AWS, PostgreSQL, MongoDB, Git, Linux

Soft Skills: Team Leadership, Communication, Problem Solving, Agile

CERTIFICATIONS

- AWS Certified Solutions Architect
- Google Cloud Professional Developer

PROJECTS

Open Source Contribution
- Contributed to popular Python libraries with 500+ GitHub stars
- Maintained documentation and reviewed pull requests
"""
    
    # Create uploads directory if needed
    upload_dir = Path("uploads/resumes")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample resume
    sample_file = upload_dir / "sample_resume.txt"
    sample_file.write_text(sample_content)
    
    print(f"✅ Created sample resume at: {sample_file}")
    return sample_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Smart Rejection System")
    parser.add_argument(
        "--mode",
        choices=["extraction", "full", "create-sample"],
        default="extraction",
        help="Test mode: extraction (no DB), full (with DB), or create-sample"
    )
    
    args = parser.parse_args()
    
    print("🎯 Smart Rejection - Test Script")
    print("=" * 50)
    
    if args.mode == "create-sample":
        create_sample_resume()
    elif args.mode == "extraction":
        test_extraction_only()
    elif args.mode == "full":
        test_full_pipeline()
