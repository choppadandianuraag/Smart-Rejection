"""
Test script for enhanced embeddings:
- Preprocessing
- Section-wise embeddings
- Hybrid BERT + TF-IDF embeddings
"""

import sys
from pathlib import Path
from pprint import pprint

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from embeddings.embedding_service import EmbeddingService
from embeddings.preprocessor import ResumePreprocessor


def test_preprocessing():
    """Test text preprocessing."""
    print("\n" + "="*60)
    print("TESTING PREPROCESSING")
    print("="*60)
    
    preprocessor = ResumePreprocessor()
    
    # Sample text with various issues
    sample_text = """
    Sr. Software Eng. at Google Inc.
    Jan 2023 - Present
    
    • Built ML pipelines using Tensorflow and Pytorch
    • Worked with K8s and AWS cloud services
    • Experience with React.js, Node.js, and Postgres
    
    Skills: Python3, JS, TS, CI/CD, REST APIs
    
    Contact: john@example.com | +1 (555) 123-4567
    """
    
    preprocessed = preprocessor.preprocess(sample_text)
    
    print("\nOriginal text:")
    print(sample_text)
    print("\nPreprocessed text:")
    print(preprocessed)
    
    # Extract skills
    skills = preprocessor.extract_skills_list(preprocessed)
    print(f"\nExtracted skills: {skills}")


def test_section_extraction():
    """Test section extraction from resume."""
    print("\n" + "="*60)
    print("TESTING SECTION EXTRACTION")
    print("="*60)
    
    service = EmbeddingService()
    
    # Fetch a resume from database
    resumes = service.fetch_all_resumes()
    
    if not resumes:
        print("No resumes in database. Please add a resume first.")
        return
    
    resume = resumes[0]
    print(f"\nResume: {resume['filename']}")
    
    # Extract sections as dictionary
    sections = service.extract_sections_dict(
        resume['raw_text'],
        resume.get('extracted_data', {})
    )
    
    print(f"\nExtracted sections: {list(sections.keys())}")
    for section_name, section_text in sections.items():
        print(f"\n--- {section_name.upper()} ---")
        print(section_text[:300] + "..." if len(section_text) > 300 else section_text)


def test_section_wise_embedding():
    """Test section-wise BERT embeddings."""
    print("\n" + "="*60)
    print("TESTING SECTION-WISE EMBEDDINGS")
    print("="*60)
    
    service = EmbeddingService()
    
    # Fetch a resume from database
    resumes = service.fetch_all_resumes()
    
    if not resumes:
        print("No resumes in database. Please add a resume first.")
        return
    
    resume = resumes[0]
    print(f"\nGenerating section-wise embeddings for: {resume['filename']}")
    
    result = service.embed_resume_section_wise(resume['id'])
    
    print("\nResult:")
    pprint(result)


def test_hybrid_embedding():
    """Test hybrid BERT + TF-IDF embeddings."""
    print("\n" + "="*60)
    print("TESTING HYBRID EMBEDDINGS")
    print("="*60)
    
    service = EmbeddingService()
    
    # Ensure TF-IDF is fitted
    if not service.embedder.is_fitted:
        print("Fitting TF-IDF model first...")
        service.fit_on_all_resumes()
    
    # Fetch a resume from database
    resumes = service.fetch_all_resumes()
    
    if not resumes:
        print("No resumes in database. Please add a resume first.")
        return
    
    resume = resumes[0]
    print(f"\nGenerating hybrid embeddings for: {resume['filename']}")
    
    result = service.embed_resume_hybrid(resume['id'])
    
    print("\nResult:")
    pprint(result)


def main():
    """Run all tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test enhanced embeddings")
    parser.add_argument("--preprocess", action="store_true", help="Test preprocessing")
    parser.add_argument("--sections", action="store_true", help="Test section extraction")
    parser.add_argument("--section-embed", action="store_true", help="Test section-wise embeddings")
    parser.add_argument("--hybrid", action="store_true", help="Test hybrid embeddings")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Default to all if no args
    if not any([args.preprocess, args.sections, args.section_embed, args.hybrid, args.all]):
        args.all = True
    
    if args.preprocess or args.all:
        test_preprocessing()
    
    if args.sections or args.all:
        test_section_extraction()
    
    if args.section_embed or args.all:
        test_section_wise_embedding()
    
    if args.hybrid or args.all:
        test_hybrid_embedding()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
