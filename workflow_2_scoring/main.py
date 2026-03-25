#!/usr/bin/env python
"""
Workflow 2: Resume Scoring Pipeline CLI
Scores applicants against job descriptions using combined or integrated scoring.

Usage:
    # Score all applicants with COMBINED scoring (Cosine 40% + ATS 60%):
    python main.py --combined

    # Score all applicants against existing job description in DB:
    python main.py

    # Score a SINGLE RESUME (without needing DB):
    python main.py --resume resume.pdf --jd job_description.txt

    # Score with a new job description file:
    python main.py --jd job_description.txt --title "Software Engineer" --company "TechCorp"

    # Score with job description text directly:
    python main.py --jd-text "We are looking for a Python developer..."

    # Limit number of applicants:
    python main.py --limit 10

    # Disable LLM (use only cosine similarity):
    python main.py --no-llm
"""

import argparse
import sys
from pathlib import Path
from uuid import UUID

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "workflow_1_preprocessing"))

from loguru import logger
from integrated_scoring import IntegratedScoringPipeline
from ats_ranking import ATSRankingSystem


def parse_resume_file(file_path: Path) -> str:
    """Parse resume file (PDF, DOCX, or TXT) and return text."""
    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")

    elif suffix == ".pdf":
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            return "\n".join(text_parts)
        except ImportError:
            logger.error("pdfplumber not installed. Run: pip install pdfplumber")
            sys.exit(1)

    elif suffix in [".docx", ".doc"]:
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            logger.error("python-docx not installed. Run: pip install python-docx")
            sys.exit(1)

    else:
        logger.error(f"Unsupported file format: {suffix}. Use PDF, DOCX, or TXT.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Score resumes against job descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Use existing JD from database
  python main.py --jd jd.txt                  # Use JD from file
  python main.py --jd-text "Looking for..."   # Use JD text directly
  python main.py --limit 10                   # Score only 10 applicants
  python main.py --no-llm                     # Disable LLM, use cosine only

  # Score a SINGLE RESUME:
  python main.py --resume resume.pdf --jd jd.txt
  python main.py --resume resume.pdf --jd-text "Python developer needed..."
        """
    )

    parser.add_argument(
        "--resume",
        type=str,
        help="Path to a single resume file (PDF, DOCX, or TXT) to score"
    )
    parser.add_argument(
        "--jd",
        type=str,
        help="Path to job description file (txt or md)"
    )
    parser.add_argument(
        "--jd-text",
        type=str,
        help="Job description text directly"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Software Engineer",
        help="Job title (default: Software Engineer)"
    )
    parser.add_argument(
        "--company",
        type=str,
        default="Company",
        help="Company name (default: Company)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of applicants to score"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM-based ATS scoring (use cosine similarity only)"
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Use COMBINED scoring: Cosine (40%%) + ATS (60%%) with pre-computed data"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="Job UUID to score against (required with --combined)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top candidates to display (default: 15)"
    )
    parser.add_argument(
        "--selected-percentile",
        type=float,
        default=10.0,
        help="Top X%% of candidates are SELECTED (default: 10)"
    )
    parser.add_argument(
        "--borderline-percentile",
        type=float,
        default=40.0,
        help="Next Y%% of candidates get FEEDBACK (default: 40)"
    )

    args = parser.parse_args()

    # Initialize pipeline
    use_llm = not args.no_llm
    pipeline = IntegratedScoringPipeline(use_llm=use_llm)

    # ========================================================================
    # SINGLE RESUME MODE
    # ========================================================================
    if args.resume:
        print("=" * 70)
        print("[*] SINGLE RESUME SCORING MODE")
        print("   Combines: Cosine Similarity (40%) + ATS Score (60%)")
        print("=" * 70)

        # Check resume file exists
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Resume file not found: {args.resume}")
            sys.exit(1)

        # Must have JD for single resume mode
        if not args.jd and not args.jd_text:
            logger.error("Single resume mode requires --jd or --jd-text")
            print("\nExample:")
            print("  python main.py --resume resume.pdf --jd job_description.txt")
            sys.exit(1)

        # Parse resume
        print(f"\n[RESUME] Parsing: {resume_path.name}")
        resume_text = parse_resume_file(resume_path)
        print(f"   Extracted {len(resume_text)} characters")

        # Get job description
        if args.jd:
            jd_path = Path(args.jd)
            if not jd_path.exists():
                logger.error(f"Job description file not found: {args.jd}")
                sys.exit(1)
            job_text = jd_path.read_text(encoding="utf-8")
            print(f"\n[JD] Loaded from: {jd_path.name}")
        else:
            job_text = args.jd_text
            print(f"\n[JD] Using provided text ({len(job_text)} chars)")

        print(f"   Title: {args.title}")
        print(f"   Company: {args.company}")

        # Process JD (don't save to DB for single resume mode)
        print("\n[PROCESSING] Extracting JD requirements...")
        jd_requirements = {}
        if pipeline.use_llm and pipeline.llm:
            jd_requirements = pipeline._extract_jd_requirements_llm(job_text)

        jd_embedding = pipeline.embedder.embed_text(job_text)

        jd_data = {
            "requirements": jd_requirements,
            "embedding": jd_embedding,
            "title": args.title,
            "company": args.company
        }

        # Print JD requirements
        if jd_requirements:
            req = jd_requirements
            print(f"\n[JD REQUIREMENTS]")
            skills = req.get('skills', {})
            print(f"   Must Have: {', '.join(skills.get('must_have', [])[:5])}")
            exp = req.get('experience', {})
            print(f"   Experience: {exp.get('min_years', 0)}-{exp.get('preferred_years', 0)} years")
            edu = req.get('education', {})
            print(f"   Education: {edu.get('min_level', 'any')}")

        # Score single resume
        print("\n[SCORING] Analyzing resume...")
        candidate_name = resume_path.stem.replace("_", " ").replace("-", " ").title()
        result = pipeline.score_single_resume(
            resume_text=resume_text,
            jd_data=jd_data,
            applicant_name=candidate_name
        )

        # Print result
        pipeline.print_single_result(result, jd_title=args.title)

        return result

    # ========================================================================
    # BATCH MODE (All Applicants)
    # ========================================================================
    print("=" * 70)
    print("[*] WORKFLOW 2: INTEGRATED RESUME SCORING")
    print("   Combines: Cosine Similarity (40%) + ATS Score (60%)")
    print("=" * 70)

    # Initialize pipeline
    use_llm = not args.no_llm
    pipeline = IntegratedScoringPipeline(use_llm=use_llm)

    # Get job description
    job_id = None
    jd_data = None

    if args.jd:
        # Read from file
        jd_path = Path(args.jd)
        if not jd_path.exists():
            logger.error(f"Job description file not found: {args.jd}")
            sys.exit(1)

        job_text = jd_path.read_text(encoding="utf-8")
        print(f"\n[JD] Loaded from file: {args.jd}")
        print(f"   Title: {args.title}")
        print(f"   Company: {args.company}")

        job_id, jd_data = pipeline.process_job_description(
            job_text=job_text,
            title=args.title,
            company=args.company
        )

    elif args.jd_text:
        # Use direct text
        print(f"\n[JD] Using provided text")
        print(f"   Title: {args.title}")
        print(f"   Company: {args.company}")

        job_id, jd_data = pipeline.process_job_description(
            job_text=args.jd_text,
            title=args.title,
            company=args.company
        )

    else:
        # Use existing JD from database
        existing_jobs = pipeline.get_job_descriptions()

        if not existing_jobs:
            print("\n[ERROR] No job descriptions found in database!")
            print("   Please provide a job description using --jd or --jd-text")
            print("\n   Examples:")
            print("     python main.py --jd job_description.txt --title 'Data Scientist'")
            print("     python main.py --jd-text 'We need a Python developer...'")
            sys.exit(1)

        jd_record = existing_jobs[0]
        job_id = UUID(jd_record['job_id'])

        print(f"\n[JD] Using existing job description from database:")
        print(f"   Title: {jd_record.get('title', 'N/A')}")
        print(f"   Company: {jd_record.get('company', 'N/A')}")

        # Extract requirements from existing JD
        jd_data = {
            "requirements": pipeline._extract_jd_requirements_llm(jd_record['raw_text']) if pipeline.use_llm else {},
            "embedding": pipeline.embedder.embed_text(jd_record['raw_text']),
            "title": jd_record.get('title'),
            "company": jd_record.get('company')
        }

    # Print JD requirements
    if jd_data and jd_data.get("requirements"):
        req = jd_data["requirements"]
        print(f"\n[JD REQUIREMENTS EXTRACTED]")
        skills = req.get('skills', {})
        print(f"   Skills:")
        print(f"     - Must Have: {len(skills.get('must_have', []))} items")
        if skills.get('must_have'):
            print(f"       {', '.join(skills['must_have'][:5])}...")
        print(f"     - Good to Have: {len(skills.get('good_to_have', []))} items")
        print(f"     - Nice to Have: {len(skills.get('nice_to_have', []))} items")

        exp = req.get('experience', {})
        print(f"   Experience: {exp.get('min_years', 0)}-{exp.get('preferred_years', 0)} years")
        if exp.get('relevant_domains'):
            print(f"     Domains: {', '.join(exp['relevant_domains'][:3])}")

        edu = req.get('education', {})
        print(f"   Education: {edu.get('min_level', 'N/A')} (preferred: {edu.get('preferred_level', 'N/A')})")
        if edu.get('fields'):
            print(f"     Fields: {', '.join(edu['fields'][:3])}")

    # Score applicants
    print("\n[SCORING] Processing applicants...")
    results = pipeline.score_all_applicants(
        job_id=job_id,
        jd_data=jd_data,
        limit=args.limit
    )

    if not results:
        print("\n[WARNING] No applicants found or scored.")
        print("   Make sure you have processed resumes using workflow_1_preprocessing first.")
        sys.exit(0)

    # Print rankings
    pipeline.print_rankings(results, top_k=args.top_k)

    # Zone Classification
    print("\n" + "=" * 70)
    print("[ZONE] CANDIDATE ZONE CLASSIFICATION (Percentile-Based)")
    print(f"   Top {args.selected_percentile}% SELECTED, Next {args.borderline_percentile}% FEEDBACK, Rest REJECTED")
    print("=" * 70)

    classified = pipeline.classify_candidates(
        results,
        selected_percentile=args.selected_percentile,
        borderline_percentile=args.borderline_percentile
    )
    pipeline.print_zone_classification(classified)

    # Save classification to database (update status + create feedback emails)
    print("\n[DB] Saving classification to database...")
    save_stats = pipeline.save_classification_to_db(classified, job_id)
    print(f"   Statuses updated: {save_stats['status_updated']}")
    print(f"   Feedback emails created: {save_stats['feedback_emails_created']}")
    if save_stats['feedback_emails_skipped'] > 0:
        print(f"   Skipped (already exist): {save_stats['feedback_emails_skipped']}")

    # Get borderline candidates for feedback
    borderline = pipeline.get_borderline_candidates(classified)

    # Summary
    print(f"\n[SUMMARY]")
    print(f"   Total Scored: {len(results)}")
    print(f"   Scoring Method: {'LLM + Cosine' if use_llm else 'Cosine Only'}")
    if results:
        avg_score = sum(r['final_score'] for r in results) / len(results)
        print(f"   Average Score: {avg_score:.1f}/100")
        print(f"   Highest Score: {results[0]['final_score']:.1f}/100 ({results[0]['name']})")
        print(f"   Lowest Score: {results[-1]['final_score']:.1f}/100 ({results[-1]['name']})")

    # Zone summary
    summary = classified.get('summary', {})
    print(f"\n[ZONE SUMMARY]")
    print(f"   Selected: {summary.get('selected_count', 0)} candidates (moving to interview)")
    print(f"   Borderline: {summary.get('borderline_count', 0)} candidates (will receive feedback)")
    print(f"   Rejected: {summary.get('rejected_count', 0)} candidates (no action)")

    if borderline:
        print(f"\n[NEXT STEP] {len(borderline)} borderline candidates have pending feedback emails")
        print("   Run workflow 3 to generate email content for these candidates")

    return results, classified


if __name__ == "__main__":
    main()
