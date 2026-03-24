#!/usr/bin/env python3
"""
Batch Feedback Processor
========================

Generates feedback emails for all applicants with 'feedback' status in match_history.

Usage:
    # Generate feedback for all 'feedback' status applicants
    python batch_feedback.py --job-id <uuid>

    # Generate feedback and save to files
    python batch_feedback.py --job-id <uuid> --output-dir ./feedback_emails/

    # Dry run (show what would be processed)
    python batch_feedback.py --job-id <uuid> --dry-run
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path
from uuid import UUID
from datetime import datetime

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)


def get_feedback_applicants(db_client, job_id: UUID, status: str = "feedback"):
    """
    Get all UNIQUE applicants with specified status from match_history.
    Skips applicants who already have feedback emails generated.

    Args:
        db_client: Supabase client
        job_id: Job UUID
        status: Status to filter by (default: 'feedback')

    Returns:
        List of unique applicant records with match data
    """
    try:
        # Get match history records with specified status
        result = db_client._client.table("match_history").select(
            "id, applicant_id, overall_score, status"
        ).eq("job_id", str(job_id)).eq("status", status).order(
            "overall_score", desc=True
        ).execute()

        if not result.data:
            return []

        # Get applicants who already have feedback emails for this job
        existing_feedback = db_client._client.table("feedback_emails").select(
            "applicant_id"
        ).eq("job_id", str(job_id)).execute()

        existing_applicant_ids = {r["applicant_id"] for r in (existing_feedback.data or [])}

        # Deduplicate by applicant_id - keep highest scoring entry
        seen_applicants = set()
        unique_matches = []
        for match in result.data:
            applicant_id = match["applicant_id"]
            # Skip if already processed or already has feedback
            if applicant_id in seen_applicants:
                continue
            if applicant_id in existing_applicant_ids:
                logger.info(f"  Skipping {applicant_id} - already has feedback email")
                continue
            seen_applicants.add(applicant_id)
            unique_matches.append(match)

        if not unique_matches:
            return []

        # Get applicant profiles for names/emails
        applicant_ids = [r["applicant_id"] for r in unique_matches]

        profiles_result = db_client._client.table("applicant_profiles").select(
            "applicant_id, name, email, raw_text"
        ).in_("applicant_id", applicant_ids).execute()

        # Merge data
        profiles_map = {p["applicant_id"]: p for p in profiles_result.data}

        merged = []
        for match in unique_matches:
            profile = profiles_map.get(match["applicant_id"], {})
            merged.append({
                "match_id": match["id"],
                "applicant_id": match["applicant_id"],
                "name": profile.get("name", "Candidate"),
                "email": profile.get("email", ""),
                "raw_text": profile.get("raw_text", ""),
                "overall_score": match["overall_score"],
                "status": match["status"]
            })

        return merged

    except Exception as e:
        logger.error(f"Error fetching feedback applicants: {e}")
        return []


def get_job_info(db_client, job_id: UUID):
    """Get job title and company from database."""
    try:
        result = db_client._client.table("job_descriptions").select(
            "title, company, raw_text"
        ).eq("job_id", str(job_id)).single().execute()

        return result.data if result.data else {"title": "Open Position", "company": "Company"}
    except Exception as e:
        logger.error(f"Error fetching job info: {e}")
        return {"title": "Open Position", "company": "Company"}


async def generate_batch_feedback(
    job_id: str,
    status: str = "feedback",
    output_dir: str = None,
    dry_run: bool = False,
    limit: int = None
):
    """
    Generate feedback emails for all applicants with specified status.

    Args:
        job_id: Job UUID
        status: Status to filter by
        output_dir: Directory to save feedback emails
        dry_run: If True, only show what would be processed
        limit: Maximum number of applicants to process
    """
    from database.supabase_client_v2 import get_db_client
    from workflow_3_feedback.feedback_engine import create_feedback_engine

    logger.info(f"Starting batch feedback generation for job {job_id}")
    logger.info(f"Status filter: {status}")

    # Get database client
    db_client = get_db_client()

    # Get job info
    job_info = get_job_info(db_client, UUID(job_id))
    job_title = job_info.get("title", "Open Position")
    company = job_info.get("company", "Company")

    logger.info(f"Job: {job_title} at {company}")

    # Get applicants with specified status
    applicants = get_feedback_applicants(db_client, UUID(job_id), status)

    if limit:
        applicants = applicants[:limit]

    if not applicants:
        logger.warning(f"No applicants found with status '{status}'")
        return []

    logger.info(f"Found {len(applicants)} applicants with status '{status}'")

    if dry_run:
        logger.info("\n[DRY RUN] Would process the following applicants:")
        for i, app in enumerate(applicants, 1):
            logger.info(f"  {i}. {app['name']} ({app['email']}) - Score: {app['overall_score']:.4f}")
        return applicants

    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path}")

    # Initialize feedback engine
    logger.info("Initializing feedback engine...")
    engine = create_feedback_engine()
    await engine.initialize()

    # Process each applicant
    results = []
    for i, app in enumerate(applicants, 1):
        logger.info(f"\n[{i}/{len(applicants)}] Generating feedback for {app['name']}...")

        try:
            email_content = await engine.generate_feedback_email(
                applicant_id=app["applicant_id"],
                job_id=job_id,
                applicant_name=app["name"],
                job_title=job_title,
                match_score=float(app["overall_score"]),
                db_client=db_client
            )

            # Save to database
            from database.models_v2 import FeedbackEmailCreate
            from decimal import Decimal

            feedback_record = FeedbackEmailCreate(
                applicant_id=UUID(app["applicant_id"]),
                job_id=UUID(job_id),
                match_history_id=app.get("match_id"),
                subject=f"Application Feedback - {job_title}",
                body=email_content,
                recipient_email=app["email"],
                recipient_name=app["name"],
                match_score=Decimal(str(app["overall_score"])),
                llm_model="Qwen/Qwen2.5-7B-Instruct",
                status="generated"
            )
            saved_feedback = db_client.create_feedback_email(feedback_record)

            result = {
                "applicant_id": app["applicant_id"],
                "name": app["name"],
                "email": app["email"],
                "score": float(app["overall_score"]),
                "feedback": email_content,
                "feedback_id": saved_feedback.id if saved_feedback else None,
                "status": "generated"
            }
            results.append(result)

            # Save to file if output_dir specified
            if output_dir:
                filename = f"{app['name'].replace(' ', '_')}_{app['applicant_id'][:8]}.txt"
                filepath = output_path / filename
                with open(filepath, "w") as f:
                    f.write(f"To: {app['email']}\n")
                    f.write(f"Subject: Application Feedback - {job_title}\n")
                    f.write(f"Score: {app['overall_score']:.2%}\n")
                    f.write("-" * 50 + "\n\n")
                    f.write(email_content)
                logger.success(f"  Saved to {filepath}")

            # Print preview
            preview = email_content[:200].replace('\n', ' ')
            logger.success(f"  Generated & saved to DB: {preview}...")

        except Exception as e:
            logger.error(f"  Error generating feedback for {app['name']}: {e}")
            results.append({
                "applicant_id": app["applicant_id"],
                "name": app["name"],
                "email": app["email"],
                "score": float(app["overall_score"]),
                "feedback": None,
                "status": "error",
                "error": str(e)
            })

    # Summary
    successful = sum(1 for r in results if r["status"] == "generated")
    failed = len(results) - successful

    logger.info("\n" + "=" * 50)
    logger.info("BATCH FEEDBACK SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total processed: {len(results)}")
    logger.success(f"Successful: {successful}")
    if failed > 0:
        logger.error(f"Failed: {failed}")

    return results


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch Feedback Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="Job UUID to process feedback for"
    )
    parser.add_argument(
        "--status",
        type=str,
        default="feedback",
        choices=["selected", "feedback", "rejected"],
        help="Status to filter applicants (default: feedback)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save feedback emails"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without generating"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of applicants to process"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate HF_TOKEN
    if not os.environ.get("HF_TOKEN") and not args.dry_run:
        logger.error("HF_TOKEN environment variable is not set!")
        logger.info("Please set HF_TOKEN in your .env file")
        sys.exit(1)

    results = asyncio.run(generate_batch_feedback(
        job_id=args.job_id,
        status=args.status,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        limit=args.limit
    ))

    if not args.dry_run:
        logger.info(f"\nProcessed {len(results)} applicants")


if __name__ == "__main__":
    main()
