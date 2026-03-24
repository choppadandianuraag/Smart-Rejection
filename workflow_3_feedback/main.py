#!/usr/bin/env python3
"""
CLI Entry Point for Feedback RAG Pipeline
==========================================

Command-line interface for testing and running feedback generation.

Usage:
    # Generate feedback for a specific applicant
    python main.py --applicant-id <uuid> --job-id <uuid>

    # Start the webhook server
    python main.py --serve

    # Test with sample data
    python main.py --test
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Feedback RAG Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate feedback for an applicant
    python main.py --applicant-id 123e4567-e89b-12d3-a456-426614174000 \\
                   --job-id 123e4567-e89b-12d3-a456-426614174001 \\
                   --name "John Doe" \\
                   --job-title "Software Engineer"

    # Start webhook server
    python main.py --serve --port 8001

    # Run test with sample data
    python main.py --test
        """
    )

    parser.add_argument(
        "--applicant-id",
        type=str,
        help="Applicant UUID"
    )
    parser.add_argument(
        "--job-id",
        type=str,
        help="Job UUID"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Candidate",
        help="Applicant name (default: Candidate)"
    )
    parser.add_argument(
        "--job-title",
        type=str,
        default="Open Position",
        help="Job title (default: Open Position)"
    )
    parser.add_argument(
        "--match-score",
        type=float,
        default=0.45,
        help="Match score 0-1 (default: 0.45)"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the webhook server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Server port (default: 8001)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for generated email"
    )

    return parser.parse_args()


async def generate_feedback_cli(
    applicant_id: str,
    job_id: str,
    name: str,
    job_title: str,
    match_score: float,
    output_file: str = None
):
    """Generate feedback from CLI."""
    from .feedback_engine import create_feedback_engine

    logger.info("Initializing feedback engine...")
    engine = create_feedback_engine()
    await engine.initialize()

    logger.info(f"Generating feedback for {name}...")
    logger.info(f"  Applicant ID: {applicant_id}")
    logger.info(f"  Job ID: {job_id}")
    logger.info(f"  Job Title: {job_title}")
    logger.info(f"  Match Score: {match_score:.1%}")

    email_content = await engine.generate_feedback_email(
        applicant_id=applicant_id,
        job_id=job_id,
        applicant_name=name,
        job_title=job_title,
        match_score=match_score
    )

    print("\n" + "=" * 60)
    print(f"SUBJECT: Application Feedback - {job_title}")
    print("=" * 60)
    print(email_content)
    print("=" * 60 + "\n")

    if output_file:
        with open(output_file, "w") as f:
            f.write(f"Subject: Application Feedback - {job_title}\n\n")
            f.write(email_content)
        logger.success(f"Email saved to {output_file}")


async def run_test():
    """Run with sample test data."""
    from .feedback_engine import create_feedback_engine
    from .vector_store import ResumeVectorStore

    logger.info("Running test with sample data...")

    # Initialize vector store with test data
    vector_store = ResumeVectorStore(persist_dir="chroma_db/feedback_test")

    # Add Anuraag's resume
    test_applicant_id = "anuraag-choppadandi-001"
    sample_resume = {
        "skills": """
        Programming Languages: Python, C, R
        AI/ML: Scikit-learn, LangChain, RAG Pipelines, Prompt Engineering, HuggingFace Transformers
        Web Development: React, TypeScript, Tailwind CSS, FastAPI
        Databases: PostgreSQL, ChromaDB, Supabase
        Tools: Git/GitHub, Docker, Jupyter Notebooks
        Deployment: Hugging Face Spaces, Netlify
        Soft Skills: Problem-solving, Team Collaboration, Quick Learner
        """,
        "experience": """
        Projects:

        1. FAQ Chatbot using RAG (Nov 2024)
        - Built retrieval-augmented generation chatbot with LangChain
        - Implemented hybrid search (semantic + BM25) with cross-encoder reranking
        - Processed 500+ documents, achieved 85% answer accuracy
        - Deployed on Hugging Face Spaces with FastAPI backend
        Tech: Python, LangChain, ChromaDB, FastAPI, HuggingFace

        2. Cardiovascular Disease Prediction System (Sep 2024)
        - End-to-end ML pipeline: EDA, preprocessing, model training
        - Achieved 76% recall on 70,000+ patient records
        - Real-time predictions via REST API (<200ms response)
        - Dockerized deployment with CI/CD
        Tech: Python, Scikit-learn, FastAPI, Docker

        3. Department Achievement Tracker (Aug 2024)
        - Full-stack web app for tracking academic achievements
        - Built responsive UI with React, TypeScript, Tailwind CSS
        - Integrated Supabase for backend and authentication
        - Currently in active use by department
        Tech: React, TypeScript, Tailwind CSS, Supabase
        """,
        "education": """
        BTech in Artificial Intelligence and Data Science
        VNR VJIET | Expected Graduation: 2027
        CGPA: 8.46

        Certifications:
        - Machine Learning Specialization (Coursera)
        - Large Language Models (NPTEL)
        - Data Analytics Job Simulation (Deloitte)
        """,
        "summary": """
        AI/ML enthusiast and full-stack developer pursuing BTech in AI & Data Science.
        Built and deployed 2 production systems including RAG chatbot and ML prediction APIs.
        Strong foundation in Python, machine learning, and modern web development.
        Passionate about building AI-powered applications and solving real-world problems.
        """
    }

    # Add Software Engineer job description
    test_job_id = "software-engineer-entry-001"
    sample_job = {
        "requirements": """
        Requirements:
        - B.E. / B.Tech in Computer Science, IT, or a related field (2024/2025 pass-out)
        - Solid understanding of data structures, algorithms, and OOP concepts
        - Proficiency in at least one language: Python, Java, JavaScript, or C++
        - Familiarity with Git and basic version control workflows
        - Understanding of REST APIs and how web applications work
        - Good problem-solving skills and ability to learn quickly

        Nice to Have:
        - Exposure to any framework: React, Node.js, Django, Spring, etc.
        - Personal projects, open-source contributions, or internship experience
        - Familiarity with SQL/NoSQL databases
        - Basic knowledge of cloud platforms (AWS, GCP, or Azure)

        Responsibilities:
        - Write clean, maintainable code across frontend, backend, or full-stack
        - Collaborate with senior engineers on feature development, bug fixes, and code reviews
        - Participate in sprint planning, standups, and retrospectives
        - Write unit tests and assist with QA processes
        - Document code and technical decisions clearly
        - Learn and adapt to the team's tech stack, tools, and engineering practices
        """,
        "title": "Software Engineer (Entry-Level)"
    }

    logger.info("Adding resume to vector store...")
    vector_store.add_resume(test_applicant_id, sample_resume)

    logger.info("Adding job to vector store...")
    vector_store.add_job(test_job_id, sample_job)

    # Create engine with test vector store
    engine = create_feedback_engine(vector_store=vector_store)
    await engine.initialize()

    logger.info("Generating feedback...")
    email_content = await engine.generate_feedback_email(
        applicant_id=test_applicant_id,
        job_id=test_job_id,
        applicant_name="Anuraag Choppadandi",
        job_title="Software Engineer (Entry-Level)",
        match_score=0.75
    )

    print("\n" + "=" * 60)
    print("FEEDBACK EMAIL")
    print("=" * 60)
    print(f"Subject: Application Feedback - Software Engineer (Entry-Level)")
    print("-" * 60)
    print(email_content)
    print("=" * 60 + "\n")

    logger.success("Test completed successfully!")


def start_server(port: int):
    """Start the webhook server."""
    import uvicorn

    logger.info(f"Starting Feedback RAG webhook server on port {port}")

    uvicorn.run(
        "workflow_3_feedback.webhook_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Validate HF_TOKEN is set
    if not os.environ.get("HF_TOKEN"):
        logger.error("HF_TOKEN environment variable is not set!")
        logger.info("Please set HF_TOKEN in your .env file or environment")
        sys.exit(1)

    if args.serve:
        start_server(args.port)

    elif args.test:
        asyncio.run(run_test())

    elif args.applicant_id and args.job_id:
        asyncio.run(generate_feedback_cli(
            applicant_id=args.applicant_id,
            job_id=args.job_id,
            name=args.name,
            job_title=args.job_title,
            match_score=args.match_score,
            output_file=args.output
        ))

    else:
        print("Usage:")
        print("  python main.py --test                    # Run with test data")
        print("  python main.py --serve                   # Start webhook server")
        print("  python main.py --applicant-id <id> ...")
        print("\nRun 'python main.py --help' for full options")


if __name__ == "__main__":
    main()
