#!/usr/bin/env python3
"""
Webhook Server for Feedback RAG Pipeline
=========================================

FastAPI server that generates feedback emails for rejected candidates.

Workflow:
1. Make.com calls this endpoint with applicant_id and job_id
2. Server fetches resume from applicant_profiles.raw_text (Supabase)
3. Server fetches job desc from job_descriptions.description (Supabase)
4. Generates feedback email using LLM
5. Returns email content for Gmail integration
"""

import sys
import os
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from supabase import create_client, Client
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

from .prompts import FEEDBACK_SYSTEM_PROMPT, FEEDBACK_EMAIL_TEMPLATE

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)

# Global clients
supabase_client: Optional[Client] = None
hf_client: Optional[InferenceClient] = None

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize clients on startup."""
    global supabase_client, hf_client

    logger.info("Initializing clients...")

    # Initialize Supabase
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if supabase_url and supabase_key:
        supabase_client = create_client(supabase_url, supabase_key)
        logger.success("Supabase client initialized")
    else:
        logger.error("SUPABASE_URL or SUPABASE_KEY not set!")

    # Initialize HuggingFace client
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_client = InferenceClient(model=LLM_MODEL, token=hf_token.strip().strip("'").strip('"'))
        logger.success(f"HuggingFace client initialized ({LLM_MODEL})")
    else:
        logger.error("HF_TOKEN not set!")

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Feedback RAG Webhook",
    description="Webhook endpoint for generating rejection feedback emails",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request body for feedback generation."""
    applicant_id: str = Field(..., description="Applicant UUID")
    job_id: str = Field(..., description="Job UUID")


class FeedbackResponse(BaseModel):
    """Response with generated email content."""
    status: str
    email_subject: str
    email_body: str
    applicant_email: str
    applicant_name: str
    job_title: str
    feedback_id: Optional[int] = None  # Database ID for tracking
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    supabase_ready: bool
    llm_ready: bool


class BatchRequest(BaseModel):
    """Request for batch processing rejections."""
    job_id: str = Field(..., description="Job UUID")
    threshold: float = Field(0.7, ge=0, le=1, description="Score threshold for rejection")


class EmailResult(BaseModel):
    """Individual email result."""
    applicant_id: str
    applicant_name: str
    applicant_email: str
    email_type: str  # "feedback" or "rejection"
    email_subject: str
    email_body: str
    score: float
    feedback_id: Optional[int] = None  # Database ID for tracking


class BatchResponse(BaseModel):
    """Response for batch processing."""
    status: str
    job_id: str
    job_title: str
    total_rejected: int
    feedback_count: int
    rejection_count: int
    emails: List[EmailResult]
    error: Optional[str] = None


# ============================================================================
# Helper Functions
# ============================================================================

def get_applicant_data(applicant_id: str) -> dict:
    """Fetch applicant data from Supabase."""
    response = supabase_client.table("applicant_profiles").select(
        "name, email, raw_text"
    ).eq("applicant_id", applicant_id).execute()

    if response.data and len(response.data) > 0:
        return response.data[0]
    return None


def get_job_data(job_id: str) -> dict:
    """Fetch job data from Supabase."""
    response = supabase_client.table("job_descriptions").select(
        "title, description"
    ).eq("job_id", job_id).execute()

    if response.data and len(response.data) > 0:
        return response.data[0]
    return None


def get_match_data(applicant_id: str, job_id: str) -> dict:
    """Fetch match score from match_history."""
    response = supabase_client.table("match_history").select(
        "overall_score"
    ).eq("applicant_id", applicant_id).eq("job_id", job_id).execute()

    if response.data and len(response.data) > 0:
        return response.data[0]
    return None


def generate_feedback_email(
    resume_text: str,
    job_description: str,
    applicant_name: str,
    job_title: str,
    match_score: float
) -> str:
    """Generate personalized feedback email using LLM."""

    prompt = FEEDBACK_EMAIL_TEMPLATE.format(
        resume_context=resume_text,
        job_requirements=job_description,
        candidate_name=applicant_name,
        job_title=job_title,
        match_score=int(match_score * 100)
    )

    response = hf_client.chat_completion(
        messages=[
            {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


def generate_simple_rejection(applicant_name: str, job_title: str) -> str:
    """Generate simple rejection email (no LLM needed)."""
    return f"""Dear {applicant_name},

Thank you for applying to the {job_title} position. After careful consideration, we have decided to move forward with other candidates whose qualifications more closely match our current needs. We appreciate your interest and encourage you to apply for future opportunities.

Best regards,
The Hiring Team"""


def get_rejected_applicants(job_id: str, threshold: float) -> List[dict]:
    """Get all rejected applicants for a job, sorted by score DESC."""
    # Query match_history for scores below threshold
    response = supabase_client.table("match_history").select(
        "applicant_id, overall_score"
    ).eq("job_id", job_id).lt("overall_score", threshold).order(
        "overall_score", desc=True
    ).execute()

    return response.data if response.data else []


def save_feedback_to_db(
    applicant_id: str,
    job_id: str,
    subject: str,
    body: str,
    recipient_email: str,
    recipient_name: str,
    match_score: float,
    email_type: str = "feedback"
) -> Optional[int]:
    """Save generated feedback email to database for tracking."""
    try:
        data = {
            "applicant_id": applicant_id,
            "job_id": job_id,
            "subject": subject,
            "body": body,
            "recipient_email": recipient_email,
            "recipient_name": recipient_name,
            "match_score": match_score,
            "llm_model": LLM_MODEL if email_type == "feedback" else None,
            "status": "generated"
        }

        response = supabase_client.table("feedback_emails").insert(data).execute()

        if response.data and len(response.data) > 0:
            feedback_id = response.data[0].get("id")
            logger.info(f"Feedback saved to DB with ID: {feedback_id}")
            return feedback_id
        return None

    except Exception as e:
        logger.error(f"Failed to save feedback to DB: {e}")
        return None


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        supabase_ready=supabase_client is not None,
        llm_ready=hf_client is not None
    )


@app.post("/webhook/feedback", response_model=FeedbackResponse)
async def generate_feedback(request: FeedbackRequest):
    """
    Generate feedback email for rejected candidate.

    Fetches:
    - Resume from applicant_profiles.raw_text
    - Job from job_descriptions.description
    - Score from match_history.overall_score
    """

    if supabase_client is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    if hf_client is None:
        raise HTTPException(status_code=500, detail="LLM client not initialized")

    try:
        # 1. Fetch applicant data
        logger.info(f"Fetching applicant: {request.applicant_id}")
        applicant = get_applicant_data(request.applicant_id)

        if not applicant:
            raise HTTPException(status_code=404, detail=f"Applicant not found: {request.applicant_id}")

        applicant_name = applicant.get("name", "Candidate")
        applicant_email = applicant.get("email", "")
        resume_text = applicant.get("raw_text", "")

        if not resume_text:
            raise HTTPException(status_code=400, detail="Resume text not available")

        # 2. Fetch job data
        logger.info(f"Fetching job: {request.job_id}")
        job = get_job_data(request.job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {request.job_id}")

        job_title = job.get("title", "Open Position")
        job_description = job.get("description", "")

        if not job_description:
            raise HTTPException(status_code=400, detail="Job description not available")

        # 3. Fetch match score from match_history
        logger.info(f"Fetching match score...")
        match_data = get_match_data(request.applicant_id, request.job_id)
        match_score = match_data.get("overall_score", 0.5) if match_data else 0.5

        # 4. Generate feedback email
        logger.info(f"Generating feedback for {applicant_name}")
        logger.info(f"Job: {job_title}, Match Score: {match_score:.2%}")

        email_body = generate_feedback_email(
            resume_text=resume_text,
            job_description=job_description,
            applicant_name=applicant_name,
            job_title=job_title,
            match_score=match_score
        )

        email_subject = f"Application Feedback - {job_title}"

        # Save to database
        feedback_id = save_feedback_to_db(
            applicant_id=request.applicant_id,
            job_id=request.job_id,
            subject=email_subject,
            body=email_body,
            recipient_email=applicant_email,
            recipient_name=applicant_name,
            match_score=match_score,
            email_type="feedback"
        )

        logger.success(f"Feedback generated for {applicant_name} (DB ID: {feedback_id})")

        return FeedbackResponse(
            status="success",
            email_subject=email_subject,
            email_body=email_body,
            applicant_email=applicant_email,
            applicant_name=applicant_name,
            job_title=job_title,
            feedback_id=feedback_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to generate feedback: {e}")
        return FeedbackResponse(
            status="error",
            email_subject="",
            email_body="",
            applicant_email="",
            applicant_name="",
            job_title="",
            error=str(e)
        )


@app.post("/webhook/process-rejections", response_model=BatchResponse)
async def process_rejections(request: BatchRequest):
    """
    Process all rejected applicants for a job.

    - Top 50% of rejected → personalized feedback email
    - Bottom 50% of rejected → simple rejection email
    """

    if supabase_client is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    if hf_client is None:
        raise HTTPException(status_code=500, detail="LLM client not initialized")

    try:
        # 1. Get job data
        logger.info(f"Processing rejections for job: {request.job_id}")
        job = get_job_data(request.job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {request.job_id}")

        job_title = job.get("title", "Open Position")
        job_description = job.get("description", "")

        # 2. Get all rejected applicants sorted by score
        rejected = get_rejected_applicants(request.job_id, request.threshold)
        total_rejected = len(rejected)

        if total_rejected == 0:
            return BatchResponse(
                status="success",
                job_id=request.job_id,
                job_title=job_title,
                total_rejected=0,
                feedback_count=0,
                rejection_count=0,
                emails=[]
            )

        # 3. Split into top 50% and bottom 50%
        midpoint = total_rejected // 2
        top_half = rejected[:midpoint] if midpoint > 0 else rejected
        bottom_half = rejected[midpoint:] if midpoint > 0 else []

        logger.info(f"Total rejected: {total_rejected}, Top 50%: {len(top_half)}, Bottom 50%: {len(bottom_half)}")

        emails = []

        # 4. Generate personalized feedback for top 50%
        for item in top_half:
            applicant_id = item["applicant_id"]
            score = item["overall_score"]

            applicant = get_applicant_data(applicant_id)
            if not applicant:
                continue

            name = applicant.get("name", "Candidate")
            email = applicant.get("email", "")
            resume_text = applicant.get("raw_text", "")

            if resume_text and job_description:
                email_body = generate_feedback_email(
                    resume_text=resume_text,
                    job_description=job_description,
                    applicant_name=name,
                    job_title=job_title,
                    match_score=score
                )
            else:
                email_body = generate_simple_rejection(name, job_title)

            email_subject = f"Application Feedback - {job_title}"

            # Save to database
            feedback_id = save_feedback_to_db(
                applicant_id=applicant_id,
                job_id=request.job_id,
                subject=email_subject,
                body=email_body,
                recipient_email=email,
                recipient_name=name,
                match_score=score,
                email_type="feedback"
            )

            emails.append(EmailResult(
                applicant_id=applicant_id,
                applicant_name=name,
                applicant_email=email,
                email_type="feedback",
                email_subject=email_subject,
                email_body=email_body,
                score=score,
                feedback_id=feedback_id
            ))

        # 5. Generate simple rejection for bottom 50%
        for item in bottom_half:
            applicant_id = item["applicant_id"]
            score = item["overall_score"]

            applicant = get_applicant_data(applicant_id)
            if not applicant:
                continue

            name = applicant.get("name", "Candidate")
            email = applicant.get("email", "")

            email_body = generate_simple_rejection(name, job_title)
            email_subject = f"Update on Your Application - {job_title}"

            # Save to database
            feedback_id = save_feedback_to_db(
                applicant_id=applicant_id,
                job_id=request.job_id,
                subject=email_subject,
                body=email_body,
                recipient_email=email,
                recipient_name=name,
                match_score=score,
                email_type="rejection"
            )

            emails.append(EmailResult(
                applicant_id=applicant_id,
                applicant_name=name,
                applicant_email=email,
                email_type="rejection",
                email_subject=email_subject,
                email_body=email_body,
                score=score,
                feedback_id=feedback_id
            ))

        logger.success(f"Generated {len(emails)} emails for job {job_title}")

        return BatchResponse(
            status="success",
            job_id=request.job_id,
            job_title=job_title,
            total_rejected=total_rejected,
            feedback_count=len(top_half),
            rejection_count=len(bottom_half),
            emails=emails
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to process rejections: {e}")
        return BatchResponse(
            status="error",
            job_id=request.job_id,
            job_title="",
            total_rejected=0,
            feedback_count=0,
            rejection_count=0,
            emails=[],
            error=str(e)
        )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Feedback Webhook",
        "version": "2.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/webhook/feedback": "POST - Generate single feedback email",
            "/webhook/process-rejections": "POST - Process all rejections for a job (top 50% feedback, bottom 50% simple rejection)"
        },
        "docs": "/docs"
    }


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8001))

    logger.info(f"Starting Feedback webhook server on port {port}")

    uvicorn.run(
        "webhook_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
