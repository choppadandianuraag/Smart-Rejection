#!/usr/bin/env python3
"""
Master Webhook Server - Smart Rejection System
===============================================

Unified FastAPI server combining all workflows for easy deployment.

Endpoints:
- Workflow 1 (Preprocessing): /workflow1/process, /workflow1/upload
- Workflow 2 (Scoring): /workflow2/score-job, /workflow2/score-all
- Workflow 3 (Feedback): /workflow3/feedback, /workflow3/process-rejections

Run:
    uvicorn master_webhook_server:app --host 0.0.0.0 --port 8000

Or:
    python master_webhook_server.py
"""

import sys
import os
import tempfile
import base64
import httpx
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from decimal import Decimal

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent / "workflow_1_preprocessing"))
sys.path.insert(0, str(Path(__file__).parent / "workflow_2_scoring"))
sys.path.insert(0, str(Path(__file__).parent / "workflow_3_feedback"))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from supabase import create_client, Client
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

# Import workflow-specific modules
from workflow_1_preprocessing.ingestion_pipeline import SectionAwareResumeProcessor, create_section_aware_processor
from workflow_2_scoring.integrated_scoring import IntegratedScoringPipeline
from workflow_3_feedback.prompts import FEEDBACK_SYSTEM_PROMPT, FEEDBACK_EMAIL_TEMPLATE

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    Path(__file__).parent / "logs" / "master_server_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)

# Global clients
resume_processor: Optional[SectionAwareResumeProcessor] = None
scoring_pipeline: Optional[IntegratedScoringPipeline] = None
supabase_client: Optional[Client] = None
hf_client: Optional[InferenceClient] = None

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all clients on startup."""
    global resume_processor, scoring_pipeline, supabase_client, hf_client

    logger.info("=" * 60)
    logger.info("Initializing Master Webhook Server...")
    logger.info("=" * 60)

    # Initialize Resume Processor (Workflow 1)
    logger.info("Initializing resume processor...")
    try:
        resume_processor = create_section_aware_processor(use_ocr=True, use_numarkdown=False)
        logger.success("✓ Resume processor ready")
    except Exception as e:
        logger.error(f"✗ Resume processor failed: {e}")

    # Initialize Supabase (Workflows 1, 2 & 3)
    logger.info("Initializing Supabase client...")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")

    if supabase_url and supabase_key:
        supabase_client = create_client(supabase_url, supabase_key)
        logger.success("✓ Supabase client ready")
    else:
        logger.error("✗ SUPABASE_URL or SUPABASE_KEY not set!")

    # Initialize Scoring Pipeline (Workflow 2)
    logger.info("Initializing scoring pipeline...")
    try:
        if supabase_client:
            scoring_pipeline = IntegratedScoringPipeline()
            logger.success("✓ Scoring pipeline ready")
        else:
            logger.error("✗ Scoring pipeline requires Supabase")
    except Exception as e:
        logger.error(f"✗ Scoring pipeline failed: {e}")

    # Initialize HuggingFace LLM (Workflow 3)
    logger.info("Initializing HuggingFace client...")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        hf_client = InferenceClient(model=LLM_MODEL, token=hf_token.strip().strip("'").strip('"'))
        logger.success(f"✓ HuggingFace client ready ({LLM_MODEL})")
    else:
        logger.error("✗ HF_TOKEN not set!")

    logger.info("=" * 60)
    logger.success("Master Webhook Server Ready!")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down master server...")


app = FastAPI(
    title="Master Webhook Server - Smart Rejection",
    description="Unified webhook server combining all workflow endpoints",
    version="1.0.0",
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
# SHARED MODELS
# ============================================================================

class GlobalHealthResponse(BaseModel):
    """Global health check response."""
    status: str
    workflow1_ready: bool
    workflow2_ready: bool
    workflow3_ready: bool
    services: dict


# ============================================================================
# WORKFLOW 1 MODELS (PREPROCESSING)
# ============================================================================

class WebhookPayload(BaseModel):
    """Payload from Make.com webhook."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    resume_url: Optional[str] = None
    resume_base64: Optional[str] = None
    filename: Optional[str] = "resume.pdf"


class ProcessingResult(BaseModel):
    """Response after processing."""
    status: str
    applicant_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    section_count: Optional[int] = None
    avg_confidence: Optional[float] = None
    needs_review: Optional[bool] = None
    review_reason: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# WORKFLOW 2 MODELS (SCORING)
# ============================================================================

class ScoringJobRequest(BaseModel):
    """Request to score all applicants for a specific job."""
    job_id: str = Field(..., description="Job UUID")
    use_llm: bool = Field(True, description="Use LLM for enhanced scoring")


class ScoringAllRequest(BaseModel):
    """Request to score all applicants for all jobs."""
    use_llm: bool = Field(True, description="Use LLM for enhanced scoring")
    limit: Optional[int] = Field(None, description="Limit number of applicants to process")


class ScoringApplicantRequest(BaseModel):
    """Request to score a single applicant against all jobs."""
    applicant_id: str = Field(..., description="Applicant UUID")
    use_llm: bool = Field(False, description="Use LLM for enhanced scoring")


class ScoringResult(BaseModel):
    """Individual scoring result."""
    applicant_id: str
    applicant_name: str
    match_score: float
    status: str  # selected, feedback, rejected


class ScoringResponse(BaseModel):
    """Response from scoring operation."""
    status: str
    job_id: Optional[str] = None
    job_title: Optional[str] = None
    applicants_scored: int
    results: Optional[List[ScoringResult]] = None
    status_distribution: Optional[dict] = None
    error: Optional[str] = None


# ============================================================================
# WORKFLOW 3 MODELS (FEEDBACK)
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
    feedback_id: Optional[int] = None
    error: Optional[str] = None


class BatchRequest(BaseModel):
    """Request for batch processing rejections."""
    job_id: str = Field(..., description="Job UUID")
    threshold: float = Field(0.7, ge=0, le=1, description="Score threshold for rejection")


class EmailResult(BaseModel):
    """Individual email result."""
    applicant_id: str
    applicant_name: str
    applicant_email: str
    email_type: str
    email_subject: str
    email_body: str
    score: float
    feedback_id: Optional[int] = None


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
# GLOBAL ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Master Webhook Server - Smart Rejection",
        "version": "1.0.0",
        "workflows": {
            "workflow1": {
                "name": "Resume Preprocessing",
                "endpoints": {
                    "/workflow1/process": "POST - Process resume from URL/base64",
                    "/workflow1/upload": "POST - Process resume from file upload",
                    "/workflow1/health": "GET - Workflow 1 health check"
                }
            },
            "workflow2": {
                "name": "Resume Scoring",
                "endpoints": {
                    "/workflow2/score-applicant": "POST - Score single applicant against all jobs (for Make.com)",
                    "/workflow2/score-job": "POST - Score all applicants for a specific job",
                    "/workflow2/score-all": "POST - Score all applicants for all jobs",
                    "/workflow2/health": "GET - Workflow 2 health check"
                }
            },
            "workflow3": {
                "name": "Feedback Generation",
                "endpoints": {
                    "/workflow3/feedback": "POST - Generate single feedback email",
                    "/workflow3/process-rejections": "POST - Batch process rejections",
                    "/workflow3/health": "GET - Workflow 3 health check"
                }
            }
        },
        "global": {
            "/health": "GET - Global health check (all workflows)",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health", response_model=GlobalHealthResponse)
async def global_health_check():
    """Global health check for all workflows."""
    workflow1_ready = resume_processor is not None
    workflow2_ready = scoring_pipeline is not None
    workflow3_ready = supabase_client is not None and hf_client is not None

    return GlobalHealthResponse(
        status="healthy" if (workflow1_ready and workflow2_ready and workflow3_ready) else "degraded",
        workflow1_ready=workflow1_ready,
        workflow2_ready=workflow2_ready,
        workflow3_ready=workflow3_ready,
        services={
            "resume_processor": resume_processor is not None,
            "scoring_pipeline": scoring_pipeline is not None,
            "supabase": supabase_client is not None,
            "llm": hf_client is not None
        }
    )


# ============================================================================
# WORKFLOW 1 ENDPOINTS (PREPROCESSING)
# ============================================================================

@app.get("/workflow1/health")
async def workflow1_health():
    """Health check for workflow 1."""
    return {
        "status": "healthy" if resume_processor is not None else "unavailable",
        "processor_ready": resume_processor is not None
    }


@app.post("/workflow1/process", response_model=ProcessingResult)
async def process_resume_webhook(payload: WebhookPayload):
    """
    Process resume from URL or base64 (Workflow 1).

    Accepts either:
    - resume_url: URL to download the resume from
    - resume_base64: Base64 encoded resume file
    """
    global resume_processor

    if resume_processor is None:
        raise HTTPException(status_code=503, detail="Resume processor not initialized")

    if not payload.resume_url and not payload.resume_base64:
        raise HTTPException(
            status_code=400,
            detail="Either resume_url or resume_base64 is required"
        )

    temp_file = None

    try:
        # Get resume file
        if payload.resume_url:
            logger.info(f"[WF1] Downloading resume from: {payload.resume_url}")
            temp_file = await download_file(payload.resume_url, payload.filename)
        else:
            logger.info("[WF1] Decoding base64 resume")
            temp_file = decode_base64_file(payload.resume_base64, payload.filename)

        logger.info(f"[WF1] Processing: {temp_file}")

        # Process through pipeline
        result = resume_processor.process_resume(
            file_path=Path(temp_file),
            name=payload.name,
            email=payload.email,
            contact_number=payload.phone
        )

        logger.success(f"[WF1] Processed successfully: {result['applicant_id']}")

        return ProcessingResult(
            status="success",
            applicant_id=result["applicant_id"],
            name=result["name"],
            email=result["email"],
            section_count=result["section_count"],
            avg_confidence=result["avg_confidence"],
            needs_review=result.get("needs_review"),
            review_reason=result.get("review_reason")
        )

    except ValueError as e:
        logger.error(f"[WF1] Validation error: {e}")
        return ProcessingResult(status="error", error=str(e))

    except Exception as e:
        logger.exception(f"[WF1] Processing failed: {e}")
        return ProcessingResult(status="error", error=str(e))

    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@app.post("/workflow1/upload", response_model=ProcessingResult)
async def process_resume_upload(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None)
):
    """
    Process resume from direct file upload (Workflow 1).

    Use this for multipart/form-data submissions.
    """
    global resume_processor

    if resume_processor is None:
        raise HTTPException(status_code=503, detail="Resume processor not initialized")

    # Validate file type
    allowed_extensions = {'.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )

    temp_file = None

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name

        logger.info(f"[WF1] Processing uploaded file: {file.filename}")

        # Process through pipeline
        result = resume_processor.process_resume(
            file_path=Path(temp_file),
            name=name,
            email=email,
            contact_number=phone
        )

        logger.success(f"[WF1] Processed successfully: {result['applicant_id']}")

        return ProcessingResult(
            status="success",
            applicant_id=result["applicant_id"],
            name=result["name"],
            email=result["email"],
            section_count=result["section_count"],
            avg_confidence=result["avg_confidence"],
            needs_review=result.get("needs_review"),
            review_reason=result.get("review_reason")
        )

    except Exception as e:
        logger.exception(f"[WF1] Processing failed: {e}")
        return ProcessingResult(status="error", error=str(e))

    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


# ============================================================================
# WORKFLOW 2 ENDPOINTS (SCORING)
# ============================================================================

@app.get("/workflow2/health")
async def workflow2_health():
    """Health check for workflow 2."""
    return {
        "status": "healthy" if scoring_pipeline is not None else "unavailable",
        "scoring_pipeline_ready": scoring_pipeline is not None
    }


@app.post("/workflow2/score-applicant", response_model=ScoringResponse)
async def score_applicant(request: ScoringApplicantRequest):
    """
    Score a single applicant against all jobs in the database (Workflow 2).

    Perfect for Make.com workflows - only needs applicant_id!
    Automatically finds all jobs and scores the applicant against each one.
    """
    global scoring_pipeline

    if scoring_pipeline is None:
        raise HTTPException(status_code=503, detail="Scoring pipeline not initialized")

    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase client not initialized")

    try:
        from uuid import UUID
        import numpy as np

        applicant_id_uuid = UUID(request.applicant_id)

        logger.info(f"[WF2] Scoring applicant: {request.applicant_id}")

        # Get applicant name
        applicant_response = supabase_client.table("applicant_profiles").select(
            "name"
        ).eq("applicant_id", request.applicant_id).execute()

        if not applicant_response.data or len(applicant_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Applicant not found: {request.applicant_id}")

        applicant_name = applicant_response.data[0]["name"]

        # Get all jobs
        jobs_response = supabase_client.table("job_descriptions").select(
            "job_id, title, company, description, raw_text"
        ).execute()

        if not jobs_response.data or len(jobs_response.data) == 0:
            raise HTTPException(status_code=404, detail="No jobs found in database")

        logger.info(f"[WF2] Found {len(jobs_response.data)} job(s) to score against")

        total_scored = 0
        results = []

        # Score applicant against each job
        for job in jobs_response.data:
            job_id_uuid = UUID(job["job_id"])
            job_title = job["title"]
            job_raw_text = job.get("raw_text") or job.get("description", "")

            logger.info(f"[WF2] Scoring {applicant_name} for {job_title}")

            try:
                # Prepare job data
                jd_data = {
                    "requirements": {},
                    "embedding": scoring_pipeline.embedder.embed_text(job_raw_text),
                    "title": job_title,
                    "company": job.get("company", "")
                }

                # Extract requirements if using LLM
                if request.use_llm and scoring_pipeline.use_llm and scoring_pipeline.llm:
                    jd_data["requirements"] = scoring_pipeline._extract_jd_requirements_llm(job_raw_text)

                # Score all applicants for this job (will score our new applicant too)
                scoring_pipeline.score_all_applicants(
                    job_id=job_id_uuid,
                    jd_data=jd_data,
                    limit=None
                )

                # Get the result for this specific applicant
                result_response = supabase_client.table("match_history").select(
                    "overall_score, status"
                ).eq("applicant_id", request.applicant_id).eq("job_id", job["job_id"]).execute()

                if result_response.data and len(result_response.data) > 0:
                    result = result_response.data[0]

                    results.append(ScoringResult(
                        applicant_id=request.applicant_id,
                        applicant_name=applicant_name,
                        match_score=float(result["overall_score"]),
                        status=result.get("status", "rejected")
                    ))
                    total_scored += 1

                    logger.success(f"[WF2] {applicant_name} scored {result['overall_score']:.2%} for {job_title} - Status: {result.get('status', 'rejected')}")

            except Exception as e:
                logger.error(f"[WF2] Failed to score for job {job_title}: {e}")
                continue

        if total_scored == 0:
            return ScoringResponse(
                status="error",
                applicants_scored=0,
                error="Failed to score applicant for any jobs"
            )

        # Get status distribution
        status_counts = {"selected": 0, "feedback": 0, "rejected": 0}
        for r in results:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1

        logger.success(f"[WF2] Scored {applicant_name} against {total_scored} job(s)")
        logger.info(f"[WF2] Status distribution: {status_counts}")

        # Return results for the first job (or all if needed)
        primary_result = results[0] if results else None

        return ScoringResponse(
            status="success",
            job_id=jobs_response.data[0]["job_id"] if jobs_response.data else None,
            job_title=jobs_response.data[0]["title"] if jobs_response.data else None,
            applicants_scored=1,
            results=results,
            status_distribution=status_counts
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[WF2] Applicant scoring failed: {e}")
        return ScoringResponse(
            status="error",
            applicants_scored=0,
            error=str(e)
        )


@app.post("/workflow2/score-job", response_model=ScoringResponse)
async def score_job(request: ScoringJobRequest):
    """
    Score all applicants for a specific job (Workflow 2).

    Calculates match scores using BERT embeddings and optionally LLM ranking.
    Assigns status (selected/feedback/rejected) based on percentiles.
    """
    global scoring_pipeline

    if scoring_pipeline is None:
        raise HTTPException(status_code=503, detail="Scoring pipeline not initialized")

    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase client not initialized")

    try:
        from uuid import UUID
        import numpy as np

        job_id_uuid = UUID(request.job_id)

        logger.info(f"[WF2] Starting scoring for job: {request.job_id}")

        # Get job details including raw_text
        job_response = supabase_client.table("job_descriptions").select(
            "title, company, description, raw_text"
        ).eq("job_id", request.job_id).execute()

        if not job_response.data or len(job_response.data) == 0:
            raise HTTPException(status_code=404, detail=f"Job not found: {request.job_id}")

        job = job_response.data[0]
        job_title = job["title"]
        job_raw_text = job.get("raw_text") or job.get("description", "")

        # Prepare job data
        jd_data = {
            "requirements": {},
            "embedding": scoring_pipeline.embedder.embed_text(job_raw_text),
            "title": job_title,
            "company": job.get("company", "")
        }

        # Extract requirements if using LLM
        if request.use_llm and scoring_pipeline.use_llm and scoring_pipeline.llm:
            logger.info(f"[WF2] Extracting requirements using LLM")
            jd_data["requirements"] = scoring_pipeline._extract_jd_requirements_llm(job_raw_text)

        # Score all applicants for this job
        logger.info(f"[WF2] Scoring all applicants with LLM={request.use_llm}")
        scoring_pipeline.score_all_applicants(
            job_id=job_id_uuid,
            jd_data=jd_data,
            limit=None
        )

        # Get results
        results_response = supabase_client.table("match_history").select(
            "applicant_id, overall_score, status"
        ).eq("job_id", request.job_id).order("overall_score", desc=True).execute()

        if not results_response.data:
            return ScoringResponse(
                status="success",
                job_id=request.job_id,
                job_title=job_title,
                applicants_scored=0,
                results=[],
                status_distribution={"selected": 0, "feedback": 0, "rejected": 0}
            )

        # Get applicant names
        applicant_ids = [r["applicant_id"] for r in results_response.data]
        applicants_response = supabase_client.table("applicant_profiles").select(
            "applicant_id, name"
        ).in_("applicant_id", applicant_ids).execute()

        name_map = {a["applicant_id"]: a["name"] for a in applicants_response.data}

        # Build results
        results = []
        status_counts = {"selected": 0, "feedback": 0, "rejected": 0}

        for r in results_response.data:
            status = r.get("status", "rejected")
            status_counts[status] = status_counts.get(status, 0) + 1

            results.append(ScoringResult(
                applicant_id=r["applicant_id"],
                applicant_name=name_map.get(r["applicant_id"], "Unknown"),
                match_score=float(r["overall_score"]),
                status=status
            ))

        logger.success(f"[WF2] Scored {len(results)} applicants for {job_title}")
        logger.info(f"[WF2] Status distribution: {status_counts}")

        return ScoringResponse(
            status="success",
            job_id=request.job_id,
            job_title=job_title,
            applicants_scored=len(results),
            results=results[:20],  # Return top 20 for response size
            status_distribution=status_counts
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[WF2] Scoring failed: {e}")
        return ScoringResponse(
            status="error",
            job_id=request.job_id,
            applicants_scored=0,
            error=str(e)
        )


@app.post("/workflow2/score-all", response_model=ScoringResponse)
async def score_all_jobs(request: ScoringAllRequest):
    """
    Score all applicants for all jobs in the database (Workflow 2).

    Calculates match scores for every job-applicant pair.
    Assigns status based on percentiles for each job.
    """
    global scoring_pipeline

    if scoring_pipeline is None:
        raise HTTPException(status_code=503, detail="Scoring pipeline not initialized")

    if supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase client not initialized")

    try:
        import numpy as np

        logger.info(f"[WF2] Starting scoring for ALL jobs (LLM={request.use_llm})")

        # Get all jobs
        jobs_response = supabase_client.table("job_descriptions").select(
            "job_id, title, company, description, raw_text"
        ).execute()

        if not jobs_response.data or len(jobs_response.data) == 0:
            return ScoringResponse(
                status="success",
                applicants_scored=0,
                status_distribution={"selected": 0, "feedback": 0, "rejected": 0}
            )

        total_scored = 0
        all_status_counts = {"selected": 0, "feedback": 0, "rejected": 0}

        # Score each job
        for job in jobs_response.data:
            from uuid import UUID
            job_id_uuid = UUID(job["job_id"])
            job_title = job["title"]
            job_raw_text = job.get("raw_text") or job.get("description", "")

            logger.info(f"[WF2] Scoring job: {job_title} ({job['job_id']})")

            try:
                # Prepare job data
                jd_data = {
                    "requirements": {},
                    "embedding": scoring_pipeline.embedder.embed_text(job_raw_text),
                    "title": job_title,
                    "company": job.get("company", "")
                }

                # Extract requirements if using LLM
                if request.use_llm and scoring_pipeline.use_llm and scoring_pipeline.llm:
                    jd_data["requirements"] = scoring_pipeline._extract_jd_requirements_llm(job_raw_text)

                # Score all applicants for this job
                scoring_pipeline.score_all_applicants(
                    job_id=job_id_uuid,
                    jd_data=jd_data,
                    limit=request.limit
                )

                # Count results for this job
                results_response = supabase_client.table("match_history").select(
                    "status"
                ).eq("job_id", job["job_id"]).execute()

                for r in results_response.data:
                    status = r.get("status", "rejected")
                    all_status_counts[status] = all_status_counts.get(status, 0) + 1
                    total_scored += 1

                logger.success(f"[WF2] Completed scoring for {job_title}")

            except Exception as e:
                logger.error(f"[WF2] Failed to score job {job_title}: {e}")
                continue

        logger.success(f"[WF2] Completed scoring ALL jobs - Total scored: {total_scored}")
        logger.info(f"[WF2] Overall status distribution: {all_status_counts}")

        return ScoringResponse(
            status="success",
            applicants_scored=total_scored,
            status_distribution=all_status_counts
        )

    except Exception as e:
        logger.exception(f"[WF2] Batch scoring failed: {e}")
        return ScoringResponse(
            status="error",
            applicants_scored=0,
            error=str(e)
        )


# ============================================================================
# WORKFLOW 3 ENDPOINTS (FEEDBACK)
# ============================================================================

@app.get("/workflow3/health")
async def workflow3_health():
    """Health check for workflow 3."""
    return {
        "status": "healthy" if (supabase_client and hf_client) else "unavailable",
        "supabase_ready": supabase_client is not None,
        "llm_ready": hf_client is not None
    }


@app.post("/workflow3/feedback", response_model=FeedbackResponse)
async def generate_feedback(request: FeedbackRequest):
    """
    Generate feedback email for rejected candidate (Workflow 3).

    Fetches resume and job description from Supabase,
    generates personalized feedback using LLM.
    """
    if supabase_client is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    if hf_client is None:
        raise HTTPException(status_code=500, detail="LLM client not initialized")

    try:
        # 1. Fetch applicant data
        logger.info(f"[WF3] Fetching applicant: {request.applicant_id}")
        applicant = get_applicant_data(request.applicant_id)

        if not applicant:
            raise HTTPException(status_code=404, detail=f"Applicant not found: {request.applicant_id}")

        applicant_name = applicant.get("name", "Candidate")
        applicant_email = applicant.get("email", "")
        resume_text = applicant.get("raw_text", "")

        if not resume_text:
            raise HTTPException(status_code=400, detail="Resume text not available")

        # 2. Fetch job data
        logger.info(f"[WF3] Fetching job: {request.job_id}")
        job = get_job_data(request.job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {request.job_id}")

        job_title = job.get("title", "Open Position")
        job_description = job.get("description", "")

        if not job_description:
            raise HTTPException(status_code=400, detail="Job description not available")

        # 3. Fetch match score
        logger.info(f"[WF3] Fetching match score...")
        match_data = get_match_data(request.applicant_id, request.job_id)
        match_score = match_data.get("overall_score", 0.5) if match_data else 0.5

        # 4. Generate feedback email
        logger.info(f"[WF3] Generating feedback for {applicant_name}")
        logger.info(f"[WF3] Job: {job_title}, Match Score: {match_score:.2%}")

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

        logger.success(f"[WF3] Feedback generated for {applicant_name} (DB ID: {feedback_id})")

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
        logger.exception(f"[WF3] Failed to generate feedback: {e}")
        return FeedbackResponse(
            status="error",
            email_subject="",
            email_body="",
            applicant_email="",
            applicant_name="",
            job_title="",
            error=str(e)
        )


@app.post("/workflow3/process-rejections", response_model=BatchResponse)
async def process_rejections(request: BatchRequest):
    """
    Process all rejected applicants for a job (Workflow 3).

    - Top 50% of rejected → personalized feedback email
    - Bottom 50% of rejected → simple rejection email
    """
    if supabase_client is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    if hf_client is None:
        raise HTTPException(status_code=500, detail="LLM client not initialized")

    try:
        # 1. Get job data
        logger.info(f"[WF3] Processing rejections for job: {request.job_id}")
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

        logger.info(f"[WF3] Total rejected: {total_rejected}, Top 50%: {len(top_half)}, Bottom 50%: {len(bottom_half)}")

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

        logger.success(f"[WF3] Generated {len(emails)} emails for job {job_title}")

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
        logger.exception(f"[WF3] Failed to process rejections: {e}")
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


# ============================================================================
# HELPER FUNCTIONS - WORKFLOW 1
# ============================================================================

async def download_file(url: str, filename: str) -> str:
    """Download file from URL to temp location."""
    ext = Path(filename).suffix.lower()
    if not ext:
        ext = Path(url.split('?')[0]).suffix.lower()
    if not ext:
        ext = '.pdf'

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(response.content)
            return tmp.name


def decode_base64_file(base64_content: str, filename: str) -> str:
    """Decode base64 content to temp file."""
    ext = Path(filename).suffix.lower() or '.pdf'

    if ',' in base64_content:
        base64_content = base64_content.split(',')[1]

    content = base64.b64decode(base64_content)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        return tmp.name


# ============================================================================
# HELPER FUNCTIONS - WORKFLOW 3
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
    """Generate simple rejection email."""
    return f"""Dear {applicant_name},

Thank you for applying to the {job_title} position. After careful consideration, we have decided to move forward with other candidates whose qualifications more closely match our current needs. We appreciate your interest and encourage you to apply for future opportunities.

Best regards,
The Hiring Team"""


def get_rejected_applicants(job_id: str, threshold: float) -> List[dict]:
    """Get all rejected applicants for a job, sorted by score DESC."""
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
            logger.info(f"[WF3] Feedback saved to DB with ID: {feedback_id}")
            return feedback_id
        return None

    except Exception as e:
        logger.error(f"[WF3] Failed to save feedback to DB: {e}")
        return None


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    logger.info("=" * 60)
    logger.info(f"Starting Master Webhook Server on port {port}")
    logger.info("=" * 60)
    logger.info("Available endpoints:")
    logger.info("  - GET  /")
    logger.info("  - GET  /health")
    logger.info("  - GET  /docs")
    logger.info("")
    logger.info("Workflow 1 (Preprocessing):")
    logger.info("  - POST /workflow1/process")
    logger.info("  - POST /workflow1/upload")
    logger.info("  - GET  /workflow1/health")
    logger.info("")
    logger.info("Workflow 3 (Feedback):")
    logger.info("  - POST /workflow3/feedback")
    logger.info("  - POST /workflow3/process-rejections")
    logger.info("  - GET  /workflow3/health")
    logger.info("=" * 60)

    uvicorn.run(
        "master_webhook_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
