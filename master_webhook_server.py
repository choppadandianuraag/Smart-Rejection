#!/usr/bin/env python3
"""
Master Webhook Server - Smart Rejection System
===============================================

Unified FastAPI server combining all workflows for easy deployment.

Endpoints:
- Workflow 1 (Preprocessing): /workflow1/process, /workflow1/upload
- Workflow 2 (Scoring): /workflow2/score-job, /workflow2/score-all
- Workflow 3 (Feedback): /workflow3/feedback, /workflow3/sync-feedback-emails, /workflow3/generate-pending-feedbacks

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
from datetime import datetime
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


def normalize_match_status(status: Optional[str]) -> str:
    """Normalize DB status values to supported response statuses."""
    normalized = (status or "rejected").lower()
    if normalized not in {"selected", "feedback", "rejected"}:
        return "rejected"
    return normalized


# ============================================================================
# WORKFLOW 3 MODELS (FEEDBACK)
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request body for feedback generation."""
    applicant_id: str = Field(..., description="Applicant UUID")
    job_id: Optional[str] = Field(None, description="Job UUID (optional; auto-resolved if missing)")


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
                # Use IntegratedScoringPipeline (integrated_v2) - WITH DEBUGGING
                logger.info(f"[DEBUG] Initializing IntegratedScoringPipeline with use_llm={request.use_llm}")
                integrated_pipeline = IntegratedScoringPipeline(use_llm=request.use_llm)
                logger.info(f"[DEBUG] IntegratedScoringPipeline initialized successfully")

                # Prepare job data for integrated scoring
                jd_data = {
                    "embedding": None,  # Will be loaded from job_embeddings table
                    "requirements": None,  # Will be extracted using LLM below
                    "title": job_title,
                    "company": job.get("company", ""),
                    "raw_text": job_raw_text
                }
                logger.info(f"[DEBUG] Prepared job data for {job_title}")

                # Get job embedding from database
                job_embedding_response = supabase_client.table("job_embeddings").select(
                    "embedding_vector"  # FIXED: Use correct column name
                ).eq("job_id", job["job_id"]).execute()

                if job_embedding_response.data and len(job_embedding_response.data) > 0:
                    embedding_data = job_embedding_response.data[0]["embedding_vector"]
                    # FIX: Convert string embeddings to list format
                    if isinstance(embedding_data, str):
                        # If stored as string, try to parse as JSON
                        try:
                            import json
                            embedding_list = json.loads(embedding_data)
                            jd_data["embedding"] = np.array(embedding_list)
                            logger.info(f"[DEBUG] Converted string embedding to array: shape {np.array(embedding_list).shape}")
                        except:
                            logger.warning(f"[DEBUG] Failed to parse string embedding for job {job['job_id']}")
                            jd_data["embedding"] = None
                    elif isinstance(embedding_data, list):
                        jd_data["embedding"] = np.array(embedding_data)
                        logger.info(f"[DEBUG] Loaded job embedding for {job['job_id']}: shape {np.array(embedding_data).shape}")
                    else:
                        logger.warning(f"[DEBUG] Invalid embedding format for job {job['job_id']}: type={type(embedding_data)}")
                        jd_data["embedding"] = None
                else:
                    logger.warning(f"[DEBUG] No job embedding found for job {job['job_id']}")
                    jd_data["embedding"] = None

                # Extract JD requirements using LLM (CRITICAL: needed for proper ATS scoring)
                logger.info(f"[DEBUG] Extracting JD requirements for proper scoring...")
                if integrated_pipeline.use_llm and integrated_pipeline.llm:
                    jd_requirements = integrated_pipeline._extract_jd_requirements_llm(job_raw_text)
                    jd_data["requirements"] = jd_requirements
                    logger.info(f"[DEBUG] Extracted JD requirements: skills={len(jd_requirements.get('skills', {}).get('must_have', []))} must-have")
                else:
                    # Fallback: empty requirements (will result in lower accuracy)
                    jd_data["requirements"] = {
                        "skills": {"must_have": [], "good_to_have": [], "nice_to_have": []},
                        "experience": {"min_years": 0, "preferred_years": 0, "relevant_domains": []},
                        "education": {"min_level": "bachelor", "preferred_level": "bachelor", "fields": []}
                    }
                    logger.warning(f"[DEBUG] LLM not available - using empty JD requirements")

                # OPTIMIZED: Score only the target applicant instead of all 51
                logger.info(f"[DEBUG] Scoring ONLY target applicant: {request.applicant_id}")

                # Get the specific applicant from database
                from database.supabase_client_v2 import get_db_client
                db = get_db_client()

                # Get target applicant details
                target_applicant = None
                applicants = db.get_all_applicants(limit=None)
                for applicant in applicants:
                    if str(applicant.applicant_id) == str(request.applicant_id):
                        target_applicant = applicant
                        break

                if not target_applicant:
                    logger.warning(f"[DEBUG] Target applicant {request.applicant_id} not found in database")
                    continue

                logger.info(f"[DEBUG] Found target applicant: {target_applicant.name}")

                # Get applicant embedding directly from database
                applicant_emb = None
                applicant_emb_response = supabase_client.table("applicant_embeddings").select(
                    "resume_embedding"
                ).eq("applicant_id", target_applicant.applicant_id).execute()

                if applicant_emb_response.data and len(applicant_emb_response.data) > 0:
                    embedding_raw = applicant_emb_response.data[0]["resume_embedding"]
                    # Handle different embedding formats (string vs list)
                    if isinstance(embedding_raw, str):
                        try:
                            import json
                            embedding_list = json.loads(embedding_raw)
                            applicant_emb = np.array(embedding_list)
                            logger.info(f"[DEBUG] Converted applicant string embedding to array: shape {applicant_emb.shape}")
                        except:
                            logger.warning(f"[DEBUG] Failed to parse applicant string embedding")
                            applicant_emb = None
                    elif isinstance(embedding_raw, list):
                        applicant_emb = np.array(embedding_raw)
                        logger.info(f"[DEBUG] Loaded applicant embedding array: shape {applicant_emb.shape}")
                    else:
                        logger.warning(f"[DEBUG] Invalid applicant embedding format: type={type(embedding_raw)}")
                        applicant_emb = None
                else:
                    logger.warning(f"[DEBUG] No applicant embedding found for {target_applicant.applicant_id}")

                # Calculate cosine similarity
                jd_embedding = jd_data.get("embedding")
                cosine_score = 0.0
                if applicant_emb is not None and jd_embedding is not None:
                    cosine_score = integrated_pipeline._cosine_similarity(
                        applicant_emb.reshape(1, -1),
                        jd_embedding.reshape(1, -1)
                    )
                    logger.info(f"[DEBUG] Cosine similarity: {cosine_score:.4f}")
                else:
                    logger.warning(f"[DEBUG] Missing embeddings - applicant: {applicant_emb is not None}, job: {jd_embedding is not None}")

                # Calculate ATS score using groq LLM
                ats_score = 0.0
                skills_score = 0.0
                experience_score = 0.0
                education_score = 0.0

                if integrated_pipeline.use_llm:
                    logger.info(f"[DEBUG] Calculating ATS score with groq LLM...")
                    # Get raw resume text from applicant_profiles for LLM processing
                    resume_response = supabase_client.table("applicant_profiles").select(
                        "raw_text"
                    ).eq("applicant_id", target_applicant.applicant_id).execute()

                    if resume_response.data and len(resume_response.data) > 0:
                        raw_resume_text = resume_response.data[0].get("raw_text", "")

                        if raw_resume_text:
                            # FIXED: Pass raw text string instead of parsed dictionary
                            jd_requirements = jd_data.get("requirements", {})
                            ats_result = integrated_pipeline._calculate_ats_score(raw_resume_text, jd_requirements)

                            ats_score = ats_result.get("ats_score", 0.0) / 100  # Convert from 0-100 to 0-1 scale
                            skills_score = ats_result.get("component_scores", {}).get("skills", 0.0) / 100
                            experience_score = ats_result.get("component_scores", {}).get("experience", 0.0) / 100
                            education_score = ats_result.get("component_scores", {}).get("education", 0.0) / 100

                            logger.info(f"[DEBUG] ATS score: {ats_score:.4f} (from {ats_result.get('ats_score', 0)}/100)")
                        else:
                            logger.warning(f"[DEBUG] No raw_text found for applicant {target_applicant.applicant_id}")
                    else:
                        logger.warning(f"[DEBUG] No profile data found for applicant {target_applicant.applicant_id}")

                # Calculate final combined score
                weights = integrated_pipeline.SCORE_COMBINATION_WEIGHTS
                final_score = (cosine_score * weights["cosine_similarity"] +
                              ats_score * weights["ats_score"]) * 100

                logger.info(f"[DEBUG] Final combined score: {final_score:.4f}")

                # Determine zone/status (FIX: Use database enum values)
                if final_score >= 75:
                    zone = "selected"
                elif final_score >= 50:
                    zone = "feedback"
                else:
                    zone = "rejected"

                # Create result for single applicant
                applicant_result = {
                    "applicant_id": str(target_applicant.applicant_id),
                    "name": target_applicant.name,
                    "final_score": final_score,
                    "cosine_score": cosine_score,
                    "ats_score": ats_score,
                    "skills": skills_score,
                    "experience": experience_score,
                    "education": education_score,
                    "zone": zone
                }

                logger.info(f"[DEBUG] Created result: {applicant_result}")

                if applicant_result:
                    # Convert integrated_v2 result to match_history format
                    final_score = applicant_result.get("final_score", 0)
                    logger.info(f"[DEBUG] Processing result with final_score: {final_score}")

                    # Ensure proper score scaling (0-1 for overall_score)
                    if final_score > 1:
                        overall_score = round(final_score / 100, 4)
                    else:
                        overall_score = round(final_score, 4)

                    match_data = {
                        "applicant_id": request.applicant_id,
                        "job_id": job["job_id"],
                        "overall_score": overall_score,
                        "section_scores": {
                            "final_score": final_score,
                            "cosine_score": applicant_result.get("cosine_score", 0),
                            "ats_score": applicant_result.get("ats_score", 0),
                            "skills": applicant_result.get("skills", 0),
                            "experience": applicant_result.get("experience", 0),
                            "education": applicant_result.get("education", 0)
                        },
                        "config_name": "integrated_v2",
                        "weights_used": {
                            "ats_score": 0.6,
                            "ats_components": {
                                "skills": 0.6,
                                "education": 0.15,
                                "experience": 0.25
                            },
                            "cosine_similarity": 0.4
                        },
                        "status": str(applicant_result.get("zone", "rejected")).lower()
                    }

                    # Save to match_history
                    logger.info(f"[DEBUG] Saving match_data: {match_data}")
                    supabase_client.table('match_history').insert(match_data).execute()

                    # Add to results
                    normalized_status = match_data["status"]
                    results.append(ScoringResult(
                        applicant_id=request.applicant_id,
                        applicant_name=applicant_name,
                        match_score=float(overall_score),
                        status=normalized_status
                    ))
                    total_scored += 1

                    logger.success(f"[WF2] {applicant_name} scored {final_score:.1f} for {job_title} - Status: {normalized_status}")
                else:
                    logger.warning(f"[WF2] Applicant {request.applicant_id} not found in {len(all_rankings)} scoring results for {job_title}")

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
            normalized_status = normalize_match_status(r.status)
            status_counts[normalized_status] = status_counts.get(normalized_status, 0) + 1

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
            status = normalize_match_status(r.get("status"))
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
                results = scoring_pipeline.score_all_applicants(
                    job_id=job_id_uuid,
                    jd_data=jd_data,
                    limit=request.limit
                )

                # IMPORTANT: Classify candidates into zones (percentile-based)
                if results:
                    logger.info(f"[WF2] Classifying {len(results)} candidates into zones...")
                    classified = scoring_pipeline.classify_candidates(results)

                    # Save classification to database (status updates + feedback emails)
                    save_stats = scoring_pipeline.save_classification_to_db(classified, job_id_uuid)

                    total_scored += save_stats["status_updated"]

                    # Update overall status counts
                    summary = classified.get("summary", {})
                    all_status_counts["selected"] += summary.get("selected_count", 0)
                    all_status_counts["feedback"] += summary.get("borderline_count", 0)
                    all_status_counts["rejected"] += summary.get("rejected_count", 0)

                    logger.info(f"[WF2] Zones: {summary.get('selected_count', 0)} selected, "
                               f"{summary.get('borderline_count', 0)} feedback, "
                               f"{summary.get('rejected_count', 0)} rejected")
                    logger.info(f"[WF2] Feedback emails created: {save_stats['feedback_emails_created']}")

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
        resolved_job_id = request.job_id

        # Backward-compatible behavior for Make.com payloads that omit or send null job_id.
        if not resolved_job_id:
            logger.warning(f"[WF3] No job_id provided for applicant {request.applicant_id}; resolving latest match")
            resolved_job_id = get_latest_job_id_for_applicant(request.applicant_id)

            if not resolved_job_id:
                raise HTTPException(
                    status_code=400,
                    detail="job_id is required when no match_history exists for this applicant"
                )

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
        logger.info(f"[WF3] Fetching job: {resolved_job_id}")
        job = get_job_data(resolved_job_id)

        if not job:
            raise HTTPException(status_code=404, detail=f"Job not found: {resolved_job_id}")

        job_title = job.get("title", "Open Position")
        job_description = job.get("description", "")

        if not job_description:
            raise HTTPException(status_code=400, detail="Job description not available")

        # 3. Fetch match score
        logger.info(f"[WF3] Fetching match score...")
        match_data = get_match_data(request.applicant_id, resolved_job_id)
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
            job_id=resolved_job_id,
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


@app.get("/workflow3/feedback-stats")
async def feedback_stats():
    """Get statistics about feedback_emails table."""
    if supabase_client is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    try:
        # Get all feedback emails
        all_result = supabase_client.table("feedback_emails").select(
            "id, status, subject, body"
        ).execute()

        total = len(all_result.data)

        # Count by status
        status_counts = {}
        sample_records = []

        for record in all_result.data:
            status = record.get("status", "NULL")
            status_counts[status] = status_counts.get(status, 0) + 1

            # Get sample of first 5 records
            if len(sample_records) < 5:
                sample_records.append({
                    "id": record["id"],
                    "status": status,
                    "subject": record.get("subject", "")[:50],
                    "body": record.get("body", "")[:100]
                })

        # Also check match_history for status distribution
        match_result = supabase_client.table("match_history").select(
            "id, status"
        ).execute()

        match_status_counts = {}
        for record in match_result.data:
            status = record.get("status", "NULL")
            match_status_counts[status] = match_status_counts.get(status, 0) + 1

        return {
            "feedback_emails": {
                "total_records": total,
                "status_breakdown": status_counts,
                "sample_records": sample_records
            },
            "match_history": {
                "total_records": len(match_result.data),
                "status_breakdown": match_status_counts
            }
        }

    except Exception as e:
        logger.exception(f"Failed to get feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow3/sync-feedback-emails")
async def sync_feedback_emails():
    """
    Backfill feedback_emails table from match_history records with status='feedback'.

    This creates pending feedback_email records for candidates who were classified
    as 'feedback' but don't have email records yet.
    """
    if supabase_client is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    try:
        logger.info("[WF3] Syncing feedback_emails from match_history...")

        # 1. Get all match_history records with status='feedback'
        feedback_matches = supabase_client.table("match_history").select(
            "id, applicant_id, job_id, overall_score"
        ).eq("status", "feedback").execute()

        if not feedback_matches.data:
            return {
                "status": "success",
                "message": "No feedback candidates found in match_history",
                "created": 0,
                "skipped": 0
            }

        logger.info(f"[WF3] Found {len(feedback_matches.data)} feedback candidates")

        created = 0
        skipped = 0

        for match in feedback_matches.data:
            applicant_id = match["applicant_id"]
            job_id = match["job_id"]
            match_history_id = match["id"]
            match_score = float(match.get("overall_score", 0.5))

            # Check if feedback email already exists
            existing = supabase_client.table("feedback_emails").select("id").eq(
                "applicant_id", applicant_id
            ).eq("job_id", job_id).execute()

            if existing.data:
                skipped += 1
                continue

            # Get applicant info
            applicant = supabase_client.table("applicant_profiles").select(
                "name, email"
            ).eq("applicant_id", applicant_id).execute()

            if not applicant.data:
                logger.warning(f"[WF3] Applicant not found: {applicant_id}")
                skipped += 1
                continue

            applicant_name = applicant.data[0].get("name", "Candidate")
            applicant_email = applicant.data[0].get("email", "")

            # Create feedback_emails record
            supabase_client.table("feedback_emails").insert({
                "applicant_id": applicant_id,
                "job_id": job_id,
                "match_history_id": match_history_id,
                "subject": "Pending Generation",
                "body": "Pending Generation",
                "recipient_email": applicant_email,
                "recipient_name": applicant_name,
                "match_score": match_score,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }).execute()

            created += 1
            logger.info(f"[WF3] Created feedback_email for {applicant_name}")

        logger.success(f"[WF3] Sync complete: {created} created, {skipped} skipped")

        return {
            "status": "success",
            "created": created,
            "skipped": skipped
        }

    except Exception as e:
        logger.exception(f"[WF3] Failed to sync feedback_emails: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workflow3/generate-pending-feedbacks")
async def generate_pending_feedbacks(limit: int = 100):
    """
    Generate feedback email content for all pending feedback_emails records.

    This processes candidates who were classified as 'feedback' (borderline)
    and have pending email generation.

    Args:
        limit: Maximum number of pending emails to process (default: 100)

    Returns:
        Summary of generated feedbacks
    """
    if supabase_client is None:
        raise HTTPException(status_code=500, detail="Supabase client not initialized")

    if hf_client is None:
        raise HTTPException(status_code=500, detail="LLM client not initialized")

    try:
        logger.info(f"[WF3] Fetching pending feedback emails (limit: {limit})...")

        # 1. Get all pending feedback emails
        pending_result = supabase_client.table("feedback_emails").select(
            "id, applicant_id, job_id, recipient_email, recipient_name, match_score"
        ).eq("status", "pending").limit(limit).execute()

        if not pending_result.data:
            return {
                "status": "success",
                "message": "No pending feedback emails found",
                "processed": 0,
                "failed": 0
            }

        pending_emails = pending_result.data
        logger.info(f"[WF3] Found {len(pending_emails)} pending feedback emails")

        processed = 0
        failed = 0
        results = []

        for email_record in pending_emails:
            try:
                email_id = email_record["id"]
                applicant_id = email_record["applicant_id"]
                job_id = email_record["job_id"]
                recipient_name = email_record.get("recipient_name", "Candidate")
                match_score = float(email_record.get("match_score", 0.5))

                # 2. Get applicant data
                applicant = get_applicant_data(applicant_id)
                if not applicant:
                    logger.warning(f"[WF3] Applicant not found: {applicant_id}")
                    failed += 1
                    continue

                resume_text = applicant.get("raw_text", "")
                if not resume_text:
                    logger.warning(f"[WF3] No resume text for applicant: {applicant_id}")
                    failed += 1
                    continue

                # 3. Get job data
                job = get_job_data(job_id)
                if not job:
                    logger.warning(f"[WF3] Job not found: {job_id}")
                    failed += 1
                    continue

                job_title = job.get("title", "Open Position")
                job_description = job.get("description", "")

                # 4. Generate feedback email
                logger.info(f"[WF3] Generating feedback for {recipient_name} ({email_id})...")

                email_body = generate_feedback_email(
                    resume_text=resume_text,
                    job_description=job_description,
                    applicant_name=recipient_name,
                    job_title=job_title,
                    match_score=match_score
                )

                email_subject = f"Application Feedback - {job_title}"

                # 5. Update the feedback_emails record
                supabase_client.table("feedback_emails").update({
                    "subject": email_subject,
                    "body": email_body,
                    "status": "generated",
                    "llm_model": LLM_MODEL,
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", email_id).execute()

                processed += 1
                results.append({
                    "id": email_id,
                    "applicant_name": recipient_name,
                    "job_title": job_title,
                    "status": "generated"
                })

                logger.success(f"[WF3] Feedback generated for {recipient_name}")

            except Exception as e:
                logger.error(f"[WF3] Failed to generate feedback for email {email_record.get('id')}: {e}")
                failed += 1

                # Mark as failed in database
                try:
                    supabase_client.table("feedback_emails").update({
                        "status": "failed",
                        "error_message": str(e),
                        "updated_at": datetime.utcnow().isoformat()
                    }).eq("id", email_record["id"]).execute()
                except Exception:
                    pass

        logger.success(f"[WF3] Completed: {processed} generated, {failed} failed")

        return {
            "status": "success",
            "processed": processed,
            "failed": failed,
            "results": results
        }

    except Exception as e:
        logger.exception(f"[WF3] Failed to process pending feedbacks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


def get_latest_job_id_for_applicant(applicant_id: str) -> Optional[str]:
    """Resolve latest scored job for applicant from match_history."""
    response = supabase_client.table("match_history").select(
        "job_id, scored_at"
    ).eq("applicant_id", applicant_id).order("scored_at", desc=True).limit(1).execute()

    if response.data and len(response.data) > 0:
        return response.data[0].get("job_id")
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
