#!/usr/bin/env python3
"""
Webhook Server for Resume Preprocessing Pipeline
=================================================

FastAPI server that receives form submissions and processes resumes.

Workflow:
1. Form submission → Make.com webhook → This server
2. Server downloads/processes resume
3. Returns applicant_id for Supabase update

Run:
    uvicorn webhook_server:app --host 0.0.0.0 --port 8000

Or for development:
    python webhook_server.py
"""

import sys
import os
import tempfile
import base64
import httpx
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from loguru import logger

from ingestion_pipeline import SectionAwareResumeProcessor, create_section_aware_processor

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    Path(__file__).parent.parent / "logs" / "webhook_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG"
)

# Global processor instance (initialized on startup)
processor: Optional[SectionAwareResumeProcessor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize processor on startup."""
    global processor
    logger.info("Initializing resume processor...")
    processor = create_section_aware_processor(use_ocr=True, use_numarkdown=False)
    logger.success("Resume processor ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Resume Preprocessing Webhook",
    description="Webhook endpoint for processing resumes from form submissions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for Make.com and other services
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

class WebhookPayload(BaseModel):
    """Payload from Make.com webhook."""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    resume_url: Optional[str] = None  # URL to download resume
    resume_base64: Optional[str] = None  # Base64 encoded resume
    filename: Optional[str] = "resume.pdf"  # Original filename


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


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    processor_ready: bool


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        processor_ready=processor is not None
    )


@app.post("/webhook/process", response_model=ProcessingResult)
async def process_resume_webhook(payload: WebhookPayload):
    """
    Main webhook endpoint for processing resumes.

    Accepts either:
    - resume_url: URL to download the resume from
    - resume_base64: Base64 encoded resume file

    Returns applicant_id for Supabase update.
    """
    global processor

    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    if not payload.resume_url and not payload.resume_base64:
        raise HTTPException(
            status_code=400,
            detail="Either resume_url or resume_base64 is required"
        )

    temp_file = None

    try:
        # Get resume file
        if payload.resume_url:
            logger.info(f"Downloading resume from: {payload.resume_url}")
            temp_file = await download_file(payload.resume_url, payload.filename)
        else:
            logger.info("Decoding base64 resume")
            temp_file = decode_base64_file(payload.resume_base64, payload.filename)

        logger.info(f"Processing: {temp_file}")

        # Process through pipeline
        result = processor.process_resume(
            file_path=Path(temp_file),
            name=payload.name,
            email=payload.email,
            contact_number=payload.phone
        )

        logger.success(f"Processed successfully: {result['applicant_id']}")

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
        logger.error(f"Validation error: {e}")
        return ProcessingResult(
            status="error",
            error=str(e)
        )

    except Exception as e:
        logger.exception(f"Processing failed: {e}")
        return ProcessingResult(
            status="error",
            error=str(e)
        )

    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


@app.post("/webhook/upload", response_model=ProcessingResult)
async def process_resume_upload(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None)
):
    """
    Alternative endpoint for direct file upload.

    Use this for multipart/form-data submissions.
    """
    global processor

    if processor is None:
        raise HTTPException(status_code=503, detail="Processor not initialized")

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
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file = tmp.name

        logger.info(f"Processing uploaded file: {file.filename}")

        # Process through pipeline
        result = processor.process_resume(
            file_path=Path(temp_file),
            name=name,
            email=email,
            contact_number=phone
        )

        logger.success(f"Processed successfully: {result['applicant_id']}")

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
        logger.exception(f"Processing failed: {e}")
        return ProcessingResult(
            status="error",
            error=str(e)
        )

    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)


# ============================================================================
# Helper Functions
# ============================================================================

async def download_file(url: str, filename: str) -> str:
    """Download file from URL to temp location."""
    # Determine extension from URL or filename
    ext = Path(filename).suffix.lower()
    if not ext:
        # Try to get from URL
        ext = Path(url.split('?')[0]).suffix.lower()
    if not ext:
        ext = '.pdf'  # Default

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=ext
        ) as tmp:
            tmp.write(response.content)
            return tmp.name


def decode_base64_file(base64_content: str, filename: str) -> str:
    """Decode base64 content to temp file."""
    ext = Path(filename).suffix.lower() or '.pdf'

    # Handle data URI format
    if ',' in base64_content:
        base64_content = base64_content.split(',')[1]

    content = base64.b64decode(base64_content)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=ext
    ) as tmp:
        tmp.write(content)
        return tmp.name


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    logger.info(f"Starting webhook server on port {port}")

    uvicorn.run(
        "webhook_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
