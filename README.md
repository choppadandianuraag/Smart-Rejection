# Smart Rejection - AI-Powered Resume Screening System ✅

> 📘 **[View Complete Architecture & Flow →](ARCHITECTURE.md)**
> 📘 **[V2 Section-Aware System Documentation →](README_V2.md)**
> 📘 **[Migration Guide (V1→V2) →](MIGRATION_GUIDE.md)**

## Overview

**Smart Rejection** is a **complete, production-ready** intelligent resume screening system with comprehensive AI-powered workflows for resume processing, scoring, and feedback generation.

### System Implementations

- **V1 (Legacy)**: Whole-document hybrid TF-IDF + BERT embeddings
- **V2 (Recommended)**: Section-aware pure BERT embeddings with explainable matching
- **Master Server**: Complete webhook-based feedback pipeline with database integration

## Complete Features & Workflows

### Core Resume Processing
- **Resume Extraction**: Multi-format support (PDF, DOCX, images with OCR)
- **Advanced OCR**: NuMarkdown-8B-Thinking model for document processing
- **Hybrid & Pure Embeddings**: Both TF-IDF+BERT and pure BERT implementations
- **LLM Integration**: Groq API with Llama 3.3-70b for intelligent analysis

### Scoring & Ranking System
- **ATS Scoring**: Comprehensive applicant tracking system compatibility scoring
- **Semantic Matching**: Advanced BERT-based semantic similarity analysis
- **Multi-criteria Evaluation**: Skills, experience, education, and keyword matching

### Feedback & Improvement Pipeline
- **Automated Feedback Generation**: AI-powered resume improvement suggestions
- **Webhook Integration**: Real-time processing via master webhook server
- **Vector Store Analysis**: Semantic similarity-based feedback recommendations
- **Batch Processing**: Efficient handling of multiple resume evaluations

### Database & Storage
- **PostgreSQL Backend**: Complete Supabase integration with migrations
- **V1 & V2 Schemas**: Support for both legacy and modern data structures
- **Status Tracking**: Comprehensive workflow status management
- **Feedback Storage**: Persistent storage of generated feedback and recommendations

## Quick Navigation

| What You Want | Go To |
|---------------|-------|
| Understand the entire system | **[ARCHITECTURE.md](ARCHITECTURE.md)** |
| Use the new V2 system | **[README_V2.md](README_V2.md)** |
| Migrate from V1 to V2 | **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** |
| Run V1 legacy system | This file (below) |

---

## V1 Features (Legacy System)

- **Resume Extraction**: Extract text from PDFs, DOCX, and images (OCR)
- **OCR Processing**: NuMarkdown-8B-Thinking model for advanced document OCR
- **Hybrid Embeddings**: 70% TF-IDF + 30% BERT for semantic matching
- **LLM Ranking**: Groq API with Llama 3.3-70b for keyword extraction
- **Database Storage**: Supabase with PostgreSQL backend

## Project Structure (V1 - See ARCHITECTURE.md for complete structure)

```
smart_rejection/
├── config/
│   └── settings.py               # Shared configuration
├── database/
│   ├── models.py                 # V1 Pydantic models
│   ├── supabase_client.py        # V1 database client
│   ├── schema.sql                # V1 schema (single table)
│   ├── models_v2.py              # V2 models
│   ├── supabase_client_v2.py     # V2 database client
│   └── schema_v2.sql             # V2 schema (multi-table)
├── shared/
│   └── database/migrations/      # Database migration files
├── extractors/                   # Shared: Text extraction
│   ├── base.py
│   ├── pdf_extractor.py
│   ├── docx_extractor.py
│   ├── ocr_extractor.py
│   └── resume_processor.py
├── embeddings/                   # V1 & V2 embedders
│   ├── tfidf_embedder.py        # V1 only
│   ├── bert_embedder.py         # V1 version
│   ├── bert_embedder_v2.py      # V2 version
│   ├── hybrid_embedder.py       # V1 only
│   ├── preprocessor.py          # V1 preprocessing
│   └── embedding_service.py     # V1 service
├── segmentation/                 # V2 only
│   └── __init__.py              # Section segmentation
├── workflow_3_feedback/          # Complete feedback pipeline
│   ├── feedback_engine.py       # Core feedback generation
│   ├── vector_store.py          # Similarity analysis
│   ├── batch_feedback.py        # Batch processing
│   ├── webhook_server.py        # Feedback webhook handler
│   └── main.py                  # Feedback CLI
├── master_webhook_server.py      # Master webhook orchestrator
├── start_master_server.sh        # Server startup script
├── test_master_server.py         # Server testing utilities
├── main.py                       # V1 CLI
├── main_v2.py                    # V2 CLI (recommended)
├── ats_ranking.py                # V1 LLM ranking
├── rank_resumes.py               # V1 simple ranking
├── generate_embeddings.py        # V1 batch embedder
├── ingestion_pipeline_v2.py      # V2 ingestion
├── scoring_pipeline_v2.py        # V2 scoring
├── quickstart_v2.py              # V2 examples
├── ARCHITECTURE.md               # Complete system architecture
├── README_V2.md                  # V2 documentation
├── MIGRATION_GUIDE.md            # V1→V2 migration
├── MASTER_SERVER_README.md       # Master server documentation
└── MASTER_SERVER_QUICK_REF.md    # Master server quick reference

See [ARCHITECTURE.md](ARCHITECTURE.md) for complete flow diagrams and file usage.
```

## Setup Instructions

### 1. Clone and Navigate

```bash
cd smart_rejection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
.\venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for macOS users**: For PDF to image conversion, you need to install poppler:
```bash
brew install poppler
```

**Note for OCR fallback**: If using Tesseract fallback, install:
```bash
brew install tesseract  # macOS
# or
sudo apt-get install tesseract-ocr  # Ubuntu/Debian
```

### 4. Set Up Supabase

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to SQL Editor and run the contents of `database/schema.sql`
3. Go to Settings > API to get your project URL and keys

### 5. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
HF_TOKEN=your-huggingface-token  # For NuMarkdown model
MODEL_NAME=NuMind/NuMarkdown-8B-Thinking
DEVICE=cuda  # or 'cpu'
```

### 6. Hugging Face Token (Optional)

If using NuMarkdown model:
1. Create account at [huggingface.co](https://huggingface.co)
2. Go to Settings > Access Tokens
3. Create a new token with read access
4. Add to your `.env` file

## Usage

### Process a Single Resume

```bash
python main.py process --file /path/to/resume.pdf
```

### Process All Resumes in a Directory

```bash
python main.py process-dir --directory /path/to/resumes/
```

### List All Stored Resumes

```bash
python main.py list --limit 20
```

### Get Resume Details

```bash
python main.py get --id <resume-uuid>
```

### Use Master Webhook Server

```bash
# Start the master server
./start_master_server.sh

# Test the server endpoints
python test_master_server.py
```

### Generate Feedback for Resumes

```bash
cd workflow_3_feedback
python main.py --resume_path /path/to/resume.pdf --job_description "Job requirements here"
```

## Programmatic Usage

### Basic Resume Processing

```python
from smart_rejection.main import SmartRejectionApp

# Initialize app
app = SmartRejectionApp(use_numarkdown=True)

# Process a resume
result = app.process_resume("/path/to/resume.pdf")
print(f"Resume ID: {result['id']}")
print(f"Skills found: {result['extracted_data']['skills']}")

# List all resumes
resumes = app.list_resumes(limit=10)
for r in resumes:
    print(f"- {r['filename']}: {r['status']}")
```

### Advanced Feedback Generation

```python
from workflow_3_feedback.feedback_engine import FeedbackEngine

# Initialize feedback engine
engine = FeedbackEngine()

# Generate comprehensive feedback
feedback = engine.generate_feedback(
    resume_path="/path/to/resume.pdf",
    job_description="Software Engineer position requirements..."
)

print(f"Overall Score: {feedback['score']}")
print(f"Suggestions: {feedback['improvement_suggestions']}")
```

### Master Server Integration

```python
import requests

# Send resume for processing via webhook
response = requests.post('http://localhost:8000/webhook/feedback',
                        json={
                            'resume_path': '/path/to/resume.pdf',
                            'job_description': 'Job requirements...'
                        })

feedback_result = response.json()
```

## Supported File Formats

| Format | Extension | Method |
|--------|-----------|--------|
| PDF | .pdf | pdfplumber/PyPDF2 + OCR fallback |
| Word | .docx, .doc | python-docx/mammoth |
| Images | .png, .jpg, .jpeg | NuMarkdown OCR / Tesseract |

## Requirements

- Python 3.10+
- CUDA GPU (recommended for NuMarkdown, ~8GB VRAM)
- Supabase account
- Hugging Face account (for NuMarkdown model access)

## Production Deployment

The system is production-ready with:
- **Webhook Integration**: Master server handles real-time processing
- **Database Migrations**: Automated schema updates
- **Error Handling**: Comprehensive error tracking and recovery
- **Scalable Architecture**: Supports high-volume resume processing
- **API Endpoints**: RESTful APIs for integration with external systems

## Documentation

- **[MASTER_SERVER_README.md](MASTER_SERVER_README.md)**: Complete master server setup and API documentation
- **[MASTER_SERVER_QUICK_REF.md](MASTER_SERVER_QUICK_REF.md)**: Quick reference for common operations
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and flow diagrams
- **[README_V2.md](README_V2.md)**: V2 system documentation
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)**: Migration from V1 to V2

## Troubleshooting

- Python 3.10+
- CUDA GPU (recommended for NuMarkdown, ~8GB VRAM)
- Supabase account
- Hugging Face account (for NuMarkdown model access)

## Troubleshooting

### "CUDA out of memory"
- Use `--no-numarkdown` flag for CPU-only processing
- Or set `DEVICE=cpu` in `.env`

### "Module not found"
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`

### Supabase Connection Error
- Verify SUPABASE_URL and SUPABASE_KEY in `.env`
- Check if RLS policies are set up correctly

## License

MIT License - See LICENSE file for details.
