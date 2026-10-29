# Smart Rejection: AI-Powered Resume Screening

> 📘 [Full Architecture & Flow](ARCHITECTURE.md) · [V2 Section-Aware System](README_V2.md) · [V1 → V2 Migration Guide](MIGRATION_GUIDE.md)

## Overview

**Smart Rejection** is a production-ready resume screening system. It covers the full pipeline: extracting resume text, scoring candidates against job requirements, and generating improvement feedback.

### Implementations

| Version | Description |
|---------|-------------|
| **V1 (Legacy)** | Whole-document hybrid embeddings (TF-IDF + BERT) |
| **V2 (Recommended)** | Section-aware pure BERT embeddings with explainable matching |
| **Master Server** | Webhook-driven feedback pipeline with database integration |

## Key Capabilities

### Resume Processing
- **Multi-format extraction**: PDF, DOCX, and images (via OCR)
- **Advanced OCR**: NuMarkdown-8B-Thinking for complex documents
- **Flexible embeddings**: hybrid (TF-IDF + BERT) or pure BERT
- **LLM analysis**: Groq API running Llama 3.3-70b

### Scoring & Ranking
- **ATS scoring**: checks compatibility with applicant tracking systems
- **Semantic matching**: BERT-based similarity analysis
- **Multi-criteria evaluation**: skills, experience, education, and keywords

### Feedback & Improvement
- **Automated feedback**: AI-generated suggestions for improving a resume
- **Webhook processing**: real-time handling through the master server
- **Vector store analysis**: similarity-based recommendations
- **Batch mode**: evaluate many resumes efficiently

### Data & Storage
- **PostgreSQL via Supabase**, with migrations
- **V1 and V2 schemas** supported side by side
- **Status tracking** across the whole workflow
- **Persistent feedback storage**

## Quick Navigation

| I want to... | Go to |
|--------------|-------|
| Understand the whole system | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Use the V2 system | [README_V2.md](README_V2.md) |
| Migrate from V1 to V2 | [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) |
| Run the V1 legacy system | Continue reading below |

---

## V1 (Legacy) Overview

- **Extraction**: text from PDFs, DOCX files, and images
- **OCR**: NuMarkdown-8B-Thinking for advanced document OCR
- **Embeddings**: 70% TF-IDF + 30% BERT
- **LLM ranking**: keyword extraction via Groq (Llama 3.3-70b)
- **Storage**: Supabase (PostgreSQL)

## Project Structure

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
│   └── database/migrations/      # Migration files
├── extractors/                   # Shared text extraction
│   ├── base.py
│   ├── pdf_extractor.py
│   ├── docx_extractor.py
│   ├── ocr_extractor.py
│   └── resume_processor.py
├── embeddings/                   # V1 & V2 embedders
│   ├── tfidf_embedder.py         # V1 only
│   ├── bert_embedder.py          # V1
│   ├── bert_embedder_v2.py       # V2
│   ├── hybrid_embedder.py        # V1 only
│   ├── preprocessor.py           # V1 preprocessing
│   └── embedding_service.py      # V1 service
├── segmentation/                 # V2 only: section segmentation
├── workflow_3_feedback/          # Feedback pipeline
│   ├── feedback_engine.py        # Core feedback generation
│   ├── vector_store.py           # Similarity analysis
│   ├── batch_feedback.py         # Batch processing
│   ├── webhook_server.py         # Feedback webhook handler
│   └── main.py                   # Feedback CLI
├── master_webhook_server.py      # Master webhook orchestrator
├── start_master_server.sh        # Server startup script
├── test_master_server.py         # Server test utilities
├── main.py                       # V1 CLI
├── main_v2.py                    # V2 CLI (recommended)
├── ats_ranking.py                # V1 LLM ranking
├── rank_resumes.py               # V1 simple ranking
├── generate_embeddings.py        # V1 batch embedder
├── ingestion_pipeline_v2.py      # V2 ingestion
├── scoring_pipeline_v2.py        # V2 scoring
├── quickstart_v2.py              # V2 examples
├── ARCHITECTURE.md
├── README_V2.md
├── MIGRATION_GUIDE.md
├── MASTER_SERVER_README.md
└── MASTER_SERVER_QUICK_REF.md
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for flow diagrams and details on how each file is used.

## Requirements

- Python 3.10+
- CUDA GPU recommended for NuMarkdown (~8 GB VRAM)
- Supabase account
- Hugging Face account (for NuMarkdown model access)

## Setup

### 1. Enter the project directory

```bash
cd smart_rejection
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
.\venv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**macOS:** install poppler for PDF-to-image conversion:
```bash
brew install poppler
```

**OCR fallback:** if you plan to use Tesseract, install it:
```bash
brew install tesseract                 # macOS
sudo apt-get install tesseract-ocr     # Ubuntu/Debian
```

### 4. Set up Supabase

1. Create a project at [supabase.com](https://supabase.com).
2. In the SQL Editor, run `database/schema.sql`.
3. Under Settings → API, copy your project URL and keys.

### 5. Configure the environment

```bash
cp .env.example .env
```

Then fill in `.env`:
```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-key
HF_TOKEN=your-huggingface-token   # for the NuMarkdown model
MODEL_NAME=NuMind/NuMarkdown-8B-Thinking
DEVICE=cuda                       # or 'cpu'
```

### 6. Hugging Face token (optional)

Needed only if you use NuMarkdown:
1. Sign up at [huggingface.co](https://huggingface.co).
2. Go to Settings → Access Tokens.
3. Create a token with read access and add it to `.env`.

## Usage

```bash
# Process one resume
python main.py process --file /path/to/resume.pdf

# Process every resume in a folder
python main.py process-dir --directory /path/to/resumes/

# List stored resumes
python main.py list --limit 20

# View a resume's details
python main.py get --id <resume-uuid>
```

### Master webhook server

```bash
./start_master_server.sh      # start the server
python test_master_server.py  # test its endpoints
```

### Generate feedback

```bash
cd workflow_3_feedback
python main.py --resume_path /path/to/resume.pdf --job_description "Job requirements here"
```

## Programmatic Usage

### Resume processing

```python
from smart_rejection.main import SmartRejectionApp

app = SmartRejectionApp(use_numarkdown=True)

result = app.process_resume("/path/to/resume.pdf")
print(f"Resume ID: {result['id']}")
print(f"Skills found: {result['extracted_data']['skills']}")

for r in app.list_resumes(limit=10):
    print(f"- {r['filename']}: {r['status']}")
```

### Feedback generation

```python
from workflow_3_feedback.feedback_engine import FeedbackEngine

engine = FeedbackEngine()
feedback = engine.generate_feedback(
    resume_path="/path/to/resume.pdf",
    job_description="Software Engineer position requirements..."
)

print(f"Overall Score: {feedback['score']}")
print(f"Suggestions: {feedback['improvement_suggestions']}")
```

### Master server via webhook

```python
import requests

response = requests.post(
    "http://localhost:8000/webhook/feedback",
    json={
        "resume_path": "/path/to/resume.pdf",
        "job_description": "Job requirements...",
    },
)
feedback_result = response.json()
```

## Supported File Formats

| Format | Extensions | Method |
|--------|-----------|--------|
| PDF | .pdf | pdfplumber / PyPDF2, with OCR fallback |
| Word | .docx, .doc | python-docx / mammoth |
| Images | .png, .jpg, .jpeg | NuMarkdown OCR / Tesseract |

## Production Deployment

- **Webhook integration**: the master server handles real-time processing
- **Database migrations**: automated schema updates
- **Error handling**: thorough error tracking and recovery
- **Scalability**: built for high-volume resume processing
- **REST APIs**: easy integration with external systems

## Further Documentation

- [MASTER_SERVER_README.md](MASTER_SERVER_README.md): master server setup and API reference
- [MASTER_SERVER_QUICK_REF.md](MASTER_SERVER_QUICK_REF.md): quick reference for common operations
- [ARCHITECTURE.md](ARCHITECTURE.md): architecture and flow diagrams
- [README_V2.md](README_V2.md): V2 system docs
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md): V1 → V2 migration

## Troubleshooting

**"CUDA out of memory"**
- Pass the `--no-numarkdown` flag for CPU-only processing, or set `DEVICE=cpu` in `.env`.

**"Module not found"**
- Make sure the virtual environment is activated, then run `pip install -r requirements.txt`.

**Supabase connection error**
- Check `SUPABASE_URL` and `SUPABASE_KEY` in `.env`, and confirm your RLS policies are configured correctly.

## License

MIT License. See the LICENSE file for details.
