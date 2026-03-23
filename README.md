# Smart Rejection - AI-Powered Resume Screening System

> 📘 **[View Complete Architecture & Flow →](ARCHITECTURE.md)**  
> 📘 **[V2 Section-Aware System Documentation →](README_V2.md)**  
> 📘 **[Migration Guide (V1→V2) →](MIGRATION_GUIDE.md)**

## Overview

Smart Rejection is an intelligent resume screening system with **two parallel implementations**:

- **V1 (Legacy)**: Whole-document hybrid TF-IDF + BERT embeddings
- **V2 (Recommended)**: Section-aware pure BERT embeddings with explainable matching

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
│   └── settings.py          # Shared configuration
├── database/
│   ├── models.py            # V1 Pydantic models
│   ├── supabase_client.py   # V1 database client
│   ├── schema.sql           # V1 schema (single table)
│   ├── models_v2.py         # V2 models
│   ├── supabase_client_v2.py # V2 database client
│   └── schema_v2.sql        # V2 schema (multi-table)
├── extractors/              # Shared: Text extraction
│   ├── base.py
│   ├── pdf_extractor.py
│   ├── docx_extractor.py
│   ├── ocr_extractor.py
│   └── resume_processor.py
├── embeddings/              # V1 & V2 embedders
│   ├── tfidf_embedder.py   # V1 only
│   ├── bert_embedder.py    # V1 version
│   ├── bert_embedder_v2.py # V2 version
│   ├── hybrid_embedder.py  # V1 only
│   ├── preprocessor.py     # V1 preprocessing
│   └── embedding_service.py # V1 service
├── segmentation/            # V2 only
│   └── __init__.py         # Section segmentation
├── main.py                  # V1 CLI
├── main_v2.py              # V2 CLI (recommended)
├── ats_ranking.py          # V1 LLM ranking
├── rank_resumes.py         # V1 simple ranking
├── generate_embeddings.py  # V1 batch embedder
├── ingestion_pipeline_v2.py # V2 ingestion
├── scoring_pipeline_v2.py  # V2 scoring
├── quickstart_v2.py        # V2 examples
├── ARCHITECTURE.md         # Complete system architecture
├── README_V2.md            # V2 documentation
└── MIGRATION_GUIDE.md      # V1→V2 migration

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

### Use Tesseract Fallback (No GPU Required)

```bash
python main.py process --file resume.pdf --no-numarkdown
```

## Programmatic Usage

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

## Supported File Formats

| Format | Extension | Method |
|--------|-----------|--------|
| PDF | .pdf | pdfplumber/PyPDF2 + OCR fallback |
| Word | .docx, .doc | python-docx/mammoth |
| Images | .png, .jpg, .jpeg | NuMarkdown OCR / Tesseract |

## Next Steps (Phase 2)

1. **Embedding Generation**: TF-IDF, n-grams, or transformer embeddings
2. **Similarity Search**: Cosine similarity for resume-job matching
3. **Feedback Generation**: AI-powered suggestions for improvement
4. **API Development**: FastAPI endpoints for web integration

## Requirements

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
