# Smart Rejection - AI-Powered Resume Feedback System

## Overview

Smart Rejection is an intelligent system designed to transform the traditional job rejection process by providing applicants with detailed, actionable feedback on their resumes and applications.

## Current Features (Phase 1)

- **Resume Extraction**: Extract text from PDFs, DOCX, DOC, and image files
- **OCR Processing**: Use NuMarkdown-8B-Thinking model for advanced document OCR
- **Structured Data Parsing**: Automatically extract contact info, skills, education, and experience
- **Database Storage**: Store extracted data in Supabase for efficient querying

## Project Structure

```
smart_rejection/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration management
├── database/
│   ├── __init__.py
│   ├── models.py             # Pydantic data models
│   ├── supabase_client.py    # Database operations
│   └── schema.sql            # Supabase table schema
├── extractors/
│   ├── __init__.py
│   ├── base.py               # Base extractor interface
│   ├── pdf_extractor.py      # PDF text extraction
│   ├── docx_extractor.py     # Word document extraction
│   ├── ocr_extractor.py      # NuMarkdown OCR processing
│   └── resume_processor.py   # Main processing orchestrator
├── uploads/
│   └── resumes/              # Default upload directory
├── logs/                     # Application logs
├── .env.example              # Environment template
├── requirements.txt          # Python dependencies
├── main.py                   # CLI entry point
└── README.md
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
