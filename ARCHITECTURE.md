# Smart Rejection System - Complete Architecture

## 📊 System Overview

You currently have **TWO PARALLEL SYSTEMS**:
- **V1 (Legacy)**: Whole-document hybrid TF-IDF + BERT embeddings
- **V2 (New)**: Section-aware pure BERT embeddings with dual-table storage

Both systems are fully functional and independent.

---

## 🏗️ V2 Architecture (Recommended - Section-Aware)

### **Core Pipeline Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESUME INGESTION PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

1. EXTRACTION (Reuses V1 extractors)
   ├─ PDF: PyPDF2 + pdfplumber
   ├─ DOCX: python-docx
   └─ OCR fallback: Tesseract + NuMarkdown-8B-Thinking
   
2. SEGMENTATION (NEW - No LLM required)
   ├─ Regex-based section detection
   ├─ Confidence scoring (0.0-1.0)
   ├─ Contact info extraction
   └─ Flags low-confidence resumes for manual review
   
3. EMBEDDING (NEW - Pure BERT)
   ├─ Model: sentence-transformers/all-mpnet-base-v2
   ├─ Output: 768-dimensional vectors per section
   ├─ Batch processing (32 sections at a time)
   └─ Device-aware (CUDA/MPS/CPU)
   
4. STORAGE (NEW - Dual-table design)
   ├─ applicant_profiles: Human-readable data
   ├─ applicant_embeddings: Section vectors
   └─ Atomic two-phase commit

┌─────────────────────────────────────────────────────────────────┐
│                     SCORING PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

1. JOB PROCESSING
   ├─ Segment job description into sections
   ├─ Generate BERT embeddings per section
   └─ Cache in job_embeddings table
   
2. CANDIDATE MATCHING
   ├─ Fetch all candidate section embeddings
   ├─ Compute cosine similarity (section-to-section)
   └─ Apply configurable weights
   
3. SCORING
   ├─ Skills → Requirements: 40%
   ├─ Experience → Responsibilities: 35%
   ├─ Education → Qualifications: 15%
   ├─ Projects → Responsibilities: 10%
   └─ Weight redistribution if sections missing
   
4. RANKING
   ├─ Update match_score in applicant_profiles
   ├─ Log to match_history for audit trail
   └─ Return top N candidates
```

### **V2 File Structure**

#### **Entry Point**
- **main_v2.py** - CLI with 4 commands:
  - `ingest-resume` - Process single resume
  - `ingest-directory` - Batch process folder
  - `process-job` - Segment & embed job description
  - `score-candidates` - Rank all candidates for job

#### **Core Pipelines**
- **ingestion_pipeline_v2.py** - 4-stage resume ingestion
  - `SectionAwareResumeProcessor` class
  - Handles resume updates (soft delete pattern)
  
- **scoring_pipeline_v2.py** - Weighted section matching
  - `SectionAwareScoringPipeline` class
  - Configurable section weights

#### **Segmentation Module**
- **segmentation/__init__.py**
  - `ResumeSegmenter` - Splits resume into sections
  - `JobDescriptionSegmenter` - Splits JD into sections
  - Confidence scoring & manual review flagging

#### **Embedding Module**
- **embeddings/bert_embedder_v2.py**
  - `BERTEmbedder` - Core BERT embedding generation
  - `SectionAwareEmbedder` - Weighted matching logic
  - Batch processing & device management

#### **Database Layer**
- **database/schema_v2.sql** - PostgreSQL schema:
  - `applicant_profiles` - Source of truth
  - `applicant_embeddings` - Vector store (pgvector)
  - `job_descriptions` - Job metadata
  - `job_embeddings` - Cached JD vectors
  - `scoring_config` - Configurable weights
  - `match_history` - Audit trail
  - IVFFlat indexes for fast similarity search
  
- **database/models_v2.py** - 13 Pydantic models
  
- **database/supabase_client_v2.py** - Database operations
  - Dual-table CRUD
  - Batch embedding storage
  - Top candidates retrieval

#### **Documentation**
- **README_V2.md** - Comprehensive V2 guide
- **MIGRATION_GUIDE.md** - V1→V2 migration steps
- **quickstart_v2.py** - 5 workflow examples

---

## 🔧 V1 Architecture (Legacy - Whole Document)

### **Core Pipeline Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│                    V1 EXTRACTION & STORAGE                       │
└─────────────────────────────────────────────────────────────────┘

1. EXTRACTION
   ├─ Same extraction as V2 (PDF/DOCX/OCR)
   └─ Stores entire document as single text blob
   
2. EMBEDDING GENERATION (Separate phase)
   ├─ Hybrid: 70% TF-IDF + 30% BERT
   ├─ Single embedding for entire resume
   └─ generate_embeddings.py script
   
3. STORAGE
   └─ Single table: resumes
      ├─ raw_text
      ├─ markdown_content
      ├─ extracted_data (JSONB)
      └─ embedding_vector (FLOAT8[])

┌─────────────────────────────────────────────────────────────────┐
│                     V1 RANKING METHODS                           │
└─────────────────────────────────────────────────────────────────┘

METHOD 1: Simple Hybrid Ranking (rank_resumes.py)
├─ Generate hybrid embedding for job description
├─ Compute cosine similarity with all resumes
└─ Return ranked list

METHOD 2: LLM-Enhanced Ranking (ats_ranking.py)
├─ Use Groq API (Llama 3.3-70b) to extract JD keywords
├─ Weighted scoring:
│  ├─ Skills match: 60%
│  ├─ Experience match: 25%
│  └─ Education match: 15%
└─ Combined with embedding similarity
```

### **V1 File Structure**

#### **Entry Point**
- **main.py** - Original CLI
  - `SmartRejectionApp` class
  - `process_resume()` - Single resume
  - `process_directory()` - Batch processing

#### **Extraction Layer** (Shared with V2)
- **extractors/base.py** - Abstract base class
- **extractors/pdf_extractor.py** - PDF extraction
- **extractors/docx_extractor.py** - DOCX extraction
- **extractors/ocr_extractor.py** - OCR with NuMarkdown
- **extractors/resume_processor.py** - Unified processor

#### **Embedding Layer**
- **embeddings/tfidf_embedder.py** - TF-IDF embeddings
- **embeddings/bert_embedder.py** - BERT embeddings (V1)
- **embeddings/hybrid_embedder.py** - 70/30 hybrid
- **embeddings/preprocessor.py** - Text preprocessing
- **embeddings/embedding_service.py** - Unified service

#### **Ranking Systems**
- **rank_resumes.py** - Simple hybrid ranking
  - `ResumeRanker` class
  - Cosine similarity-based
  
- **ats_ranking.py** - LLM-enhanced ranking
  - `ATSRankingSystem` class
  - Groq API integration
  - Keyword extraction + embedding
  
- **generate_embeddings.py** - Batch embedding generation
  - Handles resumes without embeddings
  - `--force` flag to re-embed all

#### **Database Layer**
- **database/schema.sql** - V1 schema (single table)
- **database/models.py** - V1 Pydantic models
- **database/supabase_client.py** - V1 database ops

#### **Configuration**
- **config/settings.py** - Shared settings
- **.env** - Environment variables (gitignored)

---

## 📁 Directory Structure

```
smart_rejection/
│
├── main.py                      # V1 CLI entry point
├── main_v2.py                   # V2 CLI entry point ⭐
├── ats_ranking.py               # V1 LLM ranking
├── rank_resumes.py              # V1 simple ranking
├── generate_embeddings.py       # V1 batch embedder
│
├── ingestion_pipeline_v2.py     # V2 ingestion pipeline ⭐
├── scoring_pipeline_v2.py       # V2 scoring pipeline ⭐
├── quickstart_v2.py             # V2 examples ⭐
│
├── segmentation/                # V2 segmentation module ⭐
│   └── __init__.py
│
├── extractors/                  # Shared: Text extraction
│   ├── base.py
│   ├── pdf_extractor.py
│   ├── docx_extractor.py
│   ├── ocr_extractor.py
│   └── resume_processor.py
│
├── embeddings/                  # Both V1 & V2 embedders
│   ├── tfidf_embedder.py       # V1 only
│   ├── bert_embedder.py        # V1 version
│   ├── bert_embedder_v2.py     # V2 version ⭐
│   ├── hybrid_embedder.py      # V1 only
│   ├── preprocessor.py         # V1 only
│   └── embedding_service.py    # V1 only
│
├── database/
│   ├── schema.sql              # V1 schema
│   ├── schema_v2.sql           # V2 schema ⭐
│   ├── models.py               # V1 models
│   ├── models_v2.py            # V2 models ⭐
│   ├── supabase_client.py      # V1 client
│   └── supabase_client_v2.py   # V2 client ⭐
│
├── config/
│   └── settings.py             # Shared configuration
│
├── README.md                    # General project info
├── README_V2.md                 # V2 documentation ⭐
├── MIGRATION_GUIDE.md           # V1→V2 migration ⭐
├── ARCHITECTURE.md              # This file ⭐
└── requirements.txt             # Python dependencies

⭐ = New V2 files
```

---

## 🔄 Complete Data Flow Comparison

### **V1 Flow (Whole Document)**
```
Resume PDF
    ↓
Extract entire text → Store raw_text
    ↓
Generate single hybrid embedding (TF-IDF + BERT)
    ↓
Store embedding_vector in resumes table
    ↓
Job Description → Single embedding
    ↓
Cosine similarity with all resumes
    ↓
Ranked list
```

### **V2 Flow (Section-Aware)**
```
Resume PDF
    ↓
Extract entire text
    ↓
Segment into sections (Experience, Skills, Education, etc.)
    ↓
Generate BERT embedding for EACH section
    ↓
Store in applicant_profiles + applicant_embeddings (multiple rows)
    ↓
Job Description → Segment into sections
    ↓
Generate BERT embedding for EACH JD section
    ↓
Match section-to-section:
    - Skills ↔ Requirements (40%)
    - Experience ↔ Responsibilities (35%)
    - Education ↔ Qualifications (15%)
    - Projects ↔ Responsibilities (10%)
    ↓
Weighted composite score
    ↓
Ranked list with explainability
```

---

## 🎯 Which Files Are Actually Used?

### **For V2 (Recommended):**
```bash
# Required files for V2:
main_v2.py                       # CLI
ingestion_pipeline_v2.py         # Ingestion
scoring_pipeline_v2.py           # Scoring
segmentation/__init__.py         # Segmentation
embeddings/bert_embedder_v2.py   # Embeddings
database/schema_v2.sql           # Database
database/models_v2.py            # Models
database/supabase_client_v2.py   # DB Client
extractors/*                     # Extraction (reused from V1)
config/settings.py               # Config

# Optional:
quickstart_v2.py                 # Examples
README_V2.md                     # Documentation
MIGRATION_GUIDE.md               # Migration help
```

### **For V1 (Legacy):**
```bash
# Simple ranking workflow:
main.py                          # CLI
extractors/*                     # Extraction
generate_embeddings.py           # Generate embeddings
rank_resumes.py                  # Simple ranking
embeddings/hybrid_embedder.py    # Hybrid embeddings
database/*                       # V1 database

# LLM ranking workflow:
main.py                          # CLI
extractors/*                     # Extraction
ats_ranking.py                   # LLM ranking (uses Groq)
embeddings/hybrid_embedder.py    # Embeddings
database/*                       # V1 database
```

---

## 🚀 Quick Start Guide

### **V2 System (Section-Aware)**

```bash
# 1. Setup database
# - Open Supabase SQL Editor
# - Run database/schema_v2.sql

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
# - Copy .env.example to .env
# - Add your Supabase credentials

# 4. Ingest resumes
python main_v2.py ingest-resume path/to/resume.pdf
python main_v2.py ingest-directory test_resumes/

# 5. Process job description
python main_v2.py process-job job_id --description "Job description text..."

# 6. Score and rank candidates
python main_v2.py score-candidates job_id --top 10
```

### **V1 System (Legacy)**

```bash
# 1. Process resumes
python main.py

# 2. Generate embeddings
python generate_embeddings.py

# 3. Rank with simple method
python rank_resumes.py

# OR rank with LLM
python ats_ranking.py
```

---

## 📊 Performance Comparison

| Feature | V1 | V2 |
|---------|----|----|
| **Embedding Type** | Hybrid (TF-IDF + BERT) | Pure BERT |
| **Granularity** | Whole document | Section-aware |
| **Database Tables** | 1 table | 6 tables |
| **Indexing** | None | pgvector IVFFlat |
| **Explainability** | Low | High (section scores) |
| **Search Speed** | O(n) linear | O(log n) with index |
| **Accuracy** | Good | Better (section matching) |
| **Storage** | Compact | Larger (multiple rows) |
| **Resume Updates** | Overwrite | Versioned (soft delete) |
| **Missing Sections** | N/A | Auto weight redistribution |

---

## 🔧 Configuration

### **V2 Section Weights** (Customizable)
Edit in [scoring_pipeline_v2.py](scoring_pipeline_v2.py#L18-L23):
```python
SECTION_MAPPINGS = [
    ('skills', 'requirements', 0.40),
    ('work_experience', 'responsibilities', 0.35),
    ('education', 'qualifications', 0.15),
    ('projects', 'responsibilities', 0.10),
]
```

### **Environment Variables**
```bash
# Required
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# Optional (V1 only)
GROQ_API_KEY=your-groq-key  # For ats_ranking.py
HUGGINGFACE_TOKEN=your-token  # For NuMarkdown OCR
```

---

## 🎓 Key Architectural Decisions

### **Why Dual Tables (V2)?**
- Separates concerns: Human data vs vectors
- Enables efficient indexing (pgvector on embeddings only)
- Allows multiple embedding versions per applicant
- Better query performance

### **Why Pure BERT vs Hybrid?**
- BERT captures semantic meaning better than TF-IDF
- 768-dim vectors sufficient for discrimination
- Consistent embedding space across all sections
- Simpler pipeline (one model vs two)

### **Why Section-Aware?**
- Better explainability (see which sections matched)
- Targeted matching (skills to requirements, etc.)
- Handles missing sections gracefully
- More accurate than whole-document comparison

### **Why No LLM for Segmentation?**
- Cost-effective (no API calls)
- Fast (regex-based)
- Consistent (deterministic)
- Good enough (confidence scoring flags edge cases)

---

## 📝 Next Steps

1. **Deploy V2 Schema**: Run `database/schema_v2.sql` in Supabase
2. **Test V2 System**: Run `python quickstart_v2.py`
3. **Compare Results**: Run both V1 and V2 on same resumes
4. **Tune Weights**: Adjust section weights based on your domain
5. **Migrate Data**: Use `MIGRATION_GUIDE.md` if switching from V1

---

## 🆘 Troubleshooting

### **V2 Issues**
- **Error: relation "applicant_profiles" does not exist**
  → Run `database/schema_v2.sql` in Supabase

- **Error: extension "vector" does not exist**
  → Enable pgvector in Supabase dashboard

- **Slow similarity search**
  → Check IVFFlat indexes are created
  → Increase `lists` parameter for larger datasets

### **V1 Issues**
- **LLM ranking fails**
  → Check `GROQ_API_KEY` in .env
  → Verify Groq API quota

- **Empty embeddings**
  → Run `python generate_embeddings.py --force`

---

**Version**: V2.0  
**Last Updated**: February 17, 2026  
**Author**: Smart Rejection System
