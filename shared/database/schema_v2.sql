-- Smart Rejection v2 - Section-Aware Resume Screening
-- New schema for section-based storage and matching

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector for BERT embeddings

-- Section type enum
CREATE TYPE section_type_enum AS ENUM (
    'contact_info',
    'summary',
    'work_experience',
    'education',
    'skills',
    'certifications',
    'projects',
    'other'
);

-- Job section type enum
CREATE TYPE job_section_type_enum AS ENUM (
    'overview',
    'requirements',
    'responsibilities',
    'qualifications',
    'nice_to_have',
    'other'
);

-- ============================================================================
-- TABLE 1: applicant_profiles (Source of truth for human-readable data)
-- ============================================================================
CREATE TABLE IF NOT EXISTS applicant_profiles (
    applicant_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Basic information
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    contact_number VARCHAR(50),
    
    -- Match score (populated by scoring pipeline)
    match_score DECIMAL(5,4),  -- 0.0000 to 1.0000
    last_scored_job_id UUID,
    
    -- Resume metadata
    original_filename TEXT,
    file_type TEXT,
    file_size_bytes INTEGER,
    raw_text TEXT,  -- Keep full text for reference
    
    -- Segmentation quality flags
    segmentation_confidence DECIMAL(4,3),  -- Overall confidence
    needs_manual_review BOOLEAN DEFAULT FALSE,
    review_reason TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_scored_at TIMESTAMPTZ
);

-- Indexes for applicant_profiles
CREATE INDEX idx_profiles_email ON applicant_profiles(email);
CREATE INDEX idx_profiles_match_score ON applicant_profiles(match_score DESC NULLS LAST);
CREATE INDEX idx_profiles_needs_review ON applicant_profiles(needs_manual_review) WHERE needs_manual_review = TRUE;
CREATE INDEX idx_profiles_created_at ON applicant_profiles(created_at DESC);

-- ============================================================================
-- TABLE 2: applicant_embeddings (Vector store - optimized for similarity search)
-- ============================================================================
CREATE TABLE IF NOT EXISTS applicant_embeddings (
    id BIGSERIAL PRIMARY KEY,
    applicant_id UUID NOT NULL REFERENCES applicant_profiles(applicant_id) ON DELETE CASCADE,
    
    -- Section information
    section_type section_type_enum NOT NULL,
    section_text TEXT NOT NULL,  -- Store for debugging/reprocessing
    
    -- BERT embedding (768 dimensions from all-mpnet-base-v2)
    embedding_vector vector(768) NOT NULL,
    
    -- Section metadata
    char_offset_start INTEGER NOT NULL,
    char_offset_end INTEGER NOT NULL,
    section_order INTEGER NOT NULL,  -- Preserve document order
    confidence_score DECIMAL(4,3),   -- Segmentation confidence for this section
    
    -- Version tracking (for resume updates)
    version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Partial unique index to ensure one active section per type per applicant per version
CREATE UNIQUE INDEX idx_embeddings_unique_active_section 
    ON applicant_embeddings(applicant_id, section_type, version) 
    WHERE is_active = TRUE;

-- Indexes for applicant_embeddings
CREATE INDEX idx_embeddings_applicant ON applicant_embeddings(applicant_id) WHERE is_active = TRUE;
CREATE INDEX idx_embeddings_section_type ON applicant_embeddings(section_type) WHERE is_active = TRUE;
CREATE INDEX idx_embeddings_confidence ON applicant_embeddings(confidence_score) WHERE confidence_score < 0.7;
CREATE INDEX idx_embeddings_version ON applicant_embeddings(applicant_id, version, is_active);

-- pgvector IVFFlat indexes for fast similarity search (one per section type)
CREATE INDEX idx_vector_contact ON applicant_embeddings 
    USING ivfflat (embedding_vector vector_cosine_ops) 
    WITH (lists = 100)
    WHERE section_type = 'contact_info' AND is_active = TRUE;

CREATE INDEX idx_vector_summary ON applicant_embeddings 
    USING ivfflat (embedding_vector vector_cosine_ops) 
    WITH (lists = 100)
    WHERE section_type = 'summary' AND is_active = TRUE;

CREATE INDEX idx_vector_experience ON applicant_embeddings 
    USING ivfflat (embedding_vector vector_cosine_ops) 
    WITH (lists = 100)
    WHERE section_type = 'work_experience' AND is_active = TRUE;

CREATE INDEX idx_vector_education ON applicant_embeddings 
    USING ivfflat (embedding_vector vector_cosine_ops) 
    WITH (lists = 100)
    WHERE section_type = 'education' AND is_active = TRUE;

CREATE INDEX idx_vector_skills ON applicant_embeddings 
    USING ivfflat (embedding_vector vector_cosine_ops) 
    WITH (lists = 100)
    WHERE section_type = 'skills' AND is_active = TRUE;

CREATE INDEX idx_vector_certifications ON applicant_embeddings 
    USING ivfflat (embedding_vector vector_cosine_ops) 
    WITH (lists = 100)
    WHERE section_type = 'certifications' AND is_active = TRUE;

CREATE INDEX idx_vector_projects ON applicant_embeddings 
    USING ivfflat (embedding_vector vector_cosine_ops) 
    WITH (lists = 100)
    WHERE section_type = 'projects' AND is_active = TRUE;

-- ============================================================================
-- TABLE 3: job_descriptions (Job postings with metadata)
-- ============================================================================
CREATE TABLE IF NOT EXISTS job_descriptions (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Job information
    title TEXT NOT NULL,
    company TEXT,
    location TEXT,
    job_type TEXT,  -- full-time, part-time, contract, etc.
    
    -- Raw text
    description TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_jobs_active ON job_descriptions(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_jobs_created_at ON job_descriptions(created_at DESC);

-- ============================================================================
-- TABLE 4: job_embeddings (Cached JD section embeddings)
-- ============================================================================
CREATE TABLE IF NOT EXISTS job_embeddings (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES job_descriptions(job_id) ON DELETE CASCADE,
    
    -- Section information
    section_type job_section_type_enum NOT NULL,
    section_text TEXT NOT NULL,
    
    -- BERT embedding
    embedding_vector vector(768) NOT NULL,
    
    -- Section metadata
    char_offset_start INTEGER NOT NULL,
    char_offset_end INTEGER NOT NULL,
    section_order INTEGER NOT NULL,
    confidence_score DECIMAL(4,3),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(job_id, section_type)
);

CREATE INDEX idx_job_embeddings_job_id ON job_embeddings(job_id);
CREATE INDEX idx_job_embeddings_section_type ON job_embeddings(section_type);

-- pgvector indexes for job embeddings
CREATE INDEX idx_job_vector_requirements ON job_embeddings 
    USING ivfflat (embedding_vector vector_cosine_ops) 
    WITH (lists = 50)
    WHERE section_type = 'requirements';

CREATE INDEX idx_job_vector_responsibilities ON job_embeddings 
    USING ivfflat (embedding_vector vector_cosine_ops) 
    WITH (lists = 50)
    WHERE section_type = 'responsibilities';

-- ============================================================================
-- TABLE 5: scoring_config (Configurable weights for matching)
-- ============================================================================
CREATE TABLE IF NOT EXISTS scoring_config (
    id SERIAL PRIMARY KEY,
    config_name VARCHAR(100) UNIQUE NOT NULL,
    
    -- Section-to-section weights as JSONB
    weights JSONB NOT NULL,
    
    -- Example weights structure:
    -- {
    --   "skills_to_requirements": 0.40,
    --   "experience_to_responsibilities": 0.35,
    --   "education_to_qualifications": 0.15,
    --   "summary_to_overview": 0.10
    -- }
    
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100)
);

-- Default scoring configuration
INSERT INTO scoring_config (config_name, weights, description, is_active) VALUES (
    'default',
    '{
        "skills_to_requirements": 0.40,
        "experience_to_responsibilities": 0.35,
        "education_to_qualifications": 0.15,
        "summary_to_overview": 0.10
    }'::jsonb,
    'Default balanced scoring with emphasis on skills and experience',
    TRUE
) ON CONFLICT (config_name) DO NOTHING;

-- Skills-focused configuration
INSERT INTO scoring_config (config_name, weights, description, is_active) VALUES (
    'skills_focused',
    '{
        "skills_to_requirements": 0.60,
        "experience_to_responsibilities": 0.25,
        "education_to_qualifications": 0.10,
        "summary_to_overview": 0.05
    }'::jsonb,
    'Heavy emphasis on technical skills match',
    FALSE
) ON CONFLICT (config_name) DO NOTHING;

-- ============================================================================
-- TABLE 6: match_history (Track scoring runs for audit)
-- ============================================================================
CREATE TABLE IF NOT EXISTS match_history (
    id BIGSERIAL PRIMARY KEY,
    applicant_id UUID NOT NULL REFERENCES applicant_profiles(applicant_id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES job_descriptions(job_id) ON DELETE CASCADE,
    
    -- Scores
    overall_score DECIMAL(5,4) NOT NULL,
    section_scores JSONB,  -- Individual section similarity scores
    
    -- Config used
    config_name VARCHAR(100),
    weights_used JSONB,
    
    -- Metadata
    scored_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(applicant_id, job_id, scored_at)
);

CREATE INDEX idx_match_history_applicant ON match_history(applicant_id);
CREATE INDEX idx_match_history_job ON match_history(job_id);
CREATE INDEX idx_match_history_score ON match_history(overall_score DESC);
CREATE INDEX idx_match_history_scored_at ON match_history(scored_at DESC);

-- ============================================================================
-- TRIGGERS
-- ============================================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_applicant_profiles_updated_at
    BEFORE UPDATE ON applicant_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_job_descriptions_updated_at
    BEFORE UPDATE ON job_descriptions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- RLS POLICIES (Enable if needed)
-- ============================================================================

ALTER TABLE applicant_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE applicant_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE job_descriptions ENABLE ROW LEVEL SECURITY;
ALTER TABLE job_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE match_history ENABLE ROW LEVEL SECURITY;

-- Allow all operations (customize based on your auth requirements)
CREATE POLICY "Allow all on profiles" ON applicant_profiles FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on embeddings" ON applicant_embeddings FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on jobs" ON job_descriptions FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on job_embeddings" ON job_embeddings FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow all on match_history" ON match_history FOR ALL USING (true) WITH CHECK (true);

-- ============================================================================
-- HELPER VIEWS
-- ============================================================================

-- View for quick applicant lookup with section counts
CREATE OR REPLACE VIEW applicant_summary AS
SELECT 
    p.applicant_id,
    p.name,
    p.email,
    p.match_score,
    p.needs_manual_review,
    p.created_at,
    COUNT(DISTINCT e.section_type) as section_count,
    AVG(e.confidence_score) as avg_confidence,
    MIN(e.confidence_score) as min_confidence
FROM applicant_profiles p
LEFT JOIN applicant_embeddings e ON p.applicant_id = e.applicant_id AND e.is_active = TRUE
GROUP BY p.applicant_id, p.name, p.email, p.match_score, p.needs_manual_review, p.created_at;

-- ============================================================================
-- UTILITY FUNCTIONS
-- ============================================================================

-- Function to compute cosine similarity (if not using built-in operator)
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector)
RETURNS FLOAT AS $$
BEGIN
    RETURN 1 - (a <=> b);
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to deactivate old embeddings when updating resume
CREATE OR REPLACE FUNCTION deactivate_old_embeddings()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE applicant_embeddings 
    SET is_active = FALSE 
    WHERE applicant_id = NEW.applicant_id 
      AND version < NEW.version;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_deactivate_old_embeddings
    AFTER INSERT ON applicant_embeddings
    FOR EACH ROW
    EXECUTE FUNCTION deactivate_old_embeddings();
