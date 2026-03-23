-- Smart Rejection - Supabase Database Schema
-- Run this SQL in your Supabase SQL Editor

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- RESUMES TABLE
CREATE TABLE IF NOT EXISTS resumes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    markdown_content TEXT NOT NULL,
    extracted_data JSONB DEFAULT '{}'::jsonb,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding_vector FLOAT8[] DEFAULT NULL,
    embedding_model TEXT DEFAULT NULL,
    processing_status TEXT DEFAULT 'pending',
    error_message TEXT DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_resumes_filename ON resumes(filename);
CREATE INDEX IF NOT EXISTS idx_resumes_status ON resumes(processing_status);
CREATE INDEX IF NOT EXISTS idx_resumes_created_at ON resumes(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_resumes_extracted_data ON resumes USING GIN(extracted_data);

-- Auto-update trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_resumes_updated_at ON resumes;
CREATE TRIGGER update_resumes_updated_at
    BEFORE UPDATE ON resumes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- RLS Policy
ALTER TABLE resumes ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all operations on resumes" ON resumes;
CREATE POLICY "Allow all operations on resumes" ON resumes
    FOR ALL USING (true) WITH CHECK (true);

-- JOB DESCRIPTIONS TABLE (Phase 2)
CREATE TABLE IF NOT EXISTS job_descriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    company TEXT,
    location TEXT,
    job_type TEXT,
    description TEXT NOT NULL,
    requirements TEXT,
    required_skills JSONB DEFAULT '[]'::jsonb,
    preferred_skills JSONB DEFAULT '[]'::jsonb,
    experience_years INTEGER,
    embedding_vector FLOAT8[] DEFAULT NULL,
    embedding_model TEXT DEFAULT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE job_descriptions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all operations on job_descriptions" ON job_descriptions;
CREATE POLICY "Allow all operations on job_descriptions" ON job_descriptions
    FOR ALL USING (true) WITH CHECK (true);

-- RESUME-JOB MATCHES TABLE (Phase 2)
CREATE TABLE IF NOT EXISTS resume_job_matches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    resume_id UUID REFERENCES resumes(id) ON DELETE CASCADE,
    job_id UUID REFERENCES job_descriptions(id) ON DELETE CASCADE,
    cosine_similarity FLOAT8,
    keyword_match_score FLOAT8,
    skills_match_score FLOAT8,
    overall_score FLOAT8,
    matching_skills JSONB DEFAULT '[]'::jsonb,
    missing_skills JSONB DEFAULT '[]'::jsonb,
    feedback JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(resume_id, job_id)
);

ALTER TABLE resume_job_matches ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Allow all operations on matches" ON resume_job_matches;
CREATE POLICY "Allow all operations on matches" ON resume_job_matches
    FOR ALL USING (true) WITH CHECK (true);

CREATE INDEX IF NOT EXISTS idx_matches_resume ON resume_job_matches(resume_id);
CREATE INDEX IF NOT EXISTS idx_matches_job ON resume_job_matches(job_id);
CREATE INDEX IF NOT EXISTS idx_matches_score ON resume_job_matches(overall_score DESC);
