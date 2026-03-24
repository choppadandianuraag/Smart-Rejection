-- Migration: Add feedback_emails table
-- Run this on existing databases to add feedback email tracking

-- Create the feedback status enum type if it doesn't exist
DO $$ BEGIN
    CREATE TYPE feedback_status_enum AS ENUM (
        'generated',   -- Email content generated
        'sent',        -- Email successfully sent
        'failed',      -- Email sending failed
        'pending'      -- Awaiting generation
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Create the feedback_emails table
CREATE TABLE IF NOT EXISTS feedback_emails (
    id BIGSERIAL PRIMARY KEY,
    applicant_id UUID NOT NULL REFERENCES applicant_profiles(applicant_id) ON DELETE CASCADE,
    job_id UUID NOT NULL REFERENCES job_descriptions(job_id) ON DELETE CASCADE,
    match_history_id BIGINT REFERENCES match_history(id) ON DELETE SET NULL,

    -- Email content
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    recipient_email VARCHAR(255) NOT NULL,
    recipient_name VARCHAR(255),

    -- Generation metadata
    match_score DECIMAL(5,4),
    llm_model VARCHAR(100),
    generation_time_ms INTEGER,

    -- Status tracking
    status feedback_status_enum DEFAULT 'generated',
    error_message TEXT,

    -- Email delivery tracking
    sent_at TIMESTAMPTZ,
    delivery_id VARCHAR(255),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for feedback_emails
CREATE INDEX IF NOT EXISTS idx_feedback_applicant ON feedback_emails(applicant_id);
CREATE INDEX IF NOT EXISTS idx_feedback_job ON feedback_emails(job_id);
CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback_emails(status);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback_emails(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_feedback_recipient ON feedback_emails(recipient_email);

-- Trigger for updated_at (uses existing function from schema)
DROP TRIGGER IF EXISTS update_feedback_emails_updated_at ON feedback_emails;
CREATE TRIGGER update_feedback_emails_updated_at
    BEFORE UPDATE ON feedback_emails
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Enable RLS
ALTER TABLE feedback_emails ENABLE ROW LEVEL SECURITY;

-- Allow all policy (customize based on auth requirements)
CREATE POLICY "Allow all on feedback_emails" ON feedback_emails FOR ALL USING (true) WITH CHECK (true);

-- Comments
COMMENT ON TABLE feedback_emails IS 'Stores generated feedback emails for tracking and audit';
COMMENT ON COLUMN feedback_emails.status IS 'Email status: generated, sent, failed, pending';
COMMENT ON COLUMN feedback_emails.delivery_id IS 'External email service tracking ID (e.g., SendGrid, SES)';
