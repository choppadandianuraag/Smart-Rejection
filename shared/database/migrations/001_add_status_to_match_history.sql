-- Migration: Add status column to match_history table
-- Run this on existing databases to add applicant status tracking

-- Create the enum type if it doesn't exist
DO $$ BEGIN
    CREATE TYPE applicant_status_enum AS ENUM (
        'selected',   -- Top 10% of applicants
        'feedback',   -- Top 50% of non-selected applicants (10%-55% range)
        'rejected'    -- Remaining applicants (bottom 45%)
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Add the status column to match_history table
ALTER TABLE match_history
ADD COLUMN IF NOT EXISTS status applicant_status_enum;

-- Add index for status column
CREATE INDEX IF NOT EXISTS idx_match_history_status ON match_history(status);

-- Comment explaining the status values
COMMENT ON COLUMN match_history.status IS 'Applicant status based on score percentiles: selected (top 10%), feedback (next 45%), rejected (bottom 45%)';
