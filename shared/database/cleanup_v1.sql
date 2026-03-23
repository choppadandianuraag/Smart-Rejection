-- Cleanup V1 Database Schema
-- Run this in Supabase SQL Editor to remove old V1 tables

-- Drop V1 table and related objects
DROP TABLE IF EXISTS resumes CASCADE;

-- Drop V1 functions if they exist
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;

-- Optional: Drop job_descriptions table if it exists from V1
DROP TABLE IF EXISTS job_descriptions CASCADE;

-- Verify cleanup
SELECT 
    tablename 
FROM 
    pg_tables 
WHERE 
    schemaname = 'public' 
    AND tablename IN ('resumes', 'job_descriptions');

-- Expected result: 0 rows (tables deleted successfully)
