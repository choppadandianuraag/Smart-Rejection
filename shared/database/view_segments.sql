-- View Segmented Resume Data in Supabase
-- Copy these SQL queries and run them in Supabase SQL Editor

-- ===========================================================================
-- 1. VIEW ALL APPLICANT PROFILES
-- ===========================================================================
SELECT 
    applicant_id,
    name,
    email,
    contact_number,
    match_score,
    segmentation_confidence,
    needs_manual_review,
    review_reason,
    created_at
FROM applicant_profiles
ORDER BY created_at DESC;

-- ===========================================================================
-- 2. VIEW ALL SECTIONS (EMBEDDINGS) FOR EACH APPLICANT
-- ===========================================================================
SELECT 
    e.applicant_id,
    p.name,
    p.email,
    e.section_type,
    e.section_order,
    e.confidence_score,
    LENGTH(e.section_text) as text_length,
    LEFT(e.section_text, 80) as preview,
    e.version,
    e.is_active
FROM applicant_embeddings e
JOIN applicant_profiles p ON e.applicant_id = p.applicant_id
WHERE e.is_active = TRUE
ORDER BY e.applicant_id, e.section_order;

-- ===========================================================================
-- 3. COUNT SECTIONS PER APPLICANT
-- ===========================================================================
SELECT 
    p.name,
    p.email,
    COUNT(e.id) as section_count,
    AVG(e.confidence_score) as avg_confidence
FROM applicant_profiles p
LEFT JOIN applicant_embeddings e ON p.applicant_id = e.applicant_id AND e.is_active = TRUE
GROUP BY p.applicant_id, p.name, p.email
ORDER BY p.created_at DESC;

-- ===========================================================================
-- 4. VIEW SPECIFIC APPLICANT'S SECTIONS (Replace with your applicant_id)
-- ===========================================================================
SELECT 
    section_order,
    section_type,
    confidence_score,
    LENGTH(section_text) as chars,
    section_text  -- Full text
FROM applicant_embeddings
WHERE applicant_id = 'YOUR_APPLICANT_ID_HERE'  -- Replace this
  AND is_active = TRUE
ORDER BY section_order;

-- ===========================================================================
-- 5. SECTION TYPE BREAKDOWN
-- ===========================================================================
SELECT 
    section_type,
    COUNT(*) as count,
    ROUND(AVG(confidence_score::numeric), 3) as avg_confidence
FROM applicant_embeddings
WHERE is_active = TRUE
GROUP BY section_type
ORDER BY count DESC;

-- ===========================================================================
-- 6. CHECK IF EMBEDDINGS WERE STORED (Vector dimension check)
-- ===========================================================================
SELECT 
    e.applicant_id,
    p.name,
    e.section_type,
    array_length(e.embedding_vector, 1) as vector_dimensions
FROM applicant_embeddings e
JOIN applicant_profiles p ON e.applicant_id = p.applicant_id
WHERE e.is_active = TRUE
LIMIT 5;

-- Expected: vector_dimensions = 768 (BERT embeddings)
