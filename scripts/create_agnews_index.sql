\timing on

-- HNSW index for the real AG News 100K / 384D dataset.
-- Distance metric: L2.
-- Same HNSW construction settings as the synthetic 100K pilot.

CREATE INDEX IF NOT EXISTS
    vec3_agnews_100k_d384_hnsw_l2_idx
ON vec3_agnews_100k_d384
USING hnsw (embedding vector_l2_ops)
WITH (
    m = 16,
    ef_construction = 100
);

ANALYZE vec3_agnews_100k_d384;

SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'vec3_agnews_100k_d384'
ORDER BY indexname;
