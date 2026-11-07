\timing on

-- Vec3 synthetic 1M / 128D PostgreSQL HNSW index.
-- Metric: Euclidean distance (L2).
-- Construction parameters match the 100K experiments.
--
-- Development setup, not an official benchmark.
-- Run only AFTER loading the 1M table.

CREATE INDEX IF NOT EXISTS
    vec3_bench_1m_d128_hnsw_l2_idx
ON vec3_bench_1m_d128
USING hnsw (embedding vector_l2_ops)
WITH (
    m = 16,
    ef_construction = 100
);

ANALYZE vec3_bench_1m_d128;

SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'vec3_bench_1m_d128'
ORDER BY indexname;
