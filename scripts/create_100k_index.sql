CREATE INDEX IF NOT EXISTS vec3_bench_100k_d128_hnsw_l2_idx
ON vec3_bench_100k_d128
USING hnsw (embedding vector_l2_ops)
WITH (
    m = 16,
    ef_construction = 100
);

ANALYZE vec3_bench_100k_d128;
