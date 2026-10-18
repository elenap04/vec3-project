CREATE INDEX IF NOT EXISTS vec3_pilot_10k_hnsw_l2_idx
ON vec3_pilot_10k
USING hnsw (embedding vector_l2_ops)
WITH (
    m = 16,
    ef_construction = 100
);

ANALYZE vec3_pilot_10k;
