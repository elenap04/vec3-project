
from pathlib import Path
import numpy as np

# Experiment configuration
SEED = 42
N = 10_000
DIM = 128
N_QUERIES = 100
K = 10
QUERY_BATCH_SIZE = 10

rng = np.random.default_rng(SEED)

# Generate base vectors and separate query vectors
vectors = rng.standard_normal((N, DIM)).astype(np.float32)
queries = rng.standard_normal(
    (N_QUERIES, DIM)
).astype(np.float32)

# Stable, unique IDs for both databases
ids = np.array([f"v{i:07d}" for i in range(N)])

# Balanced metadata buckets: 0 to 99
buckets = (np.arange(N) % 100).astype(np.int32)
rng.shuffle(buckets)

# Compute exact neighbors for each filter
ground_truth = {}

for selectivity in (100, 50, 10, 1):
    eligible = np.flatnonzero(buckets < selectivity)
    eligible_vectors = vectors[eligible]

    # Squared norms for Euclidean distance
    base_norms = np.sum(
        eligible_vectors.astype(np.float64) ** 2,
        axis=1
    )

    gt = np.empty((N_QUERIES, K), dtype=ids.dtype)

    for start in range(0, N_QUERIES, QUERY_BATCH_SIZE):
        end = min(start + QUERY_BATCH_SIZE, N_QUERIES)
        batch = queries[start:end].astype(np.float64)

        query_norms = np.sum(batch ** 2, axis=1)

        distances = (
            query_norms[:, None]
            + base_norms[None, :]
            - 2.0 * (batch @ eligible_vectors.T)
        )

        for row_idx, row in enumerate(distances):
            nearest = np.argpartition(row, K - 1)[:K]
            nearest = nearest[np.argsort(row[nearest])]

            gt[start + row_idx] = ids[eligible[nearest]]

    ground_truth[f"gt_{selectivity}"] = gt

    print(
        f"Filter {selectivity}%: "
        f"{len(eligible)} eligible vectors, "
        f"ground truth shape {gt.shape}"
    )

# Save all data locally
output_dir = Path("datasets")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "synthetic_10k_d128.npz"

np.savez_compressed(
    output_file,
    vectors=vectors,
    queries=queries,
    ids=ids,
    buckets=buckets,
    **ground_truth
)

print("\nDataset created successfully!")
print("Path:", output_file)
print("Vectors:", vectors.shape)
print("Queries:", queries.shape)
print("Seed:", SEED)
