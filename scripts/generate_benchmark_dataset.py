import argparse
from pathlib import Path
from time import perf_counter

import numpy as np


SELECTIVITIES = (100, 50, 10, 1)
K = 10


def compute_ground_truth(vectors, queries, buckets, ids):
    # Float64 arithmetic for reliable exact L2 rankings.
    base = vectors.astype(np.float64)
    query_vectors = queries.astype(np.float64)

    base_norms = np.einsum("ij,ij->i", base, base)

    eligible_indices = {
        s: np.flatnonzero(buckets < s)
        for s in SELECTIVITIES
    }

    ground_truth = {
        s: np.empty((len(queries), K), dtype=ids.dtype)
        for s in SELECTIVITIES
    }

    batch_size = 8

    for start in range(0, len(queries), batch_size):
        end = min(start + batch_size, len(queries))
        batch = query_vectors[start:end]

        query_norms = np.einsum(
            "ij,ij->i", batch, batch
        )

        distances = (
            query_norms[:, None]
            + base_norms[None, :]
            - 2.0 * (batch @ base.T)
        )

        for local_i in range(len(batch)):
            global_i = start + local_i

            for selectivity in SELECTIVITIES:
                eligible = eligible_indices[selectivity]

                candidate_distances = (
                    distances[local_i, eligible]
                )

                positions = np.argpartition(
                    candidate_distances,
                    K - 1,
                )[:K]

                positions = positions[
                    np.argsort(
                        candidate_distances[positions],
                        kind="stable",
                    )
                ]

                nearest_indices = eligible[positions]

                ground_truth[selectivity][global_i] = (
                    ids[nearest_indices]
                )

        print(
            f"Computed ground truth: "
            f"{end}/{len(queries)} queries",
            flush=True,
        )

    return ground_truth


def validate_dataset(
    vectors, queries, buckets, ids, ground_truth
):
    assert vectors.dtype == np.float32
    assert queries.dtype == np.float32
    assert len(ids) == len(vectors)
    assert len(buckets) == len(vectors)
    assert len(set(ids.tolist())) == len(ids)

    for selectivity in SELECTIVITIES:
        eligible = np.flatnonzero(buckets < selectivity)

        assert len(eligible) >= K

        expected_count = (
            len(vectors) * selectivity // 100
        )

        assert len(eligible) == expected_count

        assert ground_truth[selectivity].shape == (
            len(queries), K
        )

        # Independent brute-force validation for query #0.
        base = vectors[eligible].astype(np.float64)
        query = queries[0].astype(np.float64)

        differences = base - query

        squared_distances = np.einsum(
            "ij,ij->i",
            differences,
            differences,
        )

        nearest = np.argsort(
            squared_distances
        )[:K]

        expected_ids = ids[eligible[nearest]]
        actual_ids = ground_truth[selectivity][0]

        if not np.array_equal(
            expected_ids, actual_ids
        ):
            raise AssertionError(
                f"Ground truth mismatch at "
                f"filter {selectivity}%"
            )

        print(
            f"Filter {selectivity}% VALIDATED: "
            f"{len(eligible)} vectors"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--size",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/synthetic_100k_d128.npz"
        ),
    )

    args = parser.parse_args()

    if args.size < 100 or args.size % 100 != 0:
        parser.error(
            "--size must be a multiple of 100"
        )

    if args.dimensions < 1 or args.queries < 1:
        parser.error(
            "dimensions and queries must be positive"
        )

    if args.output.exists():
        parser.error(
            f"Output already exists: {args.output}"
        )

    started = perf_counter()
    rng = np.random.default_rng(args.seed)

    print("Generating vectors...", flush=True)

    vectors = rng.standard_normal(
        (args.size, args.dimensions),
        dtype=np.float32,
    )

    queries = rng.standard_normal(
        (args.queries, args.dimensions),
        dtype=np.float32,
    )

    # Exactly balanced metadata buckets, 0 through 99.
    buckets = np.repeat(
        np.arange(100, dtype=np.int16),
        args.size // 100,
    )

    rng.shuffle(buckets)

    ids = np.array([
        f"v{i:07d}"
        for i in range(args.size)
    ])

    print("Computing exact ground truth...", flush=True)

    ground_truth = compute_ground_truth(
        vectors,
        queries,
        buckets,
        ids,
    )

    print("Validating dataset...", flush=True)

    validate_dataset(
        vectors,
        queries,
        buckets,
        ids,
        ground_truth,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Saving dataset...", flush=True)

    np.savez_compressed(
        args.output,
        vectors=vectors,
        queries=queries,
        ids=ids,
        buckets=buckets,
        gt_100=ground_truth[100],
        gt_50=ground_truth[50],
        gt_10=ground_truth[10],
        gt_1=ground_truth[1],
        seed=np.int64(args.seed),
    )

    elapsed = perf_counter() - started

    print("\n=== DATASET SUMMARY ===")
    print(f"Output: {args.output}")
    print(f"Vectors: {vectors.shape}")
    print(f"Queries: {queries.shape}")
    print(f"Vector dtype: {vectors.dtype}")
    print(f"Seed: {args.seed}")
    print(f"Generation time: {elapsed:.2f} seconds")
    print("DATASET GENERATION PASSED")


if __name__ == "__main__":
    main()
