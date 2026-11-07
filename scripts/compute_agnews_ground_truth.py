import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np


DATASET = Path("datasets/agnews_100k_d384.npz")
OUTPUT = Path("datasets/agnews_100k_d384_ground_truth.npz")

FILTERS = (100, 50, 10, 1)
K = 10


def exact_neighbors(vectors, queries, buckets, batch_size=8):
    """
    Calculate exact L2 nearest neighbors.

    Filter s means bucket < s.
    Equal distances are ordered by original row index.
    """

    vectors = np.asarray(vectors, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    buckets = np.asarray(buckets, dtype=np.int32)

    n = len(vectors)
    nq = len(queries)

    if vectors.ndim != 2 or queries.ndim != 2:
        raise ValueError("Vectors and queries must be 2D")

    if vectors.shape[1] != queries.shape[1]:
        raise ValueError("Dimension mismatch")

    if buckets.shape != (n,):
        raise ValueError("Invalid bucket array")

    if not np.isfinite(vectors).all():
        raise ValueError("Invalid base vectors")

    if not np.isfinite(queries).all():
        raise ValueError("Invalid query vectors")

    if batch_size < 1:
        raise ValueError("Batch size must be positive")

    for selectivity in FILTERS:
        eligible = np.count_nonzero(buckets < selectivity)

        if eligible < K:
            raise ValueError(
                f"Filter {selectivity}% has fewer than {K} vectors"
            )

    answers = {
        s: np.empty((nq, K), dtype=np.int32)
        for s in FILTERS
    }

    row_indices = np.arange(n, dtype=np.int32)

    # Squared Euclidean distance:
    # ||q - x||² = ||q||² + ||x||² - 2(q · x)
    vector_norms = np.sum(vectors * vectors, axis=1)

    for start in range(0, nq, batch_size):
        end = min(start + batch_size, nq)

        q = queries[start:end]
        query_norms = np.sum(q * q, axis=1)

        squared_distances = (
            query_norms[:, None]
            + vector_norms[None, :]
            - 2.0 * (q @ vectors.T)
        )

        np.maximum(
            squared_distances,
            0.0,
            out=squared_distances,
        )

        for local_idx in range(end - start):
            # Exact global ranking with deterministic tie-breaking.
            ranking = np.lexsort(
                (
                    row_indices,
                    squared_distances[local_idx],
                )
            )

            ranked_buckets = buckets[ranking]

            # Filter the same exact ranking for each selectivity.
            for selectivity in FILTERS:
                matching = ranking[
                    ranked_buckets < selectivity
                ]

                answers[selectivity][start + local_idx] = (
                    matching[:K]
                )

        print(
            f"Ground truth progress: {end}/{nq} queries",
            flush=True,
        )

    return answers


def run_self_test():
    print("=== EXACT GROUND TRUTH SELF-TEST ===")

    rng = np.random.default_rng(42)

    vectors = rng.normal(
        size=(2000, 8)
    ).astype(np.float32)

    queries = rng.normal(
        size=(3, 8)
    ).astype(np.float32)

    buckets = np.arange(2000) % 100

    result = exact_neighbors(
        vectors,
        queries,
        buckets,
        batch_size=2,
    )

    for q_idx, query in enumerate(queries):
        distances = np.sum(
            (
                vectors.astype(np.float64)
                - query.astype(np.float64)
            ) ** 2,
            axis=1,
        )

        for selectivity in FILTERS:
            eligible = np.flatnonzero(
                buckets < selectivity
            )

            ordering = np.lexsort(
                (
                    eligible,
                    distances[eligible],
                )
            )

            expected = eligible[ordering[:K]]
            actual = result[selectivity][q_idx]

            np.testing.assert_array_equal(
                actual,
                expected,
            )

    print("All four filters verified")
    print("All three queries verified")
    print("EXACT GROUND TRUTH SELF-TEST PASSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if not DATASET.exists():
        raise SystemExit(
            f"Dataset not ready: {DATASET}"
        )

    if OUTPUT.exists():
        raise SystemExit(
            f"Ground truth already exists: {OUTPUT}"
        )

    print("=== AG NEWS EXACT GROUND TRUTH ===", flush=True)

    with np.load(DATASET, allow_pickle=False) as data:
        vectors = data["vectors"]
        queries = data["queries"]
        buckets = data["buckets"]
        ids = data["ids"]

        metadata_json = str(data["metadata_json"])

    if vectors.shape != (100000, 384):
        raise ValueError(
            f"Unexpected base shape: {vectors.shape}"
        )

    if queries.shape != (200, 384):
        raise ValueError(
            f"Unexpected query shape: {queries.shape}"
        )

    if ids.shape != (100000,):
        raise ValueError("Unexpected ID shape")

    if not np.array_equal(
        np.bincount(buckets, minlength=100),
        np.full(100, 1000),
    ):
        raise ValueError(
            "Metadata buckets are not balanced"
        )

    print(f"Base vectors: {vectors.shape}", flush=True)
    print(f"Queries: {queries.shape}", flush=True)
    print("Metric: squared L2 (same ranking as L2)", flush=True)

    started = perf_counter()

    answers = exact_neighbors(
        vectors,
        queries,
        buckets,
        batch_size=args.batch_size,
    )

    output_data = {
        "metric": np.array("l2"),
        "k": np.int64(K),
        "metadata_json": np.array(metadata_json),
    }

    for selectivity in FILTERS:
        positions = answers[selectivity]

        output_data[f"gt_indices_{selectivity}"] = (
            positions
        )

        output_data[f"gt_{selectivity}"] = (
            ids[positions]
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = OUTPUT.with_suffix(".npz.tmp")

    with temporary.open("wb") as file:
        np.savez_compressed(
            file,
            **output_data,
        )

    os.replace(temporary, OUTPUT)

    print("\n=== GROUND TRUTH COMPLETE ===")
    print(f"Output: {OUTPUT}")
    print(f"Elapsed: {perf_counter() - started:.2f} s")

    for selectivity in FILTERS:
        print(
            f"Filter {selectivity}%: "
            f"{answers[selectivity].shape}"
        )

    print("EXACT GROUND TRUTH PASSED")


if __name__ == "__main__":
    main()
