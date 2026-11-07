import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np


SELECTIVITIES = (100, 50, 10, 1)
K = 10


def stable_top_k(distances, candidate_indices):
    """
    Return exact top-K row indices.

    Break equal-distance ties using the original
    row index, without sorting all candidates.
    """
    if len(distances) < K:
        raise ValueError("Not enough candidates for top-K")

    threshold = np.partition(
        distances,
        K - 1,
    )[K - 1]

    closer = np.flatnonzero(
        distances < threshold
    )

    tied = np.flatnonzero(
        distances == threshold
    )

    needed = K - len(closer)

    if needed < 0 or len(tied) < needed:
        raise RuntimeError(
            "Invalid top-K boundary"
        )

    # candidate_indices are in ascending row order.
    # Therefore the earliest tied positions win.
    selected = np.concatenate(
        (closer, tied[:needed])
    )

    order = np.lexsort(
        (
            candidate_indices[selected],
            distances[selected],
        )
    )

    return candidate_indices[selected[order]]


def compute_ground_truth(
    vectors,
    queries,
    buckets,
    ids,
    batch_size=8,
):
    """
    Compute exact squared-L2 rankings.

    Squared L2 and L2 produce the same ordering.
    Process queries in batches to limit memory usage.
    """
    if batch_size < 1:
        raise ValueError(
            "Batch size must be positive"
        )

    base = np.asarray(
        vectors,
        dtype=np.float64,
    )

    query_vectors = np.asarray(
        queries,
        dtype=np.float64,
    )

    if base.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError(
            "Vectors and queries must be 2D"
        )

    if base.shape[1] != query_vectors.shape[1]:
        raise ValueError(
            "Vector dimension mismatch"
        )

    if not np.isfinite(base).all():
        raise ValueError(
            "Non-finite base vectors"
        )

    if not np.isfinite(query_vectors).all():
        raise ValueError(
            "Non-finite query vectors"
        )

    base_norms = np.einsum(
        "ij,ij->i",
        base,
        base,
    )

    eligible_indices = {
        selectivity: np.flatnonzero(
            buckets < selectivity
        )
        for selectivity in SELECTIVITIES
    }

    for selectivity, eligible in eligible_indices.items():
        if len(eligible) < K:
            raise ValueError(
                f"Too few vectors for {selectivity}%"
            )

    ground_truth = {
        selectivity: np.empty(
            (len(queries), K),
            dtype=ids.dtype,
        )
        for selectivity in SELECTIVITIES
    }

    for start in range(
        0,
        len(query_vectors),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(query_vectors),
        )

        batch = query_vectors[start:end]

        query_norms = np.einsum(
            "ij,ij->i",
            batch,
            batch,
        )

        distances = (
            query_norms[:, None]
            + base_norms[None, :]
            - 2.0 * (batch @ base.T)
        )

        np.maximum(
            distances,
            0.0,
            out=distances,
        )

        if not np.isfinite(distances).all():
            raise ValueError(
                "Non-finite computed distances"
            )

        for local_index in range(len(batch)):
            query_index = start + local_index

            for selectivity in SELECTIVITIES:
                eligible = eligible_indices[
                    selectivity
                ]

                candidate_distances = (
                    distances[local_index, eligible]
                )

                nearest_indices = stable_top_k(
                    candidate_distances,
                    eligible,
                )

                ground_truth[selectivity][
                    query_index
                ] = ids[nearest_indices]

        print(
            f"Computed ground truth: "
            f"{end}/{len(queries)} queries",
            flush=True,
        )

    return ground_truth


def direct_top_k_blockwise(
    vectors,
    query,
    eligible,
    block_size=8192,
):
    """
    Independent direct-distance validation.

    Avoid allocating a full N x dimensions
    float64 difference matrix.
    """
    query64 = np.asarray(
        query,
        dtype=np.float64,
    )

    distances = np.empty(
        len(eligible),
        dtype=np.float64,
    )

    for start in range(
        0,
        len(eligible),
        block_size,
    ):
        end = min(
            start + block_size,
            len(eligible),
        )

        block_indices = eligible[start:end]

        block = np.asarray(
            vectors[block_indices],
            dtype=np.float64,
        )

        difference = block - query64

        distances[start:end] = np.einsum(
            "ij,ij->i",
            difference,
            difference,
        )

    order = np.lexsort(
        (
            eligible,
            distances,
        )
    )

    return eligible[order[:K]]


def validate_dataset(
    vectors,
    queries,
    buckets,
    ids,
    ground_truth,
):
    if vectors.dtype != np.float32:
        raise ValueError("Invalid base dtype")

    if queries.dtype != np.float32:
        raise ValueError("Invalid query dtype")

    if len(ids) != len(vectors):
        raise ValueError("Invalid ID count")

    if len(buckets) != len(vectors):
        raise ValueError("Invalid bucket count")

    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Duplicate IDs")

    if len(queries) < 1:
        raise ValueError("No queries")

    if np.any(buckets < 0) or np.any(buckets > 99):
        raise ValueError("Invalid buckets")

    counts = np.bincount(
        buckets.astype(np.int32),
        minlength=100,
    )

    if not np.array_equal(
        counts,
        np.full(100, len(vectors) // 100),
    ):
        raise ValueError(
            "Metadata buckets are not balanced"
        )

    for selectivity in SELECTIVITIES:
        eligible = np.flatnonzero(
            buckets < selectivity
        )

        expected_count = (
            len(vectors) * selectivity // 100
        )

        if len(eligible) != expected_count:
            raise ValueError(
                f"Incorrect {selectivity}% population"
            )

        actual = ground_truth[selectivity]

        if actual.shape != (len(queries), K):
            raise ValueError(
                f"Invalid {selectivity}% GT shape"
            )

        for row in actual:
            if len(set(row.tolist())) != K:
                raise ValueError(
                    "Duplicate ground truth IDs"
                )

        nearest_indices = direct_top_k_blockwise(
            vectors,
            queries[0],
            eligible,
        )

        expected_ids = ids[nearest_indices]

        np.testing.assert_array_equal(
            actual[0],
            expected_ids,
            err_msg=(
                f"Ground truth mismatch "
                f"at filter {selectivity}%"
            ),
        )

        print(
            f"Filter {selectivity}% VALIDATED: "
            f"{len(eligible)} vectors",
            flush=True,
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
        "--batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/synthetic_100k_d128.npz"
        ),
    )

    args = parser.parse_args()

    if args.size < 1000 or args.size % 100 != 0:
        parser.error(
            "--size must be a multiple of 100 "
            "and at least 1000"
        )

    if args.dimensions < 1 or args.queries < 1:
        parser.error(
            "Dimensions and queries must be positive"
        )

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be positive"
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

    ids = np.asarray([
        f"v{i:07d}"
        for i in range(args.size)
    ])

    buckets = np.repeat(
        np.arange(100, dtype=np.int16),
        args.size // 100,
    )

    rng.shuffle(buckets)

    print(
        "Computing exact ground truth...",
        flush=True,
    )

    ground_truth = compute_ground_truth(
        vectors,
        queries,
        buckets,
        ids,
        batch_size=args.batch_size,
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

    temporary = args.output.with_name(
        args.output.name + ".tmp"
    )

    print("Saving dataset safely...", flush=True)

    try:
        with temporary.open("wb") as file:
            np.savez_compressed(
                file,
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

        os.replace(
            temporary,
            args.output,
        )

    except BaseException:
        temporary.unlink(
            missing_ok=True
        )
        raise

    elapsed = perf_counter() - started

    print("\n=== DATASET SUMMARY ===")
    print(f"Output: {args.output}")
    print(f"Vectors: {vectors.shape}")
    print(f"Queries: {queries.shape}")
    print(f"Vector dtype: {vectors.dtype}")
    print(f"Seed: {args.seed}")
    print(
        f"Size: "
        f"{args.output.stat().st_size / 1024**2:.1f} MiB"
    )
    print(
        f"Generation time: {elapsed:.2f} seconds"
    )
    print("DATASET GENERATION PASSED")


if __name__ == "__main__":
    main()
