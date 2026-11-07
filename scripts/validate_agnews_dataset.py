import argparse
import os
from pathlib import Path

import chromadb
import numpy as np
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector


DATASET = Path("datasets/agnews_100k_d384.npz")
NAME = "vec3_agnews_100k_d384"

EXPECTED_COUNT = 100000
DIMENSIONS = 384
BATCH_SIZE = 500
TOLERANCE = 1e-6


def read_source():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset is not ready: {DATASET}"
        )

    with np.load(DATASET, allow_pickle=False) as data:
        vectors = data["vectors"].copy()
        ids = data["ids"].copy()
        buckets = data["buckets"].copy()
        labels = data["labels"].copy()

    if vectors.shape != (EXPECTED_COUNT, DIMENSIONS):
        raise ValueError("Incorrect source vector shape")

    if vectors.dtype != np.float32:
        raise ValueError("Incorrect source vector dtype")

    if ids.shape != (EXPECTED_COUNT,):
        raise ValueError("Incorrect source IDs")

    if buckets.shape != (EXPECTED_COUNT,):
        raise ValueError("Incorrect source buckets")

    if labels.shape != (EXPECTED_COUNT,):
        raise ValueError("Incorrect source labels")

    if len(set(ids.tolist())) != EXPECTED_COUNT:
        raise ValueError("Duplicate source IDs")

    if not np.isfinite(vectors).all():
        raise ValueError("Invalid source vectors")

    counts = np.bincount(
        buckets.astype(np.int32),
        minlength=100,
    )

    if not np.array_equal(
        counts,
        np.full(100, 1000),
    ):
        raise ValueError("Source buckets are not balanced")

    return vectors, ids, buckets, labels


def check_vector(actual, expected, vector_id, database):
    """
    Accept NumPy arrays, lists and pgvector.Vector objects.

    Newer pgvector-python releases return Vector objects
    when reading PostgreSQL vector columns.
    """

    if isinstance(actual, Vector):
        actual = actual.to_numpy()

    actual = np.asarray(actual, dtype=np.float32)

    if actual.shape != (DIMENSIONS,):
        raise AssertionError(
            f"{database}: wrong dimension for {vector_id}"
        )

    if not np.isfinite(actual).all():
        raise AssertionError(
            f"{database}: non-finite vector for {vector_id}"
        )

    if not np.allclose(
        actual,
        expected,
        rtol=0,
        atol=TOLERANCE,
    ):
        difference = np.max(
            np.abs(actual - expected)
        )

        raise AssertionError(
            f"{database}: vector mismatch for {vector_id}; "
            f"max difference={difference}"
        )


def validate_chroma(vectors, ids, buckets, labels, positions):
    print("\n=== CHROMA VALIDATION ===", flush=True)

    client = chromadb.HttpClient(
        host="127.0.0.1",
        port=8000,
    )

    collection = client.get_collection(NAME)

    count = collection.count()

    print(f"Database count: {count}")

    if count != EXPECTED_COUNT:
        raise AssertionError(
            f"Chroma count mismatch: {count}"
        )

    checked = 0

    for start in range(0, len(positions), BATCH_SIZE):
        batch_positions = positions[
            start:start + BATCH_SIZE
        ]

        requested_ids = [
            str(ids[i])
            for i in batch_positions
        ]

        expected_rows = {
            str(ids[i]): int(i)
            for i in batch_positions
        }

        response = collection.get(
            ids=requested_ids,
            include=["embeddings", "metadatas"],
        )

        returned_ids = response["ids"]
        embeddings = response["embeddings"]
        metadatas = response["metadatas"]

        if len(returned_ids) != len(requested_ids):
            raise AssertionError(
                "Chroma returned an incorrect number of rows"
            )

        if set(returned_ids) != set(requested_ids):
            raise AssertionError(
                "Chroma returned incorrect IDs"
            )

        for row_index, vector_id in enumerate(returned_ids):
            source_index = expected_rows[vector_id]

            check_vector(
                embeddings[row_index],
                vectors[source_index],
                vector_id,
                "Chroma",
            )

            metadata = metadatas[row_index]

            if int(metadata["bucket"]) != int(
                buckets[source_index]
            ):
                raise AssertionError(
                    f"Chroma bucket mismatch: {vector_id}"
                )

            if int(metadata["label"]) != int(
                labels[source_index]
            ):
                raise AssertionError(
                    f"Chroma label mismatch: {vector_id}"
                )

            checked += 1

        print(
            f"Chroma checked: {checked}/{len(positions)}",
            flush=True,
        )

    print("CHROMA VALIDATION PASSED")


def validate_pgvector(vectors, ids, buckets, labels, positions):
    print("\n=== PGVECTOR VALIDATION ===", flush=True)

    password = os.environ.get("POSTGRES_PASSWORD")

    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD is missing. "
            "Load the project's .env first."
        )

    with psycopg.connect(
        host="127.0.0.1",
        port=5432,
        dbname="vec3db",
        user="vec3",
        password=password,
    ) as conn:
        register_vector(conn)

        count = conn.execute(
            """
            SELECT COUNT(*)
            FROM vec3_agnews_100k_d384
            """
        ).fetchone()[0]

        print(f"Database count: {count}")

        if count != EXPECTED_COUNT:
            raise AssertionError(
                f"pgvector count mismatch: {count}"
            )

        checked = 0

        for start in range(0, len(positions), BATCH_SIZE):
            batch_positions = positions[
                start:start + BATCH_SIZE
            ]

            requested_ids = [
                str(ids[i])
                for i in batch_positions
            ]

            expected_rows = {
                str(ids[i]): int(i)
                for i in batch_positions
            }

            rows = conn.execute(
                """
                SELECT id, bucket, label, embedding
                FROM vec3_agnews_100k_d384
                WHERE id = ANY(%s)
                """,
                (requested_ids,),
            ).fetchall()

            if len(rows) != len(requested_ids):
                raise AssertionError(
                    "pgvector returned an incorrect number of rows"
                )

            returned_ids = {
                row[0]
                for row in rows
            }

            if returned_ids != set(requested_ids):
                raise AssertionError(
                    "pgvector returned incorrect IDs"
                )

            for vector_id, bucket, label, embedding in rows:
                source_index = expected_rows[vector_id]

                check_vector(
                    embedding,
                    vectors[source_index],
                    vector_id,
                    "pgvector",
                )

                if int(bucket) != int(
                    buckets[source_index]
                ):
                    raise AssertionError(
                        f"pgvector bucket mismatch: {vector_id}"
                    )

                if int(label) != int(
                    labels[source_index]
                ):
                    raise AssertionError(
                        f"pgvector label mismatch: {vector_id}"
                    )

                checked += 1

            print(
                f"pgvector checked: {checked}/{len(positions)}",
                flush=True,
            )

    print("PGVECTOR VALIDATION PASSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of sampled vectors to check",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Validate all 100000 vectors",
    )

    args = parser.parse_args()

    if args.sample_size < 1:
        parser.error("--sample-size must be positive")

    print("=== AG NEWS DATABASE VALIDATION ===")
    print(f"Dataset: {DATASET}")
    print(f"Table/collection: {NAME}")

    vectors, ids, buckets, labels = read_source()

    if args.full:
        positions = np.arange(EXPECTED_COUNT)
        print("Mode: FULL validation")
    else:
        rng = np.random.default_rng(42)

        positions = np.sort(
            rng.choice(
                EXPECTED_COUNT,
                size=min(args.sample_size, EXPECTED_COUNT),
                replace=False,
            )
        )

        print(
            f"Mode: SAMPLE ({len(positions)} vectors)"
        )

    validate_chroma(
        vectors,
        ids,
        buckets,
        labels,
        positions,
    )

    validate_pgvector(
        vectors,
        ids,
        buckets,
        labels,
        positions,
    )

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Vectors checked per database: {len(positions)}")
    print("IDs: PASSED")
    print("Vector values: PASSED")
    print("Metadata buckets: PASSED")
    print("AG News labels: PASSED")
    print("AG NEWS DATABASE VALIDATION PASSED")


if __name__ == "__main__":
    main()
