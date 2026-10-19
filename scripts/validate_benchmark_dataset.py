import os
from pathlib import Path
from time import perf_counter

import chromadb
import numpy as np
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector


DATASET = Path("datasets/synthetic_100k_d128.npz")
NAME = "vec3_bench_100k_d128"
BATCH_SIZE = 500
SELECTIVITIES = (100, 50, 10, 1)


def check_vector(actual, expected, database, vector_id):
    # pgvector can return a pgvector.Vector object.
    # Convert it to a NumPy array before comparison.
    if isinstance(actual, Vector):
        actual = actual.to_numpy()

    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)

    if actual.shape != expected.shape:
        raise AssertionError(
            f"{database}: wrong dimensions for {vector_id}: "
            f"{actual.shape} != {expected.shape}"
        )

    if not np.allclose(
        actual,
        expected,
        rtol=0,
        atol=1e-6,
    ):
        raise AssertionError(
            f"{database}: vector mismatch for {vector_id}"
        )


def main():
    started = perf_counter()

    print("Reading dataset...", flush=True)

    with np.load(DATASET, allow_pickle=False) as data:
        vectors = data["vectors"].copy()
        ids = data["ids"].astype(str).tolist()
        buckets = data["buckets"].copy()

    total = len(ids)

    if vectors.shape != (100000, 128):
        raise AssertionError(
            f"Unexpected vector shape: {vectors.shape}"
        )

    if len(buckets) != total:
        raise AssertionError("Metadata count mismatch")

    if len(set(ids)) != total:
        raise AssertionError("Duplicate IDs in dataset")

    if not np.isfinite(vectors).all():
        raise AssertionError("Invalid vector values")

    print(f"Dataset: {DATASET}")
    print(f"Expected vectors: {total}")
    print("Connecting to databases...", flush=True)

    chroma_client = chromadb.HttpClient(
        host="127.0.0.1",
        port=8000,
    )

    collection = chroma_client.get_collection(
        name=NAME,
    )

    with psycopg.connect(
        host="127.0.0.1",
        port=5432,
        dbname="vec3db",
        user="vec3",
        password=os.environ["POSTGRES_PASSWORD"],
    ) as conn:

        register_vector(conn)

        chroma_count = collection.count()

        pg_count = conn.execute(
            f"SELECT COUNT(*) FROM {NAME}"
        ).fetchone()[0]

        print(f"Chroma count: {chroma_count}")
        print(f"pgvector count: {pg_count}")

        if chroma_count != total:
            raise AssertionError(
                f"Chroma count mismatch: {chroma_count}"
            )

        if pg_count != total:
            raise AssertionError(
                f"pgvector count mismatch: {pg_count}"
            )

        print(
            "\nChecking all vectors, IDs and metadata...",
            flush=True,
        )

        checked = 0

        for start in range(0, total, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total)

            batch_ids = ids[start:end]

            # Read the same IDs from Chroma.
            chroma_result = collection.get(
                ids=batch_ids,
                include=["embeddings", "metadatas"],
            )

            chroma_ids = chroma_result["ids"]
            chroma_embeddings = chroma_result["embeddings"]
            chroma_metadatas = chroma_result["metadatas"]

            if chroma_embeddings is None:
                raise AssertionError(
                    "Chroma returned no embeddings"
                )

            if chroma_metadatas is None:
                raise AssertionError(
                    "Chroma returned no metadata"
                )

            if (
                len(chroma_ids) != len(batch_ids)
                or len(chroma_embeddings) != len(batch_ids)
                or len(chroma_metadatas) != len(batch_ids)
            ):
                raise AssertionError(
                    f"Chroma incomplete batch: {start}"
                )

            chroma_rows = {
                vector_id: (embedding, metadata)
                for vector_id, embedding, metadata in zip(
                    chroma_ids,
                    chroma_embeddings,
                    chroma_metadatas,
                )
            }

            # Read the same IDs from PostgreSQL.
            pg_results = conn.execute(
                f"""
                SELECT id, bucket, embedding
                FROM {NAME}
                WHERE id = ANY(%s)
                """,
                (batch_ids,),
            ).fetchall()

            pg_rows = {
                vector_id: (bucket, embedding)
                for vector_id, bucket, embedding
                in pg_results
            }

            expected_id_set = set(batch_ids)

            if set(chroma_rows) != expected_id_set:
                raise AssertionError(
                    f"Chroma ID mismatch in batch {start}"
                )

            if set(pg_rows) != expected_id_set:
                raise AssertionError(
                    f"pgvector ID mismatch in batch {start}"
                )

            for i, vector_id in enumerate(batch_ids):
                source_index = start + i

                expected_vector = vectors[source_index]
                expected_bucket = int(buckets[source_index])

                chroma_vector, chroma_metadata = (
                    chroma_rows[vector_id]
                )

                if chroma_metadata is None:
                    raise AssertionError(
                        f"Missing Chroma metadata: {vector_id}"
                    )

                if (
                    chroma_metadata.get("bucket")
                    != expected_bucket
                ):
                    raise AssertionError(
                        f"Chroma metadata mismatch: {vector_id}"
                    )

                check_vector(
                    chroma_vector,
                    expected_vector,
                    "Chroma",
                    vector_id,
                )

                pg_bucket, pg_vector = pg_rows[vector_id]

                if int(pg_bucket) != expected_bucket:
                    raise AssertionError(
                        f"pgvector metadata mismatch: {vector_id}"
                    )

                check_vector(
                    pg_vector,
                    expected_vector,
                    "pgvector",
                    vector_id,
                )

            checked += len(batch_ids)

            if checked % 10000 == 0 or checked == total:
                print(
                    f"Validated: {checked}/{total}",
                    flush=True,
                )

        print(
            "\nChecking metadata filter result sets...",
            flush=True,
        )

        for selectivity in SELECTIVITIES:
            expected_ids = {
                ids[i]
                for i in range(total)
                if int(buckets[i]) < selectivity
            }

            # Chroma filter:
            # bucket < selectivity
            chroma_filter_result = collection.get(
                where={
                    "bucket": {
                        "$lt": selectivity
                    }
                },
                include=[],
            )

            chroma_filtered_ids = set(
                chroma_filter_result["ids"]
            )

            # PostgreSQL filter:
            # bucket < selectivity
            pg_filtered_ids = {
                row[0]
                for row in conn.execute(
                    f"""
                    SELECT id
                    FROM {NAME}
                    WHERE bucket < %s
                    """,
                    (selectivity,),
                )
            }

            if chroma_filtered_ids != expected_ids:
                raise AssertionError(
                    f"Chroma filter {selectivity}% mismatch"
                )

            if pg_filtered_ids != expected_ids:
                raise AssertionError(
                    f"pgvector filter {selectivity}% mismatch"
                )

            print(
                f"Filter {selectivity}% PASSED: "
                f"{len(expected_ids)} matching IDs",
                flush=True,
            )

    elapsed = perf_counter() - started

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Dataset vectors: {total}")
    print(f"Chroma vectors checked: {checked}")
    print(f"pgvector vectors checked: {checked}")
    print("All IDs: MATCH")
    print("All vectors: MATCH (absolute tolerance 1e-6)")
    print("All metadata: MATCH")
    print("All filter result sets: MATCH")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print("100K DATASET VALIDATION PASSED")


if __name__ == "__main__":
    main()
