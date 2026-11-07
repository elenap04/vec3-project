import argparse
import os
from pathlib import Path
from time import perf_counter

import chromadb
import numpy as np
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector


DATASET = Path("datasets/synthetic_1m_d128.npz")
NAME = "vec3_bench_1m_d128"

EXPECTED_COUNT = 1_000_000
DIMENSIONS = 128
BATCH_SIZE = 500


def read_dataset():
    if not DATASET.is_file():
        raise FileNotFoundError(
            f"Dataset is not ready: {DATASET}"
        )

    with np.load(DATASET, allow_pickle=False) as data:
        vectors = data["vectors"].copy()
        queries = data["queries"].copy()
        ids = data["ids"].copy()
        buckets = data["buckets"].copy()

    if vectors.shape != (
        EXPECTED_COUNT,
        DIMENSIONS,
    ):
        raise ValueError(
            f"Wrong vector shape: {vectors.shape}"
        )

    if vectors.dtype != np.float32:
        raise ValueError(
            f"Wrong vector dtype: {vectors.dtype}"
        )

    if queries.ndim != 2:
        raise ValueError("Queries must be 2D")

    if queries.shape[1] != DIMENSIONS:
        raise ValueError("Wrong query dimensions")

    if queries.dtype != np.float32:
        raise ValueError("Wrong query dtype")

    if ids.shape != (EXPECTED_COUNT,):
        raise ValueError("Wrong ID count")

    if buckets.shape != (EXPECTED_COUNT,):
        raise ValueError("Wrong bucket count")

    if len(np.unique(ids)) != EXPECTED_COUNT:
        raise ValueError("Duplicate vector IDs")

    if not np.isfinite(vectors).all():
        raise ValueError("Invalid base vectors")

    if not np.isfinite(queries).all():
        raise ValueError("Invalid query vectors")

    counts = np.bincount(
        buckets.astype(np.int32),
        minlength=100,
    )

    if not np.array_equal(
        counts,
        np.full(100, EXPECTED_COUNT // 100),
    ):
        raise ValueError(
            "Metadata buckets are not balanced"
        )

    print("=== DATASET VALIDATION ===")
    print(f"Base vectors: {vectors.shape}")
    print(f"Queries: {queries.shape}")
    print(f"Vector dtype: {vectors.dtype}")
    print(f"Unique IDs: {len(np.unique(ids))}")
    print("Balanced buckets: YES")
    print("DATASET VALIDATION PASSED")

    return vectors, ids, buckets


def load_chroma(vectors, ids, buckets):
    print("\n=== CHROMA LOADING ===", flush=True)

    client = chromadb.HttpClient(
        host="127.0.0.1",
        port=8000,
    )

    client.heartbeat()

    collection = client.get_or_create_collection(
        name=NAME,
        embedding_function=None,
    )

    print(
        f"Existing vectors: {collection.count()}",
        flush=True,
    )

    started = perf_counter()

    for start in range(
        0,
        EXPECTED_COUNT,
        BATCH_SIZE,
    ):
        end = min(
            start + BATCH_SIZE,
            EXPECTED_COUNT,
        )

        collection.upsert(
            ids=ids[start:end].tolist(),
            embeddings=vectors[start:end].tolist(),
            metadatas=[
                {"bucket": int(bucket)}
                for bucket in buckets[start:end]
            ],
        )

        if end % 50_000 == 0:
            print(
                f"Chroma loaded: {end:,}/{EXPECTED_COUNT:,}",
                flush=True,
            )

    elapsed = perf_counter() - started
    final_count = collection.count()

    print("\n=== CHROMA SUMMARY ===")
    print(f"Collection: {NAME}")
    print(f"Vectors: {final_count:,}")
    print(f"Elapsed: {elapsed:.2f} seconds")

    if final_count != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNT} vectors, "
            f"found {final_count}"
        )

    print("CHROMA 1M LOADING PASSED")


def load_pgvector(vectors, ids, buckets):
    print("\n=== PGVECTOR LOADING ===", flush=True)

    password = os.environ.get(
        "POSTGRES_PASSWORD"
    )

    if not password:
        raise RuntimeError(
            "POSTGRES_PASSWORD is not set. "
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

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS
                vec3_bench_1m_d128 (
                    id TEXT PRIMARY KEY,
                    bucket INTEGER NOT NULL,
                    embedding vector(128) NOT NULL
                )
            """
        )

        conn.commit()

        existing = conn.execute(
            """
            SELECT COUNT(*)
            FROM vec3_bench_1m_d128
            """
        ).fetchone()[0]

        print(
            f"Existing rows: {existing:,}",
            flush=True,
        )

        insert_sql = """
            INSERT INTO vec3_bench_1m_d128
                (id, bucket, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (id)
            DO UPDATE SET
                bucket = EXCLUDED.bucket,
                embedding = EXCLUDED.embedding
        """

        started = perf_counter()

        for start in range(
            0,
            EXPECTED_COUNT,
            BATCH_SIZE,
        ):
            end = min(
                start + BATCH_SIZE,
                EXPECTED_COUNT,
            )

            rows = [
                (
                    str(ids[i]),
                    int(buckets[i]),
                    Vector(vectors[i]),
                )
                for i in range(start, end)
            ]

            with conn.cursor() as cursor:
                cursor.executemany(
                    insert_sql,
                    rows,
                )

            conn.commit()

            if end % 50_000 == 0:
                print(
                    f"pgvector loaded: "
                    f"{end:,}/{EXPECTED_COUNT:,}",
                    flush=True,
                )

        elapsed = perf_counter() - started

        final_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM vec3_bench_1m_d128
            """
        ).fetchone()[0]

    print("\n=== PGVECTOR SUMMARY ===")
    print(f"Table: {NAME}")
    print(f"Rows: {final_count:,}")
    print(f"Elapsed: {elapsed:.2f} seconds")

    if final_count != EXPECTED_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COUNT} rows, "
            f"found {final_count}"
        )

    print("PGVECTOR 1M LOADING PASSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        choices=["chroma", "pgvector"],
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.validate_only and not args.db:
        parser.error(
            "--db is required unless "
            "--validate-only is used"
        )

    print("=== SYNTHETIC 1M DATASET LOADER ===")
    print(f"Dataset: {DATASET}")
    print(f"Target: {NAME}")

    vectors, ids, buckets = read_dataset()

    if args.validate_only:
        print("Validation only: no database changes")
        return

    print("Mode: development data loading")
    print("Timing is NOT an official benchmark")

    if args.db == "chroma":
        load_chroma(
            vectors,
            ids,
            buckets,
        )
    else:
        load_pgvector(
            vectors,
            ids,
            buckets,
        )


if __name__ == "__main__":
    main()
