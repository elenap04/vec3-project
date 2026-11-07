import argparse
import os
from pathlib import Path
from time import perf_counter

import chromadb
import numpy as np
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector


DATASET = Path("datasets/agnews_100k_d384.npz")
NAME = "vec3_agnews_100k_d384"

BASE_COUNT = 100000
QUERY_COUNT = 200
DIMENSIONS = 384
BATCH_SIZE = 500


def read_dataset():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset not ready: {DATASET}\n"
            "Wait until embedding generation finishes."
        )

    with np.load(DATASET, allow_pickle=False) as data:
        vectors = data["vectors"].copy()
        queries = data["queries"].copy()
        ids = data["ids"].copy()
        buckets = data["buckets"].copy()
        labels = data["labels"].copy()

    if vectors.shape != (BASE_COUNT, DIMENSIONS):
        raise ValueError(
            f"Unexpected vector shape: {vectors.shape}"
        )

    if queries.shape != (QUERY_COUNT, DIMENSIONS):
        raise ValueError(
            f"Unexpected query shape: {queries.shape}"
        )

    if vectors.dtype != np.float32:
        raise ValueError(
            f"Unexpected vector dtype: {vectors.dtype}"
        )

    if queries.dtype != np.float32:
        raise ValueError(
            f"Unexpected query dtype: {queries.dtype}"
        )

    if ids.shape != (BASE_COUNT,):
        raise ValueError("ID count mismatch")

    if buckets.shape != (BASE_COUNT,):
        raise ValueError("Bucket count mismatch")

    if labels.shape != (BASE_COUNT,):
        raise ValueError("Label count mismatch")

    if len(set(ids.tolist())) != BASE_COUNT:
        raise ValueError("Duplicate vector IDs")

    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite base vectors")

    if not np.isfinite(queries).all():
        raise ValueError("Non-finite query vectors")

    if np.any(buckets < 0) or np.any(buckets > 99):
        raise ValueError("Invalid metadata buckets")

    if np.any(labels < 0) or np.any(labels > 3):
        raise ValueError("Invalid AG News labels")

    bucket_counts = np.bincount(
        buckets.astype(np.int32),
        minlength=100,
    )

    if not np.all(bucket_counts == 1000):
        raise ValueError(
            "Metadata buckets are not balanced"
        )

    print("=== DATASET VALIDATION ===")
    print(f"Base vectors: {vectors.shape}")
    print(f"Query vectors: {queries.shape}")
    print(f"Vector dtype: {vectors.dtype}")
    print(f"Unique IDs: {len(set(ids.tolist()))}")
    print(f"Labels: {np.unique(labels).tolist()}")
    print("Balanced metadata buckets: YES")
    print("DATASET VALIDATION PASSED")

    return vectors, ids, buckets, labels


def load_chroma(vectors, ids, buckets, labels):
    client = chromadb.HttpClient(
        host="127.0.0.1",
        port=8000,
    )

    client.heartbeat()

    collection = client.get_or_create_collection(
        name=NAME,
        embedding_function=None,
        metadata={"hnsw:space": "l2"},
    )

    existing = collection.count()

    print(f"Existing Chroma vectors: {existing}")
    print("Starting AG News Chroma loading...", flush=True)

    started = perf_counter()

    for start in range(0, len(ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(ids))

        batch_ids = ids[start:end].tolist()
        batch_vectors = vectors[start:end].tolist()

        batch_metadata = [
            {
                "bucket": int(buckets[i]),
                "label": int(labels[i]),
            }
            for i in range(start, end)
        ]

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_vectors,
            metadatas=batch_metadata,
        )

        if end % 10000 == 0 or end == len(ids):
            print(
                f"Chroma loaded: {end}/{len(ids)}",
                flush=True,
            )

    elapsed = perf_counter() - started
    final_count = collection.count()

    print("\n=== CHROMA LOADING SUMMARY ===")
    print(f"Collection: {NAME}")
    print(f"Vectors: {final_count}")
    print(f"Elapsed: {elapsed:.2f} seconds")

    if final_count != len(ids):
        raise RuntimeError(
            f"Expected {len(ids)} vectors, found {final_count}"
        )

    print("CHROMA LOADING PASSED")


def load_pgvector(vectors, ids, buckets, labels):
    password = os.environ.get("POSTGRES_PASSWORD")

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
                vec3_agnews_100k_d384 (
                    id TEXT PRIMARY KEY,
                    bucket INTEGER NOT NULL,
                    label INTEGER NOT NULL,
                    embedding vector(384) NOT NULL
                )
            """
        )

        conn.commit()

        existing = conn.execute(
            """
            SELECT COUNT(*)
            FROM vec3_agnews_100k_d384
            """
        ).fetchone()[0]

        print(f"Existing pgvector rows: {existing}")
        print("Starting AG News pgvector loading...", flush=True)

        started = perf_counter()

        insert_sql = """
            INSERT INTO vec3_agnews_100k_d384
                (id, bucket, label, embedding)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id)
            DO UPDATE SET
                bucket = EXCLUDED.bucket,
                label = EXCLUDED.label,
                embedding = EXCLUDED.embedding
        """

        for start in range(0, len(ids), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(ids))

            rows = [
                (
                    str(ids[i]),
                    int(buckets[i]),
                    int(labels[i]),
                    Vector(vectors[i]),
                )
                for i in range(start, end)
            ]

            with conn.cursor() as cursor:
                cursor.executemany(insert_sql, rows)

            conn.commit()

            if end % 10000 == 0 or end == len(ids):
                print(
                    f"pgvector loaded: {end}/{len(ids)}",
                    flush=True,
                )

        elapsed = perf_counter() - started

        final_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM vec3_agnews_100k_d384
            """
        ).fetchone()[0]

    print("\n=== PGVECTOR LOADING SUMMARY ===")
    print(f"Table: {NAME}")
    print(f"Vectors: {final_count}")
    print(f"Elapsed: {elapsed:.2f} seconds")

    if final_count != len(ids):
        raise RuntimeError(
            f"Expected {len(ids)} vectors, found {final_count}"
        )

    print("PGVECTOR LOADING PASSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        choices=["chroma", "pgvector"],
        help="Database to load",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the NPZ without connecting to a database",
    )

    args = parser.parse_args()

    if not args.validate_only and not args.db:
        parser.error(
            "--db is required unless --validate-only is used"
        )

    print("=== AG NEWS DATASET LOADER ===")
    print(f"Dataset: {DATASET}")
    print(f"Collection/table: {NAME}")

    vectors, ids, buckets, labels = read_dataset()

    if args.validate_only:
        print("Validation only: no database changes")
        return

    print(f"Target database: {args.db}")

    if args.db == "chroma":
        load_chroma(vectors, ids, buckets, labels)
    else:
        load_pgvector(vectors, ids, buckets, labels)


if __name__ == "__main__":
    main()
