import argparse
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


def read_dataset():
    with np.load(DATASET, allow_pickle=False) as data:
        vectors = data["vectors"].copy()
        ids = data["ids"].copy()
        buckets = data["buckets"].copy()

    if vectors.shape != (100000, 128):
        raise ValueError(
            f"Unexpected vector shape: {vectors.shape}"
        )

    if vectors.dtype != np.float32:
        raise ValueError(
            f"Unexpected vector dtype: {vectors.dtype}"
        )

    if len(ids) != len(vectors):
        raise ValueError("ID count mismatch")

    if len(buckets) != len(vectors):
        raise ValueError("Metadata count mismatch")

    if not np.isfinite(vectors).all():
        raise ValueError("Non-finite vector values")

    return vectors, ids, buckets


def load_chroma(vectors, ids, buckets):
    client = chromadb.HttpClient(
        host="127.0.0.1",
        port=8000,
    )

    collection = client.get_or_create_collection(
        name=NAME,
        embedding_function=None,
    )

    existing = collection.count()

    print(f"Existing Chroma vectors: {existing}")

    started = perf_counter()

    for start in range(0, len(ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(ids))

        batch_ids = ids[start:end].tolist()
        batch_vectors = vectors[start:end].tolist()

        batch_metadata = [
            {"bucket": int(bucket)}
            for bucket in buckets[start:end]
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
            f"Expected {len(ids)} vectors, "
            f"found {final_count}"
        )

    print("CHROMA LOADING PASSED")


def load_pgvector(vectors, ids, buckets):
    with psycopg.connect(
        host="127.0.0.1",
        port=5432,
        dbname="vec3db",
        user="vec3",
        password=os.environ["POSTGRES_PASSWORD"],
    ) as conn:

        register_vector(conn)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS
                vec3_bench_100k_d128 (
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
            FROM vec3_bench_100k_d128
            """
        ).fetchone()[0]

        print(f"Existing pgvector rows: {existing}")

        started = perf_counter()

        insert_sql = """
            INSERT INTO vec3_bench_100k_d128
                (id, bucket, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (id)
            DO UPDATE SET
                bucket = EXCLUDED.bucket,
                embedding = EXCLUDED.embedding
        """

        for start in range(0, len(ids), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(ids))

            rows = [
                (
                    str(ids[i]),
                    int(buckets[i]),
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
            FROM vec3_bench_100k_d128
            """
        ).fetchone()[0]

    print("\n=== PGVECTOR LOADING SUMMARY ===")
    print(f"Table: {NAME}")
    print(f"Vectors: {final_count}")
    print(f"Elapsed: {elapsed:.2f} seconds")

    if final_count != len(ids):
        raise RuntimeError(
            f"Expected {len(ids)} vectors, "
            f"found {final_count}"
        )

    print("PGVECTOR LOADING PASSED")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        choices=["chroma", "pgvector"],
        required=True,
    )

    args = parser.parse_args()

    print(f"Dataset: {DATASET}")
    print(f"Target: {args.db}")
    print("Reading dataset...", flush=True)

    vectors, ids, buckets = read_dataset()

    print(
        f"Loaded dataset in memory: "
        f"{vectors.shape}",
        flush=True,
    )

    if args.db == "chroma":
        load_chroma(vectors, ids, buckets)
    else:
        load_pgvector(vectors, ids, buckets)


if __name__ == "__main__":
    main()
