
import os

import chromadb
import numpy as np
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector

DATASET = "datasets/synthetic_10k_d128.npz"
BATCH_SIZE = 500
NAME = "vec3_pilot_10k"

# Load shared dataset
with np.load(DATASET) as data:
    vectors = data["vectors"]
    ids = data["ids"].tolist()
    buckets = data["buckets"].tolist()

n = len(ids)
dim = vectors.shape[1]

assert len(vectors) == len(buckets) == n

print(f"Loaded dataset: {n} vectors, {dim} dimensions")

# -----------------------------------
# Chroma ingestion
# -----------------------------------
print("\nLoading Chroma...")

client = chromadb.HttpClient(
    host="127.0.0.1",
    port=8000
)

collection = client.get_or_create_collection(
    name=NAME,
    embedding_function=None
)

for start in range(0, n, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n)

    collection.upsert(
        ids=ids[start:end],
        embeddings=vectors[start:end].tolist(),
        metadatas=[
            {"bucket": int(b)}
            for b in buckets[start:end]
        ]
    )

    print(f"Chroma: {end}/{n}")

chroma_count = collection.count()
assert chroma_count == n

print("Chroma count:", chroma_count)

# -----------------------------------
# pgvector ingestion
# -----------------------------------
print("\nLoading pgvector...")

with psycopg.connect(
    host="127.0.0.1",
    port=5432,
    dbname="vec3db",
    user="vec3",
    password=os.environ["POSTGRES_PASSWORD"]
) as conn:

    register_vector(conn)

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {NAME} (
            id TEXT PRIMARY KEY,
            bucket INTEGER NOT NULL,
            embedding vector({dim}) NOT NULL
        )
    """)

    with conn.cursor() as cur:
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)

            batch = [
                (
                    ids[i],
                    int(buckets[i]),
                    Vector(vectors[i])
                )
                for i in range(start, end)
            ]

            cur.executemany(
                f"""
                INSERT INTO {NAME} (id, bucket, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET bucket = EXCLUDED.bucket,
                    embedding = EXCLUDED.embedding
                """,
                batch
            )

            print(f"pgvector: {end}/{n}")

    pg_count = conn.execute(
        f"SELECT COUNT(*) FROM {NAME}"
    ).fetchone()[0]

    assert pg_count == n

    print("pgvector count:", pg_count)

# -----------------------------------
# Final validation
# -----------------------------------
print("\n--------------------------")
print("Chroma vectors:", chroma_count)
print("pgvector vectors:", pg_count)
print("--------------------------")

assert chroma_count == pg_count == n

print("DATA LOADING PILOT PASSED")
