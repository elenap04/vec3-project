
import os

import chromadb
import numpy as np
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector


DATASET = "datasets/synthetic_10k_d128.npz"
K = 10

with np.load(DATASET) as data:
    queries = data["queries"]
    ground_truth = data["gt_100"]


def recall_at_k(returned, exact):
    return len(set(returned) & set(exact)) / K


chroma = chromadb.HttpClient(
    host="127.0.0.1",
    port=8000
)

collection = chroma.get_collection(
    name="vec3_pilot_10k",
    embedding_function=None
)

chroma_recalls = []
pgvector_recalls = []

with psycopg.connect(
    host="127.0.0.1",
    port=5432,
    dbname="vec3db",
    user="vec3",
    password=os.environ["POSTGRES_PASSWORD"]
) as conn:

    register_vector(conn)

    conn.execute("SET hnsw.ef_search = 80")

    for i, query in enumerate(queries):
        exact = ground_truth[i].tolist()

        # Chroma
        result = collection.query(
            query_embeddings=[query.tolist()],
            n_results=K
        )

        chroma_ids = result["ids"][0]

        chroma_recalls.append(
            recall_at_k(chroma_ids, exact)
        )

        # pgvector
        rows = conn.execute(
            """
            SELECT id
            FROM vec3_pilot_10k
            ORDER BY embedding <-> %s
            LIMIT 10
            """,
            (Vector(query),)
        ).fetchall()

        pg_ids = [row[0] for row in rows]

        pgvector_recalls.append(
            recall_at_k(pg_ids, exact)
        )

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(queries)} queries")


print("\n=== RECALL PILOT RESULTS ===")

for name, recalls in (
    ("Chroma", chroma_recalls),
    ("pgvector", pgvector_recalls)
):
    print(f"\n{name}")
    print(f"Mean Recall@10: {np.mean(recalls):.4f}")
    print(f"Min Recall@10:  {np.min(recalls):.4f}")
    print(f"Max Recall@10:  {np.max(recalls):.4f}")

print("\nRECALL PILOT COMPLETED")

