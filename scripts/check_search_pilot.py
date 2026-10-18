import os

import chromadb
import numpy as np
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector


# Load one query and its exact nearest neighbors
with np.load("datasets/synthetic_10k_d128.npz") as data:
    query = data["queries"][0]
    exact_ids = data["gt_100"][0].tolist()


def recall_at_10(returned_ids, expected_ids):
    return len(set(returned_ids) & set(expected_ids)) / 10


print("=== EXACT GROUND TRUTH ===")
print(exact_ids)


# -----------------------------------
# Chroma search
# -----------------------------------
chroma = chromadb.HttpClient(
    host="127.0.0.1",
    port=8000
)

collection = chroma.get_collection(
    name="vec3_pilot_10k",
    embedding_function=None
)

chroma_result = collection.query(
    query_embeddings=[query.tolist()],
    n_results=10
)

chroma_ids = chroma_result["ids"][0]

print("\n=== CHROMA ===")
print("Returned IDs:", chroma_ids)
print("Recall@10:", recall_at_10(chroma_ids, exact_ids))


# -----------------------------------
# pgvector search
# -----------------------------------
with psycopg.connect(
    host="127.0.0.1",
    port=5432,
    dbname="vec3db",
    user="vec3",
    password=os.environ["POSTGRES_PASSWORD"]
) as conn:

    register_vector(conn)

    # Initial HNSW search setting
    conn.execute("SET hnsw.ef_search = 80")

    sql = """
        SELECT id
        FROM vec3_pilot_10k
        ORDER BY embedding <-> %s
        LIMIT 10
    """

    vector_param = Vector(query)

    print("\n=== POSTGRESQL QUERY PLAN ===")

    plan = conn.execute(
        "EXPLAIN (ANALYZE, BUFFERS) " + sql,
        (vector_param,)
    ).fetchall()

    for (line,) in plan:
        print(line)

    rows = conn.execute(
        sql,
        (vector_param,)
    ).fetchall()

    pg_ids = [row[0] for row in rows]

    print("\n=== PGVECTOR ===")
    print("Returned IDs:", pg_ids)
    print("Recall@10:", recall_at_10(pg_ids, exact_ids))


print("\n=== SEARCH PILOT COMPLETED ===")

