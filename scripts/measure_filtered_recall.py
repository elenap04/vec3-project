
import os

import chromadb
import numpy as np
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector


DATASET = "datasets/synthetic_10k_d128.npz"
K = 10
SELECTIVITIES = (100, 50, 10, 1)

with np.load(DATASET) as data:
    queries = data["queries"].copy()
    ground_truths = {
        s: data[f"gt_{s}"].copy()
        for s in SELECTIVITIES
    }


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


with psycopg.connect(
    host="127.0.0.1",
    port=5432,
    dbname="vec3db",
    user="vec3",
    password=os.environ["POSTGRES_PASSWORD"]
) as conn:

    register_vector(conn)

    conn.execute("SET hnsw.ef_search = 80")
    conn.execute(
        "SET hnsw.iterative_scan = strict_order"
    )

    for selectivity in SELECTIVITIES:

        print(f"\n=== FILTER {selectivity}% ===")

        chroma_recalls = []
        pg_recalls = []

        chroma_short = 0
        pg_short = 0

        for i, query in enumerate(queries):
            exact = ground_truths[selectivity][i]

            # -------------------------
            # Chroma
            # -------------------------
            kwargs = {
                "query_embeddings": [query.tolist()],
                "n_results": K,
            }

            if selectivity < 100:
                kwargs["where"] = {
                    "bucket": {"$lt": selectivity}
                }

            result = collection.query(**kwargs)
            chroma_ids = result["ids"][0]

            chroma_recalls.append(
                recall_at_k(chroma_ids, exact)
            )

            if len(chroma_ids) < K:
                chroma_short += 1

            # -------------------------
            # pgvector
            # -------------------------
            if selectivity == 100:
                rows = conn.execute(
                    """
                    SELECT id
                    FROM vec3_pilot_10k
                    ORDER BY embedding <-> %s
                    LIMIT %s
                    """,
                    (Vector(query), K)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id
                    FROM vec3_pilot_10k
                    WHERE bucket < %s
                    ORDER BY embedding <-> %s
                    LIMIT %s
                    """,
                    (selectivity, Vector(query), K)
                ).fetchall()

            pg_ids = [row[0] for row in rows]

            pg_recalls.append(
                recall_at_k(pg_ids, exact)
            )

            if len(pg_ids) < K:
                pg_short += 1

        print(
            f"Chroma Mean Recall@10: "
            f"{np.mean(chroma_recalls):.4f}"
        )

        print(
            f"pgvector Mean Recall@10: "
            f"{np.mean(pg_recalls):.4f}"
        )

        print(
            f"Chroma queries with <10 results: "
            f"{chroma_short}"
        )

        print(
            f"pgvector queries with <10 results: "
            f"{pg_short}"
        )

print("\nFILTERED RECALL PILOT COMPLETED")

