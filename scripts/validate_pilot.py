
import os

import chromadb
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

with np.load("datasets/synthetic_10k_d128.npz") as data:
    vectors = data["vectors"]
    ids = data["ids"].tolist()
    buckets = data["buckets"]

chroma = chromadb.HttpClient(
    host="127.0.0.1", port=8000
)

collection = chroma.get_collection(
    name="vec3_pilot_10k",
    embedding_function=None
)

# Deterministic sampling
rng = np.random.default_rng(123)
indices = rng.choice(len(ids), 20, replace=False)
sample_ids = [ids[i] for i in indices]
expected = {ids[i]: i for i in indices}

# Validate Chroma
result = collection.get(
    ids=sample_ids,
    include=["embeddings", "metadatas"]
)

assert set(result["ids"]) == set(sample_ids)

for id_, embedding, metadata in zip(
    result["ids"],
    result["embeddings"],
    result["metadatas"]
):
    i = expected[id_]
    np.testing.assert_allclose(
        embedding, vectors[i], rtol=0, atol=1e-6
    )
    assert metadata["bucket"] == int(buckets[i])

print("CHROMA DATA VALIDATED")

# Validate PostgreSQL
with psycopg.connect(
    host="127.0.0.1",
    port=5432,
    dbname="vec3db",
    user="vec3",
    password=os.environ["POSTGRES_PASSWORD"]
) as conn:
    register_vector(conn)

    rows = conn.execute(
        """
        SELECT id, bucket, embedding
        FROM vec3_pilot_10k
        WHERE id = ANY(%s)
        """,
        (sample_ids,)
    ).fetchall()

    assert len(rows) == len(sample_ids)

    for id_, bucket, embedding in rows:
        i = expected[id_]
        assert bucket == int(buckets[i])
        np.testing.assert_allclose(
           embedding.to_numpy(), vectors[i], rtol=0, atol=1e-6
        )

    print("PGVECTOR DATA VALIDATED")

    # Verify exact filter membership
    for threshold in (50, 10, 1):
        expected_ids = set(
            np.asarray(ids)[buckets < threshold]
        )
        chroma_ids = set(collection.get(
            where={"bucket": {"$lt": threshold}},
            include=[]
        )["ids"])

        pg_ids = {
            row[0] for row in conn.execute(
                """
                SELECT id FROM vec3_pilot_10k
                WHERE bucket < %s
                """,
                (threshold,)
            ).fetchall()
        }

        assert chroma_ids == pg_ids == expected_ids

        print(
            f"Filter {threshold}% VALIDATED: "
            f"{len(expected_ids)} vectors"
        )

print("ALL DATA INTEGRITY CHECKS PASSED")
