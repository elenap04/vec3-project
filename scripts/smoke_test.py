
import os

import chromadb
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

vectors = {
    "a": [1.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0],
    "c": [0.0, 0.0, 1.0],
}
query = [0.8, 0.2, 0.0]

# Chroma
chroma = chromadb.HttpClient(
    host="127.0.0.1", port=8000
)

collection = chroma.get_or_create_collection(
    name="vec3_smoke",
    embedding_function=None,
)

collection.upsert(
    ids=list(vectors.keys()),
    embeddings=list(vectors.values()),
)

chroma_ids = collection.query(
    query_embeddings=[query],
    n_results=2,
)["ids"][0]

print("Chroma nearest IDs:", chroma_ids)

# PostgreSQL / pgvector
with psycopg.connect(
    host="127.0.0.1",
    port=5432,
    dbname="vec3db",
    user="vec3",
    password=os.environ["POSTGRES_PASSWORD"],
    autocommit=True,
) as conn:

    register_vector(conn)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS vec3_smoke (
            id TEXT PRIMARY KEY,
            embedding vector(3)
        )
    """)

    for key, embedding in vectors.items():
        conn.execute(
            """
            INSERT INTO vec3_smoke (id, embedding)
            VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE
            SET embedding = EXCLUDED.embedding
            """,
            (key, Vector(embedding)),
        )

    rows = conn.execute(
        """
        SELECT id FROM vec3_smoke
        ORDER BY embedding <-> %s
        LIMIT 2
        """,
        (Vector(query),),
    ).fetchall()

    pg_ids = [row[0] for row in rows]

print("pgvector nearest IDs:", pg_ids)

assert chroma_ids == pg_ids == ["a", "b"]
print("SMOKE TEST PASSED")
