
import os

import numpy as np
import psycopg

from pgvector import Vector
from pgvector.psycopg import register_vector


class PgvectorAdapter:
    def __init__(
        self,
        table_name="vec3_pilot_10k",
        ef_search=80,
    ):
        self.table_name = table_name

        self.conn = psycopg.connect(
            host="127.0.0.1",
            port=5432,
            dbname="vec3db",
            user="vec3",
            password=os.environ["POSTGRES_PASSWORD"],
            autocommit=True,
        )

        register_vector(self.conn)

        self.conn.execute(
            "SELECT set_config('hnsw.ef_search', %s, false)",
            (str(ef_search),),
        )

        self.conn.execute(
            "SELECT set_config('hnsw.iterative_scan', %s, false)",
            ("strict_order",),
        )

    def search(self, query, k=10, selectivity=100):
        if not 1 <= selectivity <= 100:
            raise ValueError("selectivity must be between 1 and 100")

        query = Vector(np.asarray(query, dtype=np.float32))

        if selectivity == 100:
            sql = f"""
                SELECT id
                FROM {self.table_name}
                ORDER BY embedding <-> %s
                LIMIT %s
            """
            params = (query, k)
        else:
            sql = f"""
                SELECT id
                FROM {self.table_name}
                WHERE bucket < %s
                ORDER BY embedding <-> %s
                LIMIT %s
            """
            params = (selectivity, query, k)

        rows = self.conn.execute(sql, params).fetchall()
        return [row[0] for row in rows]

    def close(self):
        self.conn.close()

