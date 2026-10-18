
import chromadb
import numpy as np


class ChromaAdapter:
    def __init__(self, collection_name="vec3_pilot_10k"):
        self.client = chromadb.HttpClient(
            host="127.0.0.1",
            port=8000,
        )

        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=None,
        )

    def search(self, query, k=10, selectivity=100):
        if not 1 <= selectivity <= 100:
            raise ValueError("selectivity must be between 1 and 100")

        query = np.asarray(query, dtype=np.float32)

        kwargs = {
            "query_embeddings": [query.tolist()],
            "n_results": k,
        }

        if selectivity < 100:
            kwargs["where"] = {
                "bucket": {"$lt": selectivity}
            }

        result = self.collection.query(**kwargs)
        return result["ids"][0]

