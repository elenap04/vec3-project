import chromadb
import numpy as np


class ChromaAdapter:
    def __init__(
        self,
        collection_name="vec3_pilot_10k",
        candidate_count=None,
    ):
        if candidate_count is not None:
            if (
                not isinstance(candidate_count, int)
                or isinstance(candidate_count, bool)
                or candidate_count < 1
            ):
                raise ValueError(
                    "candidate_count must be a positive integer"
                )

        self.candidate_count = candidate_count

        self.client = chromadb.HttpClient(
            host="127.0.0.1",
            port=8000,
        )

        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=None,
        )

    def search(self, query, k=10, selectivity=100):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")

        if not 1 <= selectivity <= 100:
            raise ValueError(
                "selectivity must be between 1 and 100"
            )

        query = np.asarray(
            query,
            dtype=np.float32,
        )

        requested = (
            self.candidate_count
            if self.candidate_count is not None
            else k
        )

        if requested < k:
            raise ValueError(
                "candidate_count must be at least k"
            )

        kwargs = {
            "query_embeddings": [query.tolist()],
            "n_results": requested,
        }

        if selectivity < 100:
            kwargs["where"] = {
                "bucket": {"$lt": selectivity}
            }

        result = self.collection.query(**kwargs)

        ids = result["ids"][0]

        # Return only top-k IDs for Recall@k.
        # The complete Chroma request is performed before
        # returning, so query timing includes oversampling.
        return ids[:k]
