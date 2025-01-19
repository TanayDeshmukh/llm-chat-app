from typing import List

from common.constants import DEFAULT_TOP_K_EMBEDDINGS
from data.data_classes import VectorDocumentChunk
from vector_store.document_vector_store import VectorDB


def run_vector_retrieval(
    vector_db: VectorDB, query: str, top_k: int = DEFAULT_TOP_K_EMBEDDINGS
) -> List[VectorDocumentChunk]:
    retrievals = vector_db.search(query, top_k=top_k)
    return retrievals
