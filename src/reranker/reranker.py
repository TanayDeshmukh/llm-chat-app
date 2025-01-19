from abc import ABC, abstractmethod
from typing import List, Optional

from rerankers import Reranker


from data.data_classes import VectorDocumentChunk, WebDocumentChunk


class ReRanker(ABC):
    @abstractmethod
    def run_reranker(
        self,
        query: str,
        documents: List[VectorDocumentChunk | WebDocumentChunk],
    ) -> List[VectorDocumentChunk | WebDocumentChunk]:
        pass


class DefaultReRanker(ReRanker):
    def __init__(self, top_k: Optional[int] = None):
        self.reranker = Reranker(
            model_name="cross-encoder", model_type="cross-encoder", verbose=0
        )
        self.top_k = top_k

    def run_reranker(
        self,
        query: str,
        documents: List[VectorDocumentChunk | WebDocumentChunk],
    ) -> List[VectorDocumentChunk | WebDocumentChunk]:

        return_k = self.top_k if self.top_k is not None else len(documents)

        document_text_list = [doc.text for doc in documents]
        document_ids = list(range(len(documents)))

        reranked_results = self.reranker.rank(
            query=query, docs=document_text_list, doc_ids=document_ids
        ).results

        reranked_document_ids = [res.document.doc_id for res in reranked_results]
        reranked_documents = [documents[i] for i in reranked_document_ids]

        return reranked_documents[:return_k]


def load_reranker(provider: str, top_k: int, **kwargs) -> Reranker:
    if provider == "default":
        return DefaultReRanker()
    else:
        print(f"No reranker implemented for {provider=}")
