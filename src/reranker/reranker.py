from typing import List
from rerankers import Reranker


from data.data_classes import VectorDocumentChunk, WebDocumentChunk


class ReRanker:
    def __init__(self):
        self.reranker = Reranker(
            model_name="cross-encoder", model_type="cross-encoder", verbose=0
        )

    def run_reranker(
        self, query: str, documents: List[VectorDocumentChunk | WebDocumentChunk]
    ):
        document_text_list = [doc.text for doc in documents]
        document_ids = list(range(len(documents)))

        reranked_results = self.reranker.rank(
            query=query, docs=document_text_list, doc_ids=document_ids
        ).results

        reranked_document_ids = [res.document.doc_id for res in reranked_results]
        reranked_documents = [documents[i] for i in reranked_document_ids]

        return reranked_documents
