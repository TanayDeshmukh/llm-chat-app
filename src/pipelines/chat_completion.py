from typing import List, Dict, Tuple

import streamlit as st

from common.constants import DEFAULT_GENERATOR_MODEL_IDENTIFIER, DEFAULT_SYSTEM_PROMPT
from data.data_classes import VectorDocumentChunk, WebDocumentChunk
from generator.generator_model import load_generator
from reranker.reranker import ReRanker
from vector_store.document_vector_store import VectorDB


class ChatCompletionPipeline:
    def __init__(self, vector_store: VectorDB):
        self.vector_store = vector_store
        self.reranker = self._load_reranker()
        self.generator = self._load_generator()

    def _load_generator(self):
        return load_generator(
            generator_type="hf_api", model_identifier=DEFAULT_GENERATOR_MODEL_IDENTIFIER
        )

    def _load_reranker(self):
        return ReRanker()

    def enrich_prompt(
        self,
        system_prompt: str,
        chat_history: List[Dict[str, str]],
        query: Dict[str, str],
        vector_documents: List[VectorDocumentChunk],
        web_documents: List[WebDocumentChunk],
    ) -> List[Dict[str, str]]:
        model_input = [{"role": "system", "content": system_prompt}]
        model_input.extend(chat_history)

        context_document_texts = []
        for doc in vector_documents:
            context_document_texts.append(doc.text)
        for doc in web_documents:
            context_document_texts.append(doc.text)

        if len(context_document_texts) > 0:
            context_string = (
                f"Use the following documents to answer the useer query: {'##'.join(context_document_texts)} "
                f"### Use the provided documents only if the help you answer the user query."
            )
            model_input.append({"role": "system", "content": context_string})

        model_input.append(query)

        return model_input

    def run_completion_pipeline(
        self,
        chat_history: List[Dict[str, str]],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Tuple[str, List[VectorDocumentChunk | WebDocumentChunk]]:

        user_query = chat_history[-1]
        chat_history = chat_history[:-1]

        vector_documents = []
        web_documents = []

        if st.session_state.use_rag:
            # retrieve documents
            vector_documents = self.vector_store.search(
                query_text=user_query["content"]
            )
            # rerank documents
            vector_documents = self.reranker.run_reranker(
                query=user_query["content"], documents=vector_documents
            )
        if st.session_state.use_web_search:
            pass

        model_input = self.enrich_prompt(
            system_prompt=system_prompt,
            chat_history=chat_history,
            query=user_query,
            vector_documents=vector_documents,
            web_documents=web_documents,
        )

        model_response = self.generator.generate_response(model_input)

        documents = vector_documents + web_documents

        return model_response, documents
