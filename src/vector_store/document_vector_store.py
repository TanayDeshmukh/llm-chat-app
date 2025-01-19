import json
import os
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from opensearchpy import OpenSearch, helpers

from data.data_classes import VectorDocumentChunk
from vector_store.embedder import load_embedder


class VectorDB:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[
                {
                    "host": st.session_state.config.opensearch_config.host,
                    "port": st.session_state.config.opensearch_config.port,
                }
            ],
            http_compress=True,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
            use_ssl=True,
            verify_certs=False,
            http_auth=("admin", os.environ["OPENSEARCH_INITIAL_ADMIN_PASSWORD"]),
        )
        self.index = st.session_state.config.opensearch_config.index_name
        self._init_index(index=self.index)
        self.embedder = load_embedder(
            provider=st.session_state.config.embedding_model_config.model_provider,
            model_name=st.session_state.config.embedding_model_config.model_name,
        )

    @staticmethod
    def _load_index_config(config_name: str) -> Dict[str, Any]:
        config_path = (
            Path(os.getcwd()) / Path("vector_store/opensearch_configs") / config_name
        )
        with open(config_path, "r") as f:
            config = json.load(f)
        return config

    def add_documents(
        self, documents: List[VectorDocumentChunk], upload_batch_size: int = 10
    ) -> int:
        errors = 0
        progress_bar = st.progress(0, text="Embedding documents..")
        for i in range(0, len(documents), upload_batch_size):
            document_actions = [
                {
                    "_op_type": "index",
                    "_index": self.index,
                    "_id": doc.id,
                    "_source": {
                        "text": doc.text,
                        "page_num": doc.page_num,
                        "start_idx": doc.start_idx,
                        "paginated_text": doc.paginated_text,
                        "file_name": doc.file_name,
                        "embedding": self.embedder.encode(doc.text).tolist(),
                    },
                }
                for doc in documents[i : i + upload_batch_size]
            ]
            _, errors_ = helpers.bulk(client=self.client, actions=document_actions)
            errors += len(errors_)
            progress_bar.progress(
                value=int(i / len(documents) * 100), text="Embedding documents.."
            )
        progress_bar.empty()
        return errors

    def search(
        self,
        query_text: str,
        top_k: int = st.session_state.config.opensearch_config.use_top_k_embeddings,
    ) -> List[VectorDocumentChunk]:
        retrieved_documents = []
        query_embedding = self.embedder.encode(query_text).tolist()
        query = {
            "_source": {"exclude": ["embedding"]},
            "size": top_k,
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": 20}}},
        }
        response = self.client.search(index=self.index, body=query)
        response_score_threshold = (
            st.session_state.config.opensearch_config.filter_confidence_threshold
        )

        if response["hits"]:
            hits = response["hits"]["hits"]
            for hit in hits:
                if hit["_score"] > response_score_threshold:
                    doc = VectorDocumentChunk(
                        id=hit["_id"],
                        text=hit["_source"]["text"],
                        page_num=hit["_source"]["page_num"],
                        start_idx=hit["_source"]["start_idx"],
                        paginated_text=hit["_source"]["paginated_text"],
                        file_name=hit["_source"]["file_name"],
                    )
                    retrieved_documents.append(doc)
        return retrieved_documents

    def _init_index(
        self,
        index,
        config_name: str = "base_config.json",
    ):
        if not self.client.indices.exists(index=index):
            index_config = self._load_index_config(config_name)
            index_config["mappings"]["properties"]["embedding"][
                "dimension"
            ] = st.session_state.config.opensearch_config.embedding_dimension
            self.client.indices.create(index=index, body=index_config)
