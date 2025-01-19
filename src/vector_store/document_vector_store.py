import json
import os
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from opensearchpy import OpenSearch, helpers

from common.constants import (
    DEFAULT_EMBEDDING_MODEL,
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    DEFAULT_INDEX,
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_TOP_K_EMBEDDINGS,
    DEFAULT_EMBEDDER_TYPE,
)
from data.data_classes import VectorDocumentChunk
from vector_store.embedder import load_embedder


class VectorDB:
    def __init__(
        self,
        index: str = DEFAULT_INDEX,
    ):
        self.client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_compress=True,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
            use_ssl=True,
            verify_certs=False,
            http_auth=("admin", os.environ["OPENSEARCH_INITIAL_ADMIN_PASSWORD"]),
        )
        self.index = index
        self._init_index(index=self.index)
        self.embedder = load_embedder(
            embedder_type=DEFAULT_EMBEDDER_TYPE,
            model_identifier=DEFAULT_EMBEDDING_MODEL,
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
        self, query_text: str, top_k: int = DEFAULT_TOP_K_EMBEDDINGS
    ) -> List[VectorDocumentChunk]:
        retrieved_documents = []
        query_embedding = self.embedder.encode(query_text).tolist()
        query = {
            "_source": {"exclude": ["embedding"]},
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}},
        }
        response = self.client.search(index=self.index, body=query)

        if response["hits"]:
            hits = response["hits"]["hits"]
            for hit in hits:
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
            ] = DEFAULT_EMBEDDING_DIMENSION
            self.client.indices.create(index=index, body=index_config)
