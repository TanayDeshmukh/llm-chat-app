import os
import torch
import numpy as np
import streamlit as st

from typing import List
from openai import OpenAI
from abc import ABC, abstractmethod
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from tenacity import retry, wait_random_exponential, stop_after_attempt


class Embedder(ABC):
    @abstractmethod
    def encode(self, text: str) -> np.array:
        pass


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_identifier: str, trust_remote_code=False):
        self.embedder = self._load_pretrained_mode(model_identifier, trust_remote_code)

    @staticmethod
    def _load_pretrained_mode(
        model_identifier: str, trust_remote_code: bool = False
    ) -> SentenceTransformer:
        pretrained_model_dir = (
            st.session_state.config.embedding_model_config.cache_dir / model_identifier
        )
        if not pretrained_model_dir.exists():
            embedder = SentenceTransformer(
                model_name_or_path=model_identifier,
                device="cuda" if torch.cuda.is_available() else "cpu",
                trust_remote_code=trust_remote_code,
            )
            embedder.save_pretrained(str(pretrained_model_dir))
        else:
            embedder = SentenceTransformer(str(pretrained_model_dir))
        return embedder

    def encode(self, sentences: List[str]) -> np.array:
        encoding = self.embedder.encode(sentences=sentences)
        return encoding


class HFAPIEmbedder(Embedder):
    def __init__(self, model_identifier: str):
        hf_api_token = os.getenv("LLM_CHAT_APP_HF_API_TOKEN")
        self.embedder = InferenceClient(model_identifier, token=hf_api_token)

    def encode(self, sentences: List[str]) -> np.array:
        encoding = [
            self.embedder.feature_extraction(sentence) for sentence in sentences
        ]
        encoding = np.asarray(encoding)
        return encoding


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model_name

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def get_embedding(self, text: str) -> list[float]:
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )

    def encode(self, sentences: List[str]) -> np.array:
        encoding = [self.get_embedding(sentence) for sentence in sentences]
        encoding = np.asarray(encoding)
        return encoding


def load_embedder(provider: str, model_name: str) -> Embedder:
    if provider == "sentence_transformer":
        return SentenceTransformerEmbedder(model_name)
    elif provider == "hf_api":
        return HFAPIEmbedder(model_name)
    elif provider == "open_ai":
        return OpenAIEmbedder(model_name)
    else:
        print(f"No embedder implemented for {provider=}")
