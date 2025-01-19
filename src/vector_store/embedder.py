import os
from typing import List

import numpy as np
import torch

from abc import ABC, abstractmethod

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

from common.constants import EMBEDDING_MODEL_CACHE_DIR


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
        pretrained_model_dir = EMBEDDING_MODEL_CACHE_DIR / model_identifier
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

    def encode(self, text: List[str]) -> np.array:
        encoding = [self.embedder.feature_extraction(text_) for text_ in text]
        encoding = np.asarray(encoding)
        return encoding


class OpenAIEmbedder(Embedder):
    def __init__(self):
        self.embedder = ""

    def encode(self, text: str) -> np.array:
        return ""


def load_embedder(embedder_type: str, model_identifier: str) -> Embedder:
    if embedder_type == "pretrained_sentence_transformer":
        return SentenceTransformerEmbedder(model_identifier)
    elif embedder_type == "hf_api":
        return HFAPIEmbedder(model_identifier)
    else:
        print(f"No embedder implemented for {embedder_type=}")
