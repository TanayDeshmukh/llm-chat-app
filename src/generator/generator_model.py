import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

import streamlit as st
from huggingface_hub import InferenceClient


class GeneratorModel(ABC):
    @abstractmethod
    def generate_response(
        self, chat_history: List[Dict[str, str]], generation_args: Dict[str, Any] = None
    ):
        pass


class HFAPIGeneratorModel(GeneratorModel):
    def __init__(self, hf_identifier: str):
        self.hf_identifier = hf_identifier
        hf_api_token = os.getenv("LLM_CHAT_APP_HF_API_TOKEN")
        self.client = InferenceClient(model=self.hf_identifier, token=hf_api_token)

    @staticmethod
    def _get_default_generation_args():
        generation_args = {
            "max_tokens": st.session_state.config.generator_model_config.max_output_tokens,
            "temperature": st.session_state.config.generator_model_config.temperature,
        }
        return generation_args

    def generate_response(
        self, chat_history: List[Dict[str, str]], generation_args: Dict[str, Any] = None
    ):
        if generation_args is None:
            generation_args = self._get_default_generation_args()

        model_output = self.client.chat_completion(chat_history, **generation_args)

        response = model_output.choices[0].message.content

        return response


def load_generator(provider: str, model_name: str) -> GeneratorModel:
    if provider == "hf_api":
        return HFAPIGeneratorModel(model_name)
    else:
        print(f"No generator implemented for {provider=}")
