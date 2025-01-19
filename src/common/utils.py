import os
from pathlib import Path

import yaml

from common.config import LLMChatConfig


def load_config(config_name: str) -> LLMChatConfig:
    config_path = Path(os.getcwd()) / Path("common/configs") / config_name
    with open(config_path, "r") as f:
        config_yml = yaml.safe_load(f)
        config = LLMChatConfig(**config_yml)
        return config
