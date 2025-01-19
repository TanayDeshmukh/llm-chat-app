from pathlib import Path
from typing import Literal, List, Optional, Any

from pydantic import BaseModel

from common.constants import (
    DEFAULT_DATA_DIR,
    DEFAULT_DB_DIR,
    DEFAULT_EMBEDDING_MODEL_CACHE_DIR,
    DEFAULT_GENERATOR_MODEL_CACHE_DIR,
    DEFAULT_FILE_STORAGE_DIR,
)


class EmbeddingModelConfig(BaseModel):
    cache_dir: Path = DEFAULT_EMBEDDING_MODEL_CACHE_DIR
    model_provider: Literal["sentence_transformer", "hf_api", "open_ai"]
    model_name: str
    embedding_dimension: int
    trust_remote_code: bool = False  # some hf models need this to be set True


class GeneratorModelConfig(BaseModel):
    cache_dir: Path = DEFAULT_GENERATOR_MODEL_CACHE_DIR
    model_provider: Literal["hf_api"]
    model_name: str
    max_output_tokens: int
    temperature: float = 0.0


class ReRankerConfig(BaseModel):
    model_provider: Optional[str] = "default"
    top_k: Optional[int] = None


class OpenSearchConfig(BaseModel):
    host: str = "localhost"
    port: int = 9200
    index_name: str
    use_top_k_embeddings: int
    embedding_dimension: int
    chunk_size: int
    overlap_ratio: float
    filter_confidence_threshold: float = 0.5


class DataBaseConfig(BaseModel):
    root_dir: Path = DEFAULT_DB_DIR
    db_name: str


class LLMChatConfig(BaseModel):
    root_storage_dir: Path = DEFAULT_DATA_DIR
    file_storage_dir: Optional[Path] = None

    embedding_model_config: EmbeddingModelConfig
    generator_model_config: GeneratorModelConfig
    opensearch_config: OpenSearchConfig
    database_config: DataBaseConfig
    reranker_config: ReRankerConfig

    use_rag: Optional[bool] = False
    use_web_search: Optional[bool] = False

    supported_file_formats: List[str]
    system_prompt: str = (
        "You are an AI assistant that provides accurate and helpful responses. "
        "Keep your responses brief, unless explicitly asked to generate longer answers"
    )

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        if self.file_storage_dir is None:
            self.file_storage_dir = DEFAULT_FILE_STORAGE_DIR

    def __post_init_post_parse__(self):
        assert (
            self.embedding_model_config.embedding_dimension
            == self.opensearch_config.embedding_dimension
        ), f"{self.embedding_model_config.embedding_dimension=} does not match {self.opensearch_config.embedding_dimension=}"
