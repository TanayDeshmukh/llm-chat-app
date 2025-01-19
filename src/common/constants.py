from pathlib import Path


# Default storage directory
DEFAULT_DATA_DIR = Path(r"D:/Projects/llm-chat-app-data")

# File storage constants
DEFAULT_FILE_STORAGE = DEFAULT_DATA_DIR / "file_storage"

# Embeddings model constants
EMBEDDING_MODEL_CACHE_DIR = DEFAULT_DATA_DIR / "embedding_model_cache_dir"
DEFAULT_EMBEDDER_TYPE = "pretrained_sentence_transformer"
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"

# Generator model constants
DEFAULT_GENERATOR_MODEL_IDENTIFIER = "meta-llama/Meta-Llama-3-8B-Instruct"

# OpenSearch constants
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
DEFAULT_INDEX = "test_index"
DEFAULT_TOP_K_EMBEDDINGS = 10
DEFAULT_EMBEDDING_DIMENSION = 768

# Database constants
DEFAULT_DB_DIR = DEFAULT_DATA_DIR / "database"
DEFAULT_DB_NAME = "uploaded_files.db"


# 384 tokens is the max number of tokens for 'all-mpnet-base-v2' embedding model
# Generally, 4 characters equals 1 token, hence the chunk size is set to  384x4
# The overlap is set to 25% of the chunk size. These parameters may be adjusted depending on the embeddings model used.
CHUNK_SIZE = 384 * 4
OVERLAP_SIZE = 384

SUPPORTED_FILE_FORMATS = ["pdf"]


DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant that provides accurate and helpful responses. "
    "Keep your responses as short as possible, unless explicitly asked to generate longer answers"
)
