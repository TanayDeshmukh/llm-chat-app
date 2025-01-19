from pathlib import Path


# Default storage directory
DEFAULT_DATA_DIR = Path(r"D:/Projects/llm-chat-app-data")

DEFAULT_FILE_STORAGE_DIR = DEFAULT_DATA_DIR / "file_storage"

# Embeddings model constants
DEFAULT_EMBEDDING_MODEL_CACHE_DIR = DEFAULT_DATA_DIR / "embedding_model_cache_dir"

# Generator model constants
DEFAULT_GENERATOR_MODEL_CACHE_DIR = DEFAULT_DATA_DIR / "generator_model_cache_dir"

# Database constants
DEFAULT_DB_DIR = DEFAULT_DATA_DIR / "database"
