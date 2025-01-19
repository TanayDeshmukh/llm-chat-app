from pathlib import Path
from typing import Dict

from streamlit.runtime.uploaded_file_manager import UploadedFile

from common.constants import DEFAULT_FILE_STORAGE, SUPPORTED_FILE_FORMATS


def save_file(
    uploaded_file: UploadedFile, meta_data: Dict, save_dir: Path = DEFAULT_FILE_STORAGE
) -> Path:
    assert (
        meta_data["file_type"] in SUPPORTED_FILE_FORMATS
    ), f"File type {meta_data['filetype']} is not in {SUPPORTED_FILE_FORMATS=}"
    save_dir.mkdir(exist_ok=True)
    file_path = save_dir / meta_data["file_name"]
    if file_path.exists():
        print("file already exists...")
    else:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return file_path
