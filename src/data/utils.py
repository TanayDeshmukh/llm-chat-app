from pathlib import Path
from typing import Dict

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def save_file(uploaded_file: UploadedFile, meta_data: Dict) -> Path:
    assert (
        meta_data["file_type"] in st.session_state.config.supported_file_formats
    ), f"File type {meta_data['filetype']} is not in {st.session_state.config.supported_file_formats=}"
    save_dir = st.session_state.config.file_storage_dir
    save_dir.mkdir(exist_ok=True)
    file_path = save_dir / meta_data["file_name"]
    if file_path.exists():
        print("file already exists...")
    else:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return file_path
