from typing import Dict

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from data.data_classes import UploadedFileMetadata
from data.utils import save_file
from data_store.uploaded_files import UploadedFilesDB
from vector_store.document_vector_store import VectorDB
from vector_store.utils import extract_text_pages_from_pdf, extract_document_chunks


def pdf_upload_pipeline(
    uploaded_file: UploadedFile,
    meta_data: Dict,
    vector_db: VectorDB,
    db: UploadedFilesDB,
):
    if db.file_exists(uploaded_file.name):
        st.warning(f"{uploaded_file.name=} already exists.")
    else:
        file_path = save_file(uploaded_file, meta_data)
        st.write(f"File saved to {file_path=}")
        pages = extract_text_pages_from_pdf(file_path)
        st.write("Chunking document..")
        documents = extract_document_chunks(
            pages,
            meta_data["file_name"],
            chunk_size=st.session_state.config.opensearch_config.chunk_size,
            overlap_ratio=st.session_state.config.opensearch_config.overlap_ratio,
        )
        st.write("Adding embeddings to vector database..")
        error = vector_db.add_documents(documents)

        uploaded_file_meta_data = UploadedFileMetadata(
            file_name=meta_data["file_name"],
            file_type=meta_data["file_type"],
            size_on_disk=meta_data["file_size"],
            num_pages=len(pages),
            num_chunks=len(documents),
            num_chunks_error_upload=error,
        )

        st.write("Writing metadata to database..")
        db.insert_file(uploaded_file_meta_data)


def run_document_upload_pipeline(
    uploaded_file: UploadedFile, vector_db: VectorDB, db: UploadedFilesDB
):
    meta_data = {
        "file_name": uploaded_file.name,
        "file_type": uploaded_file.type.split("/")[-1],
        "file_size": uploaded_file.size,
    }

    if uploaded_file.type == "application/pdf":
        pdf_upload_pipeline(uploaded_file, meta_data, vector_db, db)
    else:
        st.warning(f"{uploaded_file.type=} is not supported.")
