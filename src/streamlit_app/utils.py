import streamlit as st

from data_store.uploaded_files import UploadedFilesDB
from pipelines.chat_completion import ChatCompletionPipeline
from vector_store.document_vector_store import VectorDB


@st.cache_resource
def get_db() -> UploadedFilesDB:
    db = UploadedFilesDB()
    return db


@st.cache_resource
def get_vector_db() -> VectorDB:
    vector_db = VectorDB()
    return vector_db


def get_chat_completion_pipeline() -> ChatCompletionPipeline:
    pipeline = ChatCompletionPipeline(vector_store=get_vector_db())
    return pipeline
