import streamlit as st

from pipelines.document_upload import run_document_upload_pipeline
from streamlit_app.utils import get_db, get_vector_db


class UploadDocumentsInterface:
    def __init__(self):
        self.title = "Upload files"

    def render(self):
        st.title(self.title)

        uploaded_file = st.file_uploader(
            "Choose file to upload",
            type=st.session_state.config.supported_file_formats,
            accept_multiple_files=False,
        )

        db = get_db()
        vector_db = get_vector_db()
        existing_files = db.read_all_files_as_df()
        dataframe_placeholder = st.dataframe(
            existing_files,
        )

        if uploaded_file is not None:
            with st.status("Processing uploaded document..."):
                run_document_upload_pipeline(uploaded_file, vector_db=vector_db, db=db)
            updated_files = db.read_all_files_as_df()
            dataframe_placeholder.dataframe(updated_files)


if __name__ == "__main__":
    UploadDocumentsInterface().render()
