import streamlit as st


def run_app():
    st.set_page_config(
        page_title="LLM-CHAT-APP",
        page_icon="💬",
        layout="wide",
    )

    st.markdown(
        """
        Welcome to LLM-CHAT-APP

        **Goals of the project:**
        - **Chat Interface**: To have a chat interface that lets the users interact with an LLM.
        - **RAG**: Learning how to use RAG, with the help of OpenSearch.
        - **Pipelines**: Building various pipelines to handle file upload, RAG and text generation.

        **Choose a page from the sidebar to begin!**
        """
    )


if __name__ == "__main__":
    run_app()
