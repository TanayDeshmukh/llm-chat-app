import argparse
import sys

import streamlit as st

from common.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="001_base_config.yml",
        help="provide the name of the config yaml file (to be placed on the path 'src/common/configs')",
    )
    args, unknown = parser.parse_known_args(sys.argv[1:])
    return args


def run_app(config: str):
    st.set_page_config(
        page_title="LLM-CHAT-APP",
        page_icon="💬",
        layout="wide",
    )

    if "config" not in st.session_state:
        llm_chat_config = load_config(config_name=config)
        st.session_state.config = llm_chat_config

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

    args = parse_args()

    run_app(config=args.config)
