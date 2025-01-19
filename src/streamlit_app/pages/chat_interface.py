import time

import streamlit as st

from streamlit_app.utils import get_chat_completion_pipeline


class ChatInterface:
    def __init__(self):
        self.title = "Chat Interface"

    def initialize_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []

    def render_message(self, role: str, content: str):
        with st.chat_message(role):
            st.markdown(content)

    def render_all_messages(self):
        for message in st.session_state.messages:
            self.render_message(message["role"], message["content"])

    def stream_response(self, response: str):
        for word in response.split():
            yield word + " "
            time.sleep(0.05)

    def handle_user_input(self, pipeline):
        if prompt := st.chat_input("How can I help you?", key="chat_input"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            self.render_message(role="user", content=prompt)

            with st.spinner("Thinking..."):
                model_response, documents = pipeline.run_completion_pipeline(
                    chat_history=st.session_state.messages
                )

            with st.chat_message("assistant"):
                # if len(documents) > 0:
                #     for i, document in enumerate(documents):
                #         with st.expander(f"Document {i + 1}"):
                #             st.write(document.text)
                response = st.write_stream(self.stream_response(model_response))
            st.session_state.messages.append({"role": "assistant", "content": response})

    def render(self):
        st.title(self.title)

        self.initialize_state()

        pipeline = get_chat_completion_pipeline()

        self.render_all_messages()

        self.handle_user_input(pipeline=pipeline)

        if "use_rag" not in st.session_state:
            st.session_state.use_rag = False
        if "use_web_search" not in st.session_state:
            st.session_state.use_web_search = False

        with st.sidebar:
            st.session_state.use_rag = st.checkbox("Use RAG")
            st.session_state.use_web_search = st.checkbox("Use web search")


if __name__ == "__main__":
    ChatInterface().render()
