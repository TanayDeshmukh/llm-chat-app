# Build an LLM chat app from scratch

The goal of this project was to explore and implement a Retrieval Augmented Generation (RAG) 
powered LLM chatbot from scratch. Although frameworks such as Haystack and Langchain provide a
modular, abstracted way to build such applications, I wanted to take a more hands-on approach.
Implementing the core functionalities myself gave me a deeper understanding of how
each of the components work and interacts with other components to form a robust system.

This project is as much about learning as it is about coding. I focused in exploring foundational
concepts such as, document processing, document chunking, basic RAG workflows, document reranking,
constructing pipelines, while also emphasizing clean code and good coding practices.

Instead of prioritizing the creation of a chatbot with exceptional performance, I focused on understanding 
how all the individual pieces fit together in a well-organized Python project while keeping scalability in mind.

---

## Key features

- Learn how to build **RAG powered LLM pipelines**.
- **OpenSearch** used as the vector database store and is deployed in a Docker container.
-  Uses **Huggingface InferenceClient** to generate LLM response.
- **Modular code** provides the possibility to easily incorporate different embedders, rerankers and generators. 
- App functionality can be controlled based on simple **config files**

---

## Getting started

### Prerequisites 
1. Docker is required to run the OpenSearch container. Details for the installation can be found [here] (https://opensearch.org/docs/latest/install-and-configure/install-opensearch/docker/).
2. Huggingface API token to use the InferenceClient. Details can be found [here] (https://huggingface.co/docs/hub/security-tokens).
3. OpenAI API token (Required only if you want to use the OpenAI models).

### Setup

1. Clone the repository [llm-chat-app](https://github.com/TanayDeshmukh/llm-chat-app.git).
2. Install dependencies. ``conda env create -f environment.yml``.
3. Configure OpenSearch:
   - Modify the volumes path in ``docker-compose-opensearch.yml``. Update the local path ``D:/Projects/llm-chat-app-data/opensearch-data`` to a directory on your system where you want the OpenSearch data to be stored.
   - Set ``OPENSEARCH_INITIAL_ADMIN_PASSWORD`` environment variable. 
4. Add the huggingface API token with the name ``LLM_CHAT_APP_HF_API_TOKEN`` in your environment variables.
5. Mark ``src`` as ``Sources Root``.
6. You can set all the required configurations in a ``config.yml`` in the directory `src/common/configs`.
7. Run streamlit app with ``streamlit run app.py -- --config <config file name>``. Example: ``streamlit run app.py -- --config 001_base_config.yml``.


---

## Coming soon

- Blog posts explaining the project in detail.
- Web search in addition to simple RAG.
- Retrieved document and web search citation cards.

---

If you found this project interesting, please share it with the community, and feel free
to reach out if you have any suggestions or questions. I would love to hear from you. Keep learning!