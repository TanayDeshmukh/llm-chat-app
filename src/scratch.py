from vector_store.embedder import SentenceTransformerEmbedder, HFAPIEmbedder

embedder = HFAPIEmbedder(model_identifier="nvidia/NV-Embed-v2")

embedding = embedder.encode(["This is a test sentence"])

print(embedding.shape)

pass
