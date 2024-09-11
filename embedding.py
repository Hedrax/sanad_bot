from sentence_transformers import SentenceTransformer

class embedding:
    def __init__(self):
        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, query):
        return self.model.encode(query).tolist()