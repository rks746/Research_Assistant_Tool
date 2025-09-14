from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Searcher:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.texts = []
        self.ids = []
        
    def add_document(self, doc_id, text):
        # Split text into smaller parts if needed
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        embeddings = self.model.encode(parts)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))
        
        self.texts.extend(parts)
        self.ids.extend([doc_id]*len(parts))

        print(f"Adding {len(parts)} parts from document {doc_id}")
        print("Embedding shape:", embeddings.shape)

        
    def search(self, query, top_k=5):
        if self.index is None or len(self.texts) == 0:
            return []
        
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vec), top_k)
        print("Query:", query)
        print("Query vector shape:", query_vec.shape)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.texts):
                results.append(self.texts[idx])
        return results
