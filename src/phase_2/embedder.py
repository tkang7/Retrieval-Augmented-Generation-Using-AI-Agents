from sentence_transformers import SentenceTransformer
import json

class Embedder:
    def __init__(self):
        pass

    def create(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedder
    
    def get(self):
        return self.embedder
