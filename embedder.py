from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model {self.model_name} loaded successfully!")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise

    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        print(f"Generating embeddings for {len(texts)} text(s)...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
