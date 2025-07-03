import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle
import os
import logging

class VectorStore:
    """Classe pour gérer les embeddings et la recherche vectorielle"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Args:
            model_name: Nom du modèle d'embedding Hugging Face
        """
        self.model_name = model_name
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self.dimension = None
        self.logger = logging.getLogger(__name__)
    
    def load_embedding_model(self):
        """Charge le modèle d'embedding"""
        if self.embedding_model is None:
            self.logger.info(f"Chargement du modèle: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            self.logger.info(f"Dimension des embeddings: {self.dimension}")
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Crée les embeddings pour une liste de textes"""
        if self.embedding_model is None:
            self.load_embedding_model()
        
        self.logger.info(f"Création embeddings pour {len(texts)} textes")
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return embeddings
    
    def build_index(self, chunks: List[Dict[str, any]]):
        """Construit l'index FAISS à partir des chunks"""
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Création des embeddings
        embeddings = self.create_embeddings(texts)
        
        # Construction de l'index FAISS
        self.logger.info("Construction de l'index FAISS")
        self.index = faiss.IndexFlatIP(self.dimension)  # Produit scalaire
        
        # Normalisation pour utiliser la similarité cosinus
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.logger.info(f"Index construit avec {self.index.ntotal} vecteurs")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Dict[str, any], float]]:
        """Recherche les chunks les plus similaires à la requête"""
        if self.index is None:
            raise ValueError("Index non construit. Appelez build_index() d'abord.")
        
        if self.embedding_model is None:
            self.load_embedding_model()
        
        # Embedding de la requête
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Recherche dans l'index
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Formatage des résultats
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Index valide
                chunk = self.chunks[idx].copy()
                results.append((chunk, float(score)))
        
        return results
    
    def save_index(self, filepath: str):
        """Sauvegarde l'index et les métadonnées"""
        if self.index is None:
            raise ValueError("Aucun index à sauvegarder")
        
        # Sauvegarde de l'index FAISS
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Sauvegarde des métadonnées
        metadata = {
            'chunks': self.chunks,
            'model_name': self.model_name,
            'dimension': self.dimension
        }
        
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"Index sauvegardé: {filepath}")
    
    def load_index(self, filepath: str):
        """Charge un index sauvegardé"""
        # Chargement de l'index FAISS
        if os.path.exists(f"{filepath}.faiss"):
            self.index = faiss.read_index(f"{filepath}.faiss")
        else:
            raise FileNotFoundError(f"Index non trouvé: {filepath}.faiss")
        
        # Chargement des métadonnées
        if os.path.exists(f"{filepath}.pkl"):
            with open(f"{filepath}.pkl", 'rb') as f:
                metadata = pickle.load(f)
                self.chunks = metadata['chunks']
                self.model_name = metadata['model_name']
                self.dimension = metadata['dimension']
        else:
            raise FileNotFoundError(f"Métadonnées non trouvées: {filepath}.pkl")
        
        self.logger.info(f"Index chargé: {filepath}")
    
    def get_stats(self) -> Dict[str, any]:
        """Retourne des statistiques sur l'index"""
        if self.index is None:
            return {"status": "Index non construit"}
        
        return {
            "nb_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "nb_chunks": len(self.chunks),
            "model": self.model_name
        }
