from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import List, Dict, Tuple
import logging
import torch
from pdf_processor import PDFProcessor
from vector_store import VectorStore

class RAGSystem:
    """Système RAG complet pour chat avec PDFs"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 generation_model: str = "google/flan-t5-large",
                 device: str = None):
        """
        Args:
            embedding_model: Modèle pour les embeddings
            generation_model: Modèle pour la génération
            device: Device ('cpu', 'cuda', ou None pour auto)
        """
        self.embedding_model_name = embedding_model
        self.generation_model_name = generation_model
        
        # Détection automatique du device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Utilisation du device: {self.device}")
        
        # Composants
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore(embedding_model)
        self.generator = None
        
        # État
        self.is_ready = False
        self.current_pdf = None
    
    def load_generation_model(self):
        """Charge le modèle de génération"""
        if self.generator is None:
            self.logger.info(f"Chargement du modèle: {self.generation_model_name}")
            
            try:
                # Utilisation de pipeline pour simplicité
                self.generator = pipeline(
                    "text2text-generation",
                    model=self.generation_model_name,
                    device=0 if self.device == "cuda" else -1,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7
                )
                self.logger.info("Modèle de génération chargé avec succès")
                
            except Exception as e:
                self.logger.error(f"Erreur chargement modèle: {e}")
                raise
    
    def load_pdf(self, pdf_path: str) -> Dict[str, any]:
        """Charge et traite un PDF"""
        try:
            self.logger.info(f"Chargement du PDF: {pdf_path}")
            
            # Traitement du PDF
            result = self.pdf_processor.process_pdf(pdf_path)
            chunks = result.get('chunks', [])
            
            # Construction de l'index vectoriel
            self.vector_store.build_index(chunks)
            
            # Mise à jour de l'état
            self.current_pdf = pdf_path
            self.is_ready = True
            
            stats = {
                "pdf_path": pdf_path,
                "nb_chunks": len(chunks),
                "status": "Prêt pour les questions"
            }
            
            self.logger.info(f"PDF chargé: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Erreur chargement PDF: {e}")
            self.is_ready = False
            raise
    
    def create_prompt(self, question: str, context_chunks: List[Dict[str, any]]) -> str:
        """Crée le prompt pour le modèle génératif"""
        # Assemblage du contexte
        context_texts = [chunk['text'] for chunk in context_chunks]
        context = "\n\n".join(context_texts)
        
        # Template de prompt optimisé pour FLAN-T5
        prompt = f"""Contexte: {context}

Question: {question}

Réponds à la question en te basant uniquement sur le contexte fourni. Si la réponse n'est pas dans le contexte, dis "Je ne trouve pas cette information dans le document."

Réponse:"""
        
        return prompt
    
    def generate_answer(self, question: str, max_context_chunks: int = 3) -> Dict[str, any]:
        """Génère une réponse à partir de la question"""
        if not self.is_ready:
            return {
                "answer": "Aucun PDF chargé. Veuillez d'abord charger un document.",
                "sources": [],
                "error": "no_pdf_loaded"
            }
        
        try:
            # Chargement du modèle si nécessaire
            if self.generator is None:
                self.load_generation_model()
            
            # Recherche vectorielle
            search_results = self.vector_store.search(question, k=max_context_chunks)
            
            if not search_results:
                return {
                    "answer": "Aucun contenu pertinent trouvé dans le document.",
                    "sources": [],
                    "error": "no_relevant_content"
                }
            
            # Extraction des chunks et scores
            context_chunks = [result[0] for result in search_results]
            scores = [result[1] for result in search_results]
            
            # Création du prompt
            prompt = self.create_prompt(question, context_chunks)
            
            # Génération de la réponse
            self.logger.info("Génération de la réponse...")
            generated = self.generator(prompt, max_length=256, num_return_sequences=1)
            
            answer = generated[0]['generated_text'].strip()
            
            # Formatage des sources
            sources = []
            for i, (chunk, score) in enumerate(zip(context_chunks, scores)):
                sources.append({
                    "chunk_id": chunk.get('chunk_id', i),
                    "text": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                    "score": round(score, 3)
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Erreur génération réponse: {e}")
            return {
                "answer": f"Erreur lors de la génération: {str(e)}",
                "sources": [],
                "error": "generation_error"
            }
    
    def get_system_info(self) -> Dict[str, any]:
        """Retourne des informations sur le système"""
        info = {
            "embedding_model": self.embedding_model_name,
            "generation_model": self.generation_model_name,
            "device": self.device,
            "is_ready": self.is_ready,
            "current_pdf": self.current_pdf
        }
        
        if self.is_ready:
            info.update(self.vector_store.get_stats())
        
        return info

# Exemple d'utilisation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    rag = RAGSystem()
    pdf_path = r"C:/Users/Inaya/Desktop/rag-pdf-chat/demo/ref.pdf"
    
    try:
        # Charger le PDF en passant la variable pdf_path
        stats = rag.load_pdf(pdf_path)
        print(f"PDF chargé: {stats}")
        
        # Tester une question
        result = rag.generate_answer("De quoi parle ce document ?")
        print(f"Réponse: {result['answer']}")
        print(f"Nombre de sources: {len(result['sources'])}")
        
        print("Système RAG initialisé avec succès!")
        print(f"Infos système: {rag.get_system_info()}")
        
    except Exception as e:
        print(f"Erreur: {e}")