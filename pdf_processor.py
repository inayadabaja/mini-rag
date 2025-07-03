import PyPDF2
import re
from typing import List, Dict, Optional
import logging
from pathlib import Path
import fitz  # PyMuPDF - alternative plus robuste


class PDFProcessor:
    """Classe pour traiter et extraire le texte des fichiers PDF"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, use_pymupdf: bool = True):
        """
        Args:
            chunk_size: Taille des segments de texte (en mots)
            overlap: Chevauchement entre segments (en mots)
            use_pymupdf: Utiliser PyMuPDF au lieu de PyPDF2 (recommandé)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.use_pymupdf = use_pymupdf
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf_pypdf2(self, pdf_path: str) -> str:
        """Extrait le texte avec PyPDF2"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Vérifie que la page n'est pas vide
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text
                        else:
                            self.logger.warning(f"Page {page_num + 1} semble vide")
                    except Exception as e:
                        self.logger.warning(f"Erreur page {page_num + 1}: {e}")
                        continue
                
                return text
        
        except Exception as e:
            self.logger.error(f"Erreur lecture PDF avec PyPDF2: {e}")
            raise
    
    def extract_text_from_pdf_pymupdf(self, pdf_path: str) -> str:
        """Extrait le texte avec PyMuPDF (plus robuste)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page_text
                    else:
                        self.logger.warning(f"Page {page_num + 1} semble vide")
                        
                except Exception as e:
                    self.logger.warning(f"Erreur page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            return text
            
        except Exception as e:
            self.logger.error(f"Erreur lecture PDF avec PyMuPDF: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrait tout le texte d'un PDF en utilisant la méthode choisie"""
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")
        
        if self.use_pymupdf:
            try:
                return self.extract_text_from_pdf_pymupdf(pdf_path)
            except ImportError:
                self.logger.warning("PyMuPDF non disponible, utilisation de PyPDF2")
                return self.extract_text_from_pdf_pypdf2(pdf_path)
            except Exception as e:
                self.logger.warning(f"Échec PyMuPDF, tentative avec PyPDF2: {e}")
                return self.extract_text_from_pdf_pypdf2(pdf_path)
        else:
            return self.extract_text_from_pdf_pypdf2(pdf_path)
    
    def clean_text(self, text: str) -> str:
        """Nettoie le texte extrait"""
        if not text.strip():
            self.logger.warning("Texte vide après extraction")
            return ""
        
        # Remplace les sauts de ligne multiples par un seul
        text = re.sub(r'\n+', '\n', text)
        
        # Supprime les espaces multiples mais garde les sauts de ligne
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Supprime les caractères de contrôle mais garde les caractères utiles
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Supprime les lignes très courtes (moins de 3 caractères)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) >= 3:  # Garde les lignes avec au moins 3 caractères
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def create_chunks(self, text: str) -> List[Dict[str, any]]:
        """Découpe le texte en segments avec métadonnées"""
        if not text.strip():
            return []
        
        # Divise par mots en préservant la ponctuation
        words = re.findall(r'\S+', text)
        chunks = []
        
        if len(words) <= self.chunk_size:
            # Si le texte est plus court que la taille de chunk, créer un seul chunk
            chunk_info = {
                'text': text,
                'chunk_id': 0,
                'start_word': 0,
                'end_word': len(words),
                'word_count': len(words),
                'char_count': len(text)
            }
            chunks.append(chunk_info)
            return chunks
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Métadonnées du chunk
            chunk_info = {
                'text': chunk_text,
                'chunk_id': len(chunks),
                'start_word': i,
                'end_word': min(i + self.chunk_size, len(words)),
                'word_count': len(chunk_words),
                'char_count': len(chunk_text)
            }
            
            chunks.append(chunk_info)
            
            # Arrête si on a atteint la fin
            if i + self.chunk_size >= len(words):
                break
        
        return chunks
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, any]:
        """Récupère les informations sur le PDF"""
        try:
            if self.use_pymupdf:
                doc = fitz.open(pdf_path)
                info = {
                    'pages': len(doc),
                    'metadata': doc.metadata,
                    'encrypted': doc.is_encrypted
                }
                doc.close()
                return info
            else:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return {
                        'pages': len(pdf_reader.pages),
                        'metadata': pdf_reader.metadata,
                        'encrypted': pdf_reader.is_encrypted
                    }
        except Exception as e:
            self.logger.error(f"Erreur récupération info PDF: {e}")
            return {}
    
    def process_pdf(self, pdf_path: str) -> Dict[str, any]:
        """Pipeline complet de traitement du PDF"""
        self.logger.info(f"Traitement du PDF: {pdf_path}")
        
        # Informations sur le PDF
        pdf_info = self.get_pdf_info(pdf_path)
        self.logger.info(f"PDF info: {pdf_info.get('pages', 'inconnu')} pages")
        
        # Extraction du texte
        raw_text = self.extract_text_from_pdf(pdf_path)
        self.logger.info(f"Texte extrait: {len(raw_text)} caractères")
        
        if not raw_text.strip():
            self.logger.warning("Aucun texte extrait du PDF")
            return {
                'pdf_info': pdf_info,
                'raw_text': raw_text,
                'cleaned_text': "",
                'chunks': [],
                'success': False,
                'error': "Aucun texte extrait"
            }
        
        # Nettoyage
        cleaned_text = self.clean_text(raw_text)
        self.logger.info(f"Texte nettoyé: {len(cleaned_text)} caractères")
        
        # Création des chunks
        chunks = self.create_chunks(cleaned_text)
        self.logger.info(f"Créé {len(chunks)} segments")
        
        return {
            'pdf_info': pdf_info,
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'chunks': chunks,
            'success': True,
            'stats': {
                'raw_chars': len(raw_text),
                'cleaned_chars': len(cleaned_text),
                'num_chunks': len(chunks),
                'avg_chunk_size': sum(c['word_count'] for c in chunks) / len(chunks) if chunks else 0
            }
        }
