# mini-rag

## Présentation

**mini-rag** est une application simple de RAG (Retrieval-Augmented Generation) permettant de discuter avec le contenu de vos fichiers PDF grâce à l'intelligence artificielle. Elle combine l'extraction de texte, la vectorisation, la recherche sémantique et la génération de réponses pour répondre à vos questions sur un document PDF.

## Fonctionnalités

- 📁 **Upload de PDF** : Chargez un fichier PDF et segmentez automatiquement son contenu.
- 🔎 **Recherche sémantique** : Les passages pertinents sont retrouvés grâce à des embeddings et la recherche vectorielle (FAISS).
- 🤖 **Génération de réponses** : Un modèle de génération (Flan-T5) synthétise une réponse à partir des passages retrouvés.
- 💬 **Interface conviviale** : Utilisation de Gradio pour une expérience utilisateur simple et interactive.

## Dépendances principales

- Python 3.10+
- gradio
- transformers
- sentence-transformers
- faiss-cpu
- PyPDF2, PyMuPDF (fitz)

Voir `requierements.txt` pour la liste complète.

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone <url-du-repo>
   cd rag-pdf-chat
   ```
2. Installez les dépendances :
   ```bash
   pip install -r requierements.txt
   ```

## Utilisation

Lancez l'application avec :
```bash
python app.py
```
Puis ouvrez l'URL Gradio affichée dans votre navigateur.

## Exemple d'utilisation

1. Uploadez un PDF via l'interface.
2. Attendez le traitement (quelques secondes).
3. Posez vos questions sur le contenu du document.

Exemples de questions :
- "De quoi parle ce document ?"
- "Quels sont les points principaux abordés ?"
- "Que dit le document à propos de [sujet spécifique] ?"

## Architecture

- `app.py` : Interface Gradio et logique principale.
- `rag_system.py` : Orchestration du pipeline RAG (extraction, vectorisation, génération).
- `pdf_processor.py` : Extraction et segmentation du texte PDF.
- `vector_store.py` : Gestion des embeddings et de la recherche vectorielle.

## Remarques

- Les modèles utilisés sont téléchargeables automatiquement via Hugging Face.
- L'application fonctionne sur CPU ou GPU (si disponible).
- Pour des documents volumineux, le traitement peut prendre plus de temps.

## Démo

Une vidéo de démonstration est disponible dans `demo_mini_rag.mp4`.