import gradio as gr
import os
import tempfile
from rag_system import RAGSystem
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation du système RAG global
rag_system = RAGSystem()

def upload_pdf(file):
    """Traite l'upload d'un PDF"""
    if file is None:
        return "❌ Aucun fichier sélectionné", ""
    
    try:
        # Le fichier est déjà sauvé temporairement par Gradio
        file_path = file.name
        
        # Chargement dans le système RAG
        stats = rag_system.load_pdf(file_path)
        
        status_message = f"✅ PDF chargé avec succès!\n"
        status_message += f"📄 Fichier: {os.path.basename(file_path)}\n"
        status_message += f"📝 Segments: {stats['nb_chunks']}\n"
        status_message += f"🎯 Statut: {stats['status']}"
        
        welcome_message = "Bonjour ! Votre document a été chargé. Vous pouvez maintenant me poser des questions à son sujet."
        
        return status_message, welcome_message
        
    except Exception as e:
        error_message = f"❌ Erreur lors du chargement: {str(e)}"
        logger.error(f"Erreur upload PDF: {e}")
        return error_message, ""

def chat_with_pdf(message, history):
    """Gère la conversation avec le PDF"""
    if not rag_system.is_ready:
        bot_message = "⚠️ Veuillez d'abord charger un PDF avant de poser des questions."
        history.append([message, bot_message])
        return history, ""
    
    if not message.strip():
        bot_message = "Veuillez poser une question."
        history.append([message, bot_message])
        return history, ""
    
    try:
        # Génération de la réponse
        result = rag_system.generate_answer(message)
        
        if result.get('error'):
            bot_message = f"❌ {result['answer']}"
        else:
            bot_message = result['answer']
            
            # Ajout des sources si disponibles
            if result.get('sources'):
                bot_message += "\n\n📚 **Sources utilisées:**\n"
                for i, source in enumerate(result['sources'][:2]):  # Limite à 2 sources
                    bot_message += f"{i+1}. {source['text'][:150]}... (Score: {source['score']})\n"
        
        history.append([message, bot_message])
        return history, ""
        
    except Exception as e:
        bot_message = f"❌ Erreur: {str(e)}"
        history.append([message, bot_message])
        logger.error(f"Erreur chat: {e}")
        return history, ""

def clear_chat():
    """Efface l'historique du chat"""
    return []

def get_system_status():
    """Retourne le statut du système"""
    info = rag_system.get_system_info()
    
    status = f"🤖 **Statut du système RAG**\n\n"
    status += f"📊 Modèle embeddings: `{info['embedding_model']}`\n"
    status += f"🧠 Modèle génératif: `{info['generation_model']}`\n"
    status += f"💻 Device: `{info['device']}`\n"
    status += f"📄 PDF chargé: `{'Oui' if info['is_ready'] else 'Non'}`\n"
    
    if info['is_ready']:
        status += f"📝 Nombre de segments: `{info.get('nb_chunks', 'N/A')}`\n"
        status += f"🎯 Prêt pour les questions: ✅"
    else:
        status += f"🎯 Prêt: ❌ (Chargez un PDF)"
    
    return status

# Création de l'interface Gradio
def create_interface():
    """Crée l'interface Gradio"""
    
    with gr.Blocks(
        title="RAG PDF Chat",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-container {
            height: 500px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🤖 RAG PDF Chat
        ## Chattez avec vos documents PDF grâce à l'IA
        
        **Comment utiliser:**
        1. 📁 Uploadez votre fichier PDF
        2. ⏳ Attendez le traitement (quelques secondes)
        3. 💬 Posez vos questions sur le contenu du document
        """)
        
        with gr.Row():
            # Colonne de gauche - Upload et infos
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Chargement du document")
                
                pdf_upload = gr.File(
                    label="Sélectionnez votre PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                upload_status = gr.Textbox(
                    label="Statut du chargement",
                    interactive=False,
                    lines=4
                )
                
                gr.Markdown("### ⚙️ Informations système")
                
                system_status = gr.Textbox(
                    label="Statut du système",
                    value=get_system_status(),
                    interactive=False,
                    lines=8
                )
                
                refresh_btn = gr.Button("🔄 Actualiser statut", variant="secondary")
            
            # Colonne de droite - Chat
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Conversation")
                
                chatbot = gr.Chatbot(
                    label="Conversation avec votre PDF",
                    height=400,
                    show_label=True,
                    container=True,
                    elem_classes=["chat-container"]
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Votre question",
                        placeholder="Posez une question sur le document...",
                        lines=2,
                        scale=4
                    )
                    
                    with gr.Column(scale=1):
                        send_btn = gr.Button("📤 Envoyer", variant="primary")
                        clear_btn = gr.Button("🗑️ Effacer", variant="secondary")
                
                # Zone de message de bienvenue
                welcome_msg = gr.Textbox(
                    label="Messages",
                    interactive=False,
                    visible=False
                )
        
        # Exemples de questions
        gr.Markdown("""
        ### 💡 Exemples de questions que vous pouvez poser:
        - "De quoi parle ce document ?"
        - "Quels sont les points principaux abordés ?"
        - "Résume-moi le contenu en quelques phrases"
        - "Que dit le document à propos de [sujet spécifique] ?"
        """)
        
        # Event handlers
        pdf_upload.change(
            fn=upload_pdf,
            inputs=[pdf_upload],
            outputs=[upload_status, welcome_msg]
        )
        
        # Ajout automatique du message de bienvenue
        welcome_msg.change(
            fn=lambda msg, hist: hist + [["", msg]] if msg else hist,
            inputs=[welcome_msg, chatbot],
            outputs=[chatbot]
        )
        
        send_btn.click(
            fn=chat_with_pdf,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            fn=chat_with_pdf,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot]
        )
        
        refresh_btn.click(
            fn=get_system_status,
            outputs=[system_status]
        )
    
    return demo

if __name__ == "__main__":
    # Création et lancement de l'interface
    demo = create_interface()
    
    # Lancement avec options de développement
    demo.launch(
        share=False,  # Mettre True pour partager publiquement
        debug=True,   # Mode debug pour développement
        server_name="0.0.0.0",  # Accessible depuis le réseau local
        server_port=7860,  # Port par défaut
        show_error=True
    )