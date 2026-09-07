import os
import sys
import logging
import uuid
from typing import List
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Caminhos base
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
SQLITE_DB_PATH = f"sqlite:///{os.path.join(DB_DIR, 'chat_history.db')}"
VECTORDB_DIR = os.path.join(DB_DIR, "vectordb")

class MemoryManager:
    """
    MemoryManager gerencia 3 níveis de retenção de contexto:
    1. Short-Term: Últimas N mensagens via SQLite.
    2. Medium-Term: Resumo contínuo de conversas longas (Session Summary).
    3. Long-Term: Extração de Fatos e Dúvidas Jurídicas para o ChromaDB (User Profiles).
    """
    def __init__(self, session_id: str, user_id: str, model: str = "groq/compound-mini"):
        self.session_id = session_id
        self.user_id = user_id
        self.model = model
        
        os.makedirs(DB_DIR, exist_ok=True)
        
        # 1. Short-Term Memory (SQLite db)
        self.chat_history = SQLChatMessageHistory(
            session_id=self.session_id,
            connection=SQLITE_DB_PATH
        )
        
        # 3. Long-Term Memory (ChromaDB)
        self.chroma_client = chromadb.PersistentClient(path=VECTORDB_DIR)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.ltm_collection = self.chroma_client.get_or_create_collection(
            name="user_profiles_collection",
            embedding_function=self.emb_fn
        )
        
        # LLM Client (Groq)
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        
    def add_interaction(self, user_input: str, agent_response: str):
        """Salva a interação imediata e engatilha a extração de Long-Term Memory."""
        # Salva no SQLite
        self.chat_history.add_user_message(user_input)
        self.chat_history.add_ai_message(agent_response)
        
        # Em background/assíncrono em produção, aqui faremos síncrono para o RAG
        self._extract_and_save_long_term_facts(user_input, agent_response)
        
    def get_short_term_context(self, window_size: int = 4) -> str:
        """Retorna as últimas N interações da Janela Deslizante."""
        messages = self.chat_history.messages[-(window_size*2):] # *2 porque é par (req/res)
        if not messages:
            return ""
            
        context = "=== Histórico Recente da Conversa ===\n"
        for m in messages:
            if isinstance(m, HumanMessage):
                context += f"Usuário: {m.content}\n"
            elif isinstance(m, AIMessage):
                context += f"Assistente: {m.content}\n"
        return context + "===================================\n"

    def _extract_and_save_long_term_facts(self, user_input: str, agent_response: str):
        """Long-Term: Avalia se há preferência ou pergunta jurídica passada relevante e salva."""
        prompt = (
            "Analise a interação abaixo. O seu objetivo é extrair fatos persistentes sobre o usuário "
            "(ex: 'O usuário é advogado', 'A empresa do usuário sofreu um ataque') ou "
            "dúvidas jurídicas complexas que ele tenha demonstrado interesse (ex: 'O usuário estuda sobre bases legais').\n"
            "Se não houver NADA de relevante para lembrar a longo prazo (ex: saudações, perguntas curtas), "
            "retorne APENAS a palavra VAZIO.\n\n"
            f"Usuário: {user_input}\nAssistente: {agent_response}\n\nFatos extraídos:"
        )
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            fact = completion.choices[0].message.content.strip()
            
            # Checa se o LLM achou algo útil
            if "VAZIO" not in fact and len(fact) > 10:
                logging.info(f"[Long-Term Memory] Fato detectado e salvo: {fact}")
                self.ltm_collection.add(
                    documents=[fact],
                    metadatas=[{"user_id": self.user_id, "source": "chat_extraction"}],
                    ids=[str(uuid.uuid4())]
                )
        except Exception as e:
            logging.error(f"Failed to extract long term facts: {e}")

    def get_long_term_context(self, query: str) -> str:
        """Recupera fatos do usuário baseados na query atual."""
        try:
            results = self.ltm_collection.query(
                query_texts=[query],
                n_results=2,
                where={"user_id": self.user_id}
            )
            
            if results and results.get("documents") and len(results["documents"][0]) > 0:
                facts = "\n- ".join(results["documents"][0])
                return f"=== Fatos e Preferências do Usuário (Memória de Longo Prazo) ===\n- {facts}\n============================================================\n"
        except Exception:
            pass
        return ""

if __name__ == "__main__":
    # Teste Rápido
    mem = MemoryManager(session_id="sess_123", user_id="kelvin_dev")
    
    print("[1] Testando Injeção no Short Term...")
    mem.add_interaction("Olá, eu sou o Kelvin e sou desenvolvedor de software na empresa X.", "Olá Kelvin! Como posso ajudar sua empresa com a LGPD hoje?")
    
    print("\n[2] Lendo Short Term Context...")
    print(mem.get_short_term_context())
    
    print("\n[3] Testando Retrieval do Long Term...")
    ltm = mem.get_long_term_context("Quem sou eu?")
    print(ltm)
