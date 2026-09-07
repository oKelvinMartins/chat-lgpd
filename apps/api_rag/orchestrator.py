import os
import sys
import logging
from openai import OpenAI

# Injecting root path for packages
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from packages.ai_agents.router import RouterAgent
from packages.ai_agents.researcher import ResearcherAgent
from packages.ai_agents.critic import CriticAgent
from packages.ai_agents.memory import MemoryManager

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class RAGOrchestrator:
    def __init__(self, model_name: str = "groq/compound-mini"):
        self.router = RouterAgent(model="openai/gpt-oss-20b")
        self.researcher = ResearcherAgent(model="openai/gpt-oss-20b")
        self.critic = CriticAgent(model="openai/gpt-oss-20b")
        
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = model_name

    def condense_question(self, short_term_context: str, user_query: str) -> str:
        """Query Reformulator: Extrai a pergunta isolada considerando o histórico."""
        if not short_term_context.strip():
            return user_query
            
        prompt = (
            "Dada a seguinte conversa recente e uma nova pergunta, reescreva a nova pergunta para que "
            "ela faça sentido por si só, sem depender do histórico. Seja muito objetivo.\n"
            "Se a pergunta já for independente, apenas repita ela.\n\n"
            f"{short_term_context}\n"
            f"Nova pergunta original: {user_query}\n\n"
            "Pergunta reescrita isolada (apenas a pergunta):"
        )
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Failed to condense question: {e}")
            return user_query

    def process_message(self, session_id: str, user_id: str, message: str) -> str:
        logging.info(f"--- Iniciando Orquestração para Sessão {session_id} ---")
        try:
            # 1. Carregar Memórias
            memory = MemoryManager(session_id=session_id, user_id=user_id, model=self.model)
            short_context = memory.get_short_term_context()
            long_context = memory.get_long_term_context(message)
            
            # 2. Reformular Pergunta
            standalone_query = self.condense_question(short_context, message)
            logging.info(f"Standalone Query: {standalone_query}")
            
            # 3. Router
            router_decision = self.router.evaluate_query(standalone_query)
            if not router_decision.is_lgpd_related:
                logging.warning("Router abortou: Fora do escopo LGPD.")
                memory.add_interaction(message, router_decision.refusal_message)
                return router_decision.refusal_message
                
            # 4. Enriquecer Query para o Researcher
            enriched_query = standalone_query
            if long_context:
                enriched_query = f"{long_context}\nPergunta Atual: {standalone_query}"
                
            # 5. Researcher + Critic
            max_retries = 1
            current_attempt = 0
            final_answer = ""
            
            while current_attempt <= max_retries:
                logging.info(f"Researcher: Gerando resposta (Tentativa {current_attempt + 1})")
                research_response = self.researcher.research_and_answer(enriched_query)
                final_answer = research_response.answer
                
                logging.info("Critic: Avaliando resposta gerada")
                critic_decision = self.critic.evaluate_output(enriched_query, final_answer)
                
                if critic_decision.is_faithful:
                    logging.info("Critic APROVOU a resposta.")
                    break
                else:
                    logging.warning(f"Critic REPROVOU. Motivo: {critic_decision.reasoning}")
                    enriched_query = (
                        f"SUA TENTATIVA ANTERIOR FOI REJEITADA PELO REVISOR.\n"
                        f"MOTIVO: {critic_decision.reasoning}\n"
                        f"SUGESTÃO: {critic_decision.suggested_correction}\n\n"
                        f"Corrija e melhore a resposta para a seguinte pergunta: {standalone_query}"
                    )
                    current_attempt += 1
                    
            # 6. Atualizar Histórico
            memory.add_interaction(message, final_answer)
            logging.info("--- Orquestração Concluída ---")
            return final_answer

        except Exception as e:
            logging.error(f"Orchestration Error: {e}")
            return "Desculpe, ocorreu um erro na comunicação com o provedor de IA (provavelmente limite de requisições excedido ou falha de conexão). Por favor, aguarde alguns segundos e tente novamente."
