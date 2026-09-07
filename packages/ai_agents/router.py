import os
import sys
import logging
from openai import OpenAI

# Injecting the root of the project to allow absolute imports for packages
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from packages.shared_contracts.schemas import RouterDecision

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class RouterAgent:
    """
    Router Agent: Input Guardrail.
    Uses OpenAI Structured Outputs (Pydantic) to strictly evaluate if a query is LGPD-related.
    """
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = model
        
        self.system_prompt = (
            "Você é o Router (Input Guardrail) de um Chatbot Jurídico rigoroso focado 100% na LGPD (Lei Geral de Proteção de Dados - Brasil).\n"
            "Sua única função é determinar se a pergunta do usuário possui alguma relação com privacidade, proteção de dados, segurança da informação, ou LGPD.\n"
            "Responda EXATAMENTE dentro do esquema JSON fornecido. Você NUNCA deve responder a pergunta, apenas avaliá-la."
        )

    def evaluate_query(self, query: str) -> RouterDecision:
        logging.info(f"RouterAgent evaluating query: '{query}'")
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query},
                ],
                response_format=RouterDecision,
            )
            
            decision = completion.choices[0].message.parsed
            logging.info(f"Router Decision: {decision.is_lgpd_related} (Confidence: {decision.confidence})")
            return decision
            
        except Exception as e:
            logging.error(f"Failed to route query: {e}")
            raise

if __name__ == "__main__":
    # Test queries
    test_queries = [
        "Quais são os direitos do titular de dados segundo a LGPD?",
        "Qual a receita para fazer um bolo de chocolate perfeito?",
        "Minha empresa sofreu um vazamento de dados, o que faço?"
    ]
    
    router = RouterAgent()
    for q in test_queries:
        res = router.evaluate_query(q)
        print(f"\nQuery: {q}")
        print(f"Is LGPD Related? {res.is_lgpd_related}")
        print(f"Reasoning: {res.reasoning}")
        if res.refusal_message:
            print(f"Refusal Message: {res.refusal_message}")
