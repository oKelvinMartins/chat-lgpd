import os
import sys
import logging
from openai import OpenAI

# Injecting the root of the project to allow absolute imports for packages
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from packages.shared_contracts.schemas import CriticDecision

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class CriticAgent:
    """
    Critic Agent: Output Guardrail.
    Evaluates if the Researcher's answer is faithful to the provided context.
    """
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = model
        
        self.system_prompt = (
            "Você é o Juiz de Fidelidade (Output Guardrail) de um Chatbot Jurídico rigoroso.\n"
            "Sua única função é comparar a Resposta Gerada pelo Pesquisador com o Contexto fornecido.\n"
            "Se a resposta afirmar QUALQUER COISA que não esteja explicitamente no Contexto, ela é considerada uma alucinação e você deve marcar is_faithful como falso.\n"
            "Responda EXATAMENTE no formato JSON fornecido."
        )

    def evaluate_output(self, context: str, answer: str) -> CriticDecision:
        logging.info("CriticAgent evaluating answer for hallucinations...")
        
        evaluation_prompt = (
            f"=== CONTEXTO DA LEI ===\n{context}\n\n"
            f"=== RESPOSTA DO PESQUISADOR ===\n{answer}\n\n"
            "A resposta acima alucina fatos ou é 100% fiel ao contexto?"
        )
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": evaluation_prompt},
                ],
                response_format=CriticDecision,
            )
            
            decision = completion.choices[0].message.parsed
            logging.info(f"Critic Decision: Faithful? {decision.is_faithful}")
            return decision
            
        except Exception as e:
            logging.error(f"Failed to critique answer: {e}")
            raise

if __name__ == "__main__":
    critic = CriticAgent()
    
    mock_context = "Art. 18. O titular dos dados tem direito a obter do controlador, em relação aos dados do titular por ele tratados, a qualquer momento e mediante requisição: I - confirmação da existência de tratamento;"
    
    # Test 1: Faithful Answer
    faithful_answer = "Segundo o Art. 18, inciso I, o titular dos dados tem o direito de requisitar a confirmação da existência de tratamento dos seus dados."
    
    # Test 2: Hallucinated Answer
    hallucinated_answer = "Segundo o Art. 18, o titular pode pedir a confirmação do tratamento de dados e também exigir uma indenização financeira imediata de 50 mil reais do controlador."
    
    print("\n--- Test 1 (Faithful) ---")
    res1 = critic.evaluate_output(mock_context, faithful_answer)
    print(f"Is Faithful? {res1.is_faithful}")
    print(f"Reasoning: {res1.reasoning}")
    
    print("\n--- Test 2 (Hallucination) ---")
    res2 = critic.evaluate_output(mock_context, hallucinated_answer)
    print(f"Is Faithful? {res2.is_faithful}")
    print(f"Reasoning: {res2.reasoning}")
    if res2.suggested_correction:
        print(f"Correction: {res2.suggested_correction}")
