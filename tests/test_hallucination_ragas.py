"""
Teste de Alucinação com RAGAS — Chat LGPD
==========================================
Avalia quantitativamente o pipeline RAG multi-agente usando métricas RAGAS:
- Faithfulness: a resposta é fiel ao contexto recuperado? (anti-alucinação)
- Answer Relevancy: a resposta é relevante à pergunta?
- Context Precision: os chunks retornados são relevantes?
- Context Recall: o retriever trouxe informação suficiente?
"""

# --- Workaround: pyenv Python on macOS may lack _lzma C extension ---
# The `datasets` library (required by ragas) imports `lzma` unconditionally.
# We create a stub so the import chain doesn't break. lzma is not used in our flow.
import sys
import types

try:
    import _lzma  # noqa: F401
except ImportError:
    _lzma_stub = types.ModuleType("_lzma")
    _lzma_stub.LZMACompressor = None
    _lzma_stub.LZMADecompressor = None
    _lzma_stub.LZMAError = type("LZMAError", (Exception,), {})
    _lzma_stub.FORMAT_AUTO = 0
    _lzma_stub.FORMAT_XZ = 1
    _lzma_stub.FORMAT_ALONE = 2
    _lzma_stub.FORMAT_RAW = 3
    _lzma_stub.CHECK_NONE = 0
    _lzma_stub.CHECK_CRC32 = 1
    _lzma_stub.CHECK_CRC64 = 4
    _lzma_stub.CHECK_SHA256 = 10
    _lzma_stub.MF_HC3 = 3
    _lzma_stub.MF_HC4 = 4
    _lzma_stub.MF_BT2 = 18
    _lzma_stub.MF_BT3 = 19
    _lzma_stub.MF_BT4 = 20
    _lzma_stub.MODE_FAST = 1
    _lzma_stub.MODE_NORMAL = 2
    _lzma_stub.PRESET_DEFAULT = 6
    _lzma_stub.PRESET_EXTREME = 0
    _lzma_stub._encode_filter_properties = lambda *a, **kw: b""
    _lzma_stub._decode_filter_properties = lambda *a, **kw: {}
    _lzma_stub.is_check_supported = lambda check_id: False
    sys.modules["_lzma"] = _lzma_stub
# --- End _lzma workaround ---

# --- Workaround: ragas 0.4.x imports VertexAI from langchain-community but ---
# --- it was moved to langchain-google-vertexai in recent versions. We don't ---
# --- use VertexAI, so we create a stub to satisfy the import chain.         ---
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
except (ImportError, ModuleNotFoundError):
    import importlib

    # Ensure the parent module path exists
    if "langchain_community.chat_models" not in sys.modules:
        # If the parent exists but the submodule doesn't, just add the stub
        pass

    vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class _ChatVertexAIStub:
        """Stub class — VertexAI not available/needed."""
        pass

    vertexai_stub.ChatVertexAI = _ChatVertexAIStub
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_stub
# --- End VertexAI workaround ---

import os
import json
import asyncio
import logging
import time
from datetime import datetime

# Inject project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings

from packages.ai_agents.researcher import ResearcherAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# 1. DATASET DE AVALIAÇÃO — Perguntas sobre a LGPD com Ground Truth
# ============================================================================

EVALUATION_DATASET = [
    {
        "user_input": "O que são dados pessoais sensíveis segundo a LGPD?",
        "reference": (
            "Segundo o Art. 5º, II da LGPD, dados pessoais sensíveis são dados sobre "
            "origem racial ou étnica, convicção religiosa, opinião política, filiação a "
            "sindicato ou a organização de caráter religioso, filosófico ou político, "
            "dado referente à saúde ou à vida sexual, dado genético ou biométrico, "
            "quando vinculado a uma pessoa natural."
        ),
    },
    {
        "user_input": "Quais são os direitos do titular de dados?",
        "reference": (
            "O Art. 18 da LGPD garante ao titular os seguintes direitos: confirmação da "
            "existência de tratamento; acesso aos dados; correção de dados incompletos, "
            "inexatos ou desatualizados; anonimização, bloqueio ou eliminação de dados "
            "desnecessários, excessivos ou tratados em desconformidade; portabilidade dos "
            "dados; eliminação dos dados tratados com consentimento; informação sobre "
            "entidades públicas e privadas com as quais houve compartilhamento; informação "
            "sobre a possibilidade de não fornecer consentimento e consequências; e "
            "revogação do consentimento."
        ),
    },
    {
        "user_input": "Quais são as bases legais para tratamento de dados pessoais?",
        "reference": (
            "O Art. 7º da LGPD estabelece as bases legais para tratamento de dados pessoais: "
            "consentimento do titular; cumprimento de obrigação legal ou regulatória; "
            "execução de políticas públicas pela administração pública; realização de estudos "
            "por órgão de pesquisa; execução de contrato; exercício regular de direitos em "
            "processo judicial, administrativo ou arbitral; proteção da vida ou incolumidade "
            "física do titular ou de terceiro; tutela da saúde; atender aos interesses "
            "legítimos do controlador ou de terceiro; e proteção do crédito."
        ),
    },
    {
        "user_input": "O que é a ANPD e qual sua função?",
        "reference": (
            "A Autoridade Nacional de Proteção de Dados (ANPD) é o órgão da administração "
            "pública federal responsável por zelar pela proteção dos dados pessoais e por "
            "implementar e fiscalizar o cumprimento da LGPD, conforme Art. 55-A e seguintes. "
            "Compete à ANPD elaborar diretrizes para a Política Nacional de Proteção de "
            "Dados Pessoais e da Privacidade, fiscalizar e aplicar sanções, entre outras "
            "atribuições previstas no Art. 55-J."
        ),
    },
    {
        "user_input": "Quais são as sanções administrativas previstas na LGPD?",
        "reference": (
            "O Art. 52 da LGPD prevê as seguintes sanções administrativas aplicáveis pela "
            "ANPD: advertência com prazo para adoção de medidas corretivas; multa simples "
            "de até 2% do faturamento da empresa no último exercício, limitada a R$ 50 "
            "milhões por infração; multa diária; publicização da infração; bloqueio dos "
            "dados pessoais; eliminação dos dados pessoais; suspensão parcial do "
            "funcionamento do banco de dados por até 6 meses; suspensão do exercício da "
            "atividade de tratamento; e proibição parcial ou total do exercício de "
            "atividades relacionadas a tratamento de dados."
        ),
    },
    {
        "user_input": "O que é o encarregado de proteção de dados (DPO)?",
        "reference": (
            "Conforme Art. 5º, VIII e Art. 41 da LGPD, o encarregado pelo tratamento de "
            "dados pessoais é a pessoa indicada pelo controlador e operador para atuar como "
            "canal de comunicação entre o controlador, os titulares dos dados e a ANPD. "
            "Suas atividades incluem aceitar reclamações e comunicações dos titulares, "
            "prestar esclarecimentos e adotar providências; receber comunicações da ANPD; "
            "orientar funcionários e contratados sobre práticas de proteção de dados; e "
            "executar demais atribuições determinadas pelo controlador ou em normas "
            "complementares."
        ),
    },
    {
        "user_input": "Quando o tratamento de dados pessoais pode ser realizado sem consentimento?",
        "reference": (
            "O tratamento de dados pessoais sem consentimento pode ocorrer nas hipóteses "
            "previstas no Art. 7º, incisos II a X: cumprimento de obrigação legal; "
            "execução de políticas públicas; realização de estudos por órgão de pesquisa; "
            "execução de contrato; exercício regular de direitos; proteção da vida; "
            "tutela da saúde; legítimo interesse do controlador; e proteção do crédito. "
            "Para dados sensíveis, o Art. 11, II prevê hipóteses específicas."
        ),
    },
    {
        "user_input": "O que deve conter o relatório de impacto à proteção de dados pessoais?",
        "reference": (
            "Conforme Art. 5º, XVII e Art. 38 da LGPD, o relatório de impacto à proteção "
            "de dados pessoais deve conter, no mínimo, a descrição dos tipos de dados "
            "coletados, a metodologia utilizada para a coleta e para a garantia da "
            "segurança das informações e a análise do controlador com relação a medidas, "
            "salvaguardas e mecanismos de mitigação de risco adotados."
        ),
    },
    {
        "user_input": "Quais são as responsabilidades do controlador e do operador de dados?",
        "reference": (
            "O controlador, conforme Art. 5º, VI, é a pessoa a quem competem as decisões "
            "referentes ao tratamento de dados pessoais. O operador, conforme Art. 5º, VII, "
            "é quem realiza o tratamento em nome do controlador. Os Art. 42 a 45 da LGPD "
            "dispõem sobre responsabilidade e ressarcimento de danos: o controlador ou "
            "operador que causar dano patrimonial, moral, individual ou coletivo em "
            "violação à legislação é obrigado a repará-lo. O operador responde "
            "solidariamente quando descumprir obrigações da legislação ou instruções "
            "lícitas do controlador."
        ),
    },
    {
        "user_input": "Como deve ser feita a transferência internacional de dados segundo a LGPD?",
        "reference": (
            "O Art. 33 da LGPD permite a transferência internacional de dados pessoais "
            "apenas nos seguintes casos: para países ou organismos internacionais que "
            "proporcionem grau de proteção adequado; quando o controlador oferecer garantias "
            "suficientes (cláusulas contratuais específicas, normas corporativas globais, "
            "selos, certificados e códigos de conduta); quando necessária para cooperação "
            "jurídica internacional; quando necessária para proteção da vida; quando "
            "autorizada pela ANPD; quando resultar de compromisso em acordo de cooperação "
            "internacional; quando necessária para execução de política pública; ou quando "
            "o titular tiver dado consentimento específico."
        ),
    },
]


# ============================================================================
# 2. EXECUÇÃO DO PIPELINE RAG REAL
# ============================================================================

def run_pipeline_for_evaluation() -> list[SingleTurnSample]:
    """
    Executa o ResearcherAgent para cada pergunta do dataset.
    Coleta response + retrieved_contexts para avaliação RAGAS.
    
    Uses reduced context (n_results=4) to fit within Groq free-tier TPM limits,
    with retry logic and delays between questions.
    """
    logger.info("=" * 60)
    logger.info("FASE 1: Executando pipeline RAG para cada pergunta...")
    logger.info("=" * 60)

    researcher = ResearcherAgent()
    
    # Monkey-patch hybrid_search to use fewer results (4 instead of 7)
    # This keeps context within Groq free-tier token limits (8000 TPM)
    _original_hybrid_search = researcher.hybrid_search
    researcher.hybrid_search = lambda query, n_results=4: _original_hybrid_search(query, n_results)
    
    samples = []

    for i, item in enumerate(EVALUATION_DATASET):
        query = item["user_input"]
        reference = item["reference"]

        logger.info(f"\n[{i+1}/{len(EVALUATION_DATASET)}] Processando: '{query}'")

        # Delay between questions to avoid rate limiting (Groq free tier)
        if i > 0:
            delay = 5
            logger.info(f"   ⏳ Aguardando {delay}s para evitar rate limit...")
            time.sleep(delay)

        try:
            # 1. Retrieval: get contexts (used by RAGAS for evaluation)
            retrieved_contexts = researcher.hybrid_search(query)

            # 2. Generation: get answer (retry with smaller context on failure)
            max_retries = 2
            response = None
            for attempt in range(max_retries + 1):
                try:
                    response = researcher.research_and_answer(query)
                    break
                except Exception as e:
                    error_msg = str(e)
                    if ("413" in error_msg or "rate_limit" in error_msg.lower()) and attempt < max_retries:
                        # Reduce context further and retry
                        reduced_n = max(2, 4 - attempt - 1)
                        logger.warning(f"   ⚠️ Contexto muito grande. Tentando com n_results={reduced_n}...")
                        researcher.hybrid_search = lambda q, n=reduced_n: _original_hybrid_search(q, n)
                        retrieved_contexts = researcher.hybrid_search(query)
                        time.sleep(10)  # Extra delay after rate limit
                    elif "429" in error_msg and attempt < max_retries:
                        logger.warning(f"   ⚠️ Rate limit. Aguardando 20s...")
                        time.sleep(20)
                    else:
                        raise

            if response is None:
                logger.error(f"   ❌ Falha após {max_retries + 1} tentativas. Pulando pergunta.")
                continue

            # 3. Build RAGAS sample
            sample = SingleTurnSample(
                user_input=query,
                response=response.answer,
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            )
            samples.append(sample)

            logger.info(f"   → Resposta: {response.answer[:100]}...")
            logger.info(f"   → Citações: {response.citations}")
            logger.info(f"   → Contextos recuperados: {len(retrieved_contexts)}")

            # Reset to default n_results for next question
            researcher.hybrid_search = lambda query, n_results=4: _original_hybrid_search(query, n_results)

        except Exception as e:
            logger.error(f"   ❌ Erro ao processar pergunta: {e}")
            logger.info("   Pulando para a próxima pergunta...")
            # Reset hybrid_search for next question
            researcher.hybrid_search = lambda query, n_results=4: _original_hybrid_search(query, n_results)
            continue

    logger.info(f"\n✅ Pipeline concluído: {len(samples)}/{len(EVALUATION_DATASET)} perguntas processadas.")
    return samples


# ============================================================================
# 3. AVALIAÇÃO RAGAS
# ============================================================================

def run_ragas_evaluation(samples: list[SingleTurnSample]) -> dict:
    """
    Avalia os samples com métricas RAGAS usando Groq como LLM juiz.
    """
    logger.info("\n" + "=" * 60)
    logger.info("FASE 2: Avaliando com RAGAS...")
    logger.info("=" * 60)

    # Setup LLM evaluator via Groq
    # Using qwen3.8-27b — available on current Groq free tier
    groq_llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    evaluator_llm = LangchainLLMWrapper(groq_llm)

    # Define metrics
    # NOTE: ResponseRelevancy removed because it sends n>1 in API calls,
    # which Groq doesn't support. The 3 remaining metrics still provide
    # a comprehensive anti-hallucination evaluation:
    # - Faithfulness: is the answer grounded in retrieved context?
    # - Context Precision: are the retrieved chunks relevant?
    # - Context Recall: did the retriever fetch enough info?
    faithfulness = Faithfulness(llm=evaluator_llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    context_recall = LLMContextRecall(llm=evaluator_llm)

    # Create evaluation dataset
    eval_dataset = EvaluationDataset(samples=samples)

    # Run evaluation
    logger.info("Executando avaliação RAGAS (isso pode levar alguns minutos)...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_precision, context_recall],
    )

    return results


# ============================================================================
# 4. RELATÓRIO DE RESULTADOS
# ============================================================================

def print_report(results) -> None:
    """Imprime relatório formatado com os resultados da avaliação."""

    print("\n" + "=" * 80)
    print("📊  RELATÓRIO DE AVALIAÇÃO RAGAS — TESTE DE ALUCINAÇÃO")
    print("=" * 80)

    # Convert to pandas for nice display
    df = results.to_pandas()

    # Print per-question results
    print("\n📋 Resultados por Pergunta:")
    print("-" * 80)

    for idx, row in df.iterrows():
        query = row.get("user_input", "N/A")
        faith = row.get("faithfulness", "N/A")
        ctx_prec = row.get("context_precision", "N/A")
        ctx_recall = row.get("context_recall", "N/A")

        # Format scores
        faith_str = f"{faith:.2f}" if isinstance(faith, (int, float)) else str(faith)
        prec_str = f"{ctx_prec:.2f}" if isinstance(ctx_prec, (int, float)) else str(ctx_prec)
        recall_str = f"{ctx_recall:.2f}" if isinstance(ctx_recall, (int, float)) else str(ctx_recall)

        # Emoji indicators
        faith_emoji = "✅" if isinstance(faith, (int, float)) and faith >= 0.8 else "⚠️" if isinstance(faith, (int, float)) and faith >= 0.5 else "❌"

        print(f"\n  {idx+1}. {query[:70]}...")
        print(f"     {faith_emoji} Faithfulness: {faith_str}  |  🎯 Ctx Precision: {prec_str}  |  📚 Ctx Recall: {recall_str}")

    # Print aggregate scores
    print("\n" + "=" * 80)
    print("📈 SCORES MÉDIOS (AGGREGATE)")
    print("=" * 80)

    metrics_cols = ["faithfulness", "context_precision", "context_recall"]
    for col in metrics_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            emoji = "✅" if mean_val >= 0.8 else "⚠️" if mean_val >= 0.5 else "❌"
            bar = "█" * int(mean_val * 20) + "░" * (20 - int(mean_val * 20))
            print(f"  {emoji} {col:30s} {bar} {mean_val:.4f}")

    print("\n" + "-" * 80)
    print("LEGENDA: ✅ ≥ 0.80 (Bom)  |  ⚠️ ≥ 0.50 (Atenção)  |  ❌ < 0.50 (Crítico)")
    print("-" * 80)

    # Faithfulness interpretation
    faith_mean = df["faithfulness"].mean() if "faithfulness" in df.columns else 0
    if faith_mean >= 0.9:
        print("\n🎉 EXCELENTE! O pipeline raramente alucina. Faithfulness muito alto.")
    elif faith_mean >= 0.7:
        print("\n👍 BOM. O pipeline é majoritariamente fiel, mas há espaço para melhoria.")
    elif faith_mean >= 0.5:
        print("\n⚠️  ATENÇÃO. O pipeline alucina com frequência moderada. Revisar guardrails.")
    else:
        print("\n🚨 CRÍTICO. O pipeline alucina significativamente. Ação corretiva necessária.")

    return df


def save_results(df, results) -> str:
    """Salva resultados em JSON para análise posterior."""
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"ragas_evaluation_{timestamp}.json")

    # Build serializable report
    report = {
        "timestamp": timestamp,
        "num_samples": len(df),
        "aggregate_scores": {},
        "per_question": [],
    }

    metrics_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for col in metrics_cols:
        if col in df.columns:
            report["aggregate_scores"][col] = round(float(df[col].mean()), 4)

    for idx, row in df.iterrows():
        q_result = {"user_input": row.get("user_input", "")}
        for col in metrics_cols:
            if col in df.columns:
                val = row[col]
                q_result[col] = round(float(val), 4) if isinstance(val, (int, float)) else None
        report["per_question"].append(q_result)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"\n💾 Resultados salvos em: {filepath}")
    return filepath


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("\n🛡️  Chat LGPD — Teste de Alucinação com RAGAS")
    print("=" * 50)

    # Phase 1: Run pipeline
    samples = run_pipeline_for_evaluation()

    # Phase 2: Evaluate with RAGAS
    results = run_ragas_evaluation(samples)

    # Phase 3: Report
    df = print_report(results)

    # Phase 4: Save
    save_results(df, results)

    print("\n✅ Avaliação concluída com sucesso!")


if __name__ == "__main__":
    main()
