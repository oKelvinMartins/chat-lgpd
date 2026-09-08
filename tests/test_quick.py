"""
Teste RÁPIDO de Alucinação com RAGAS — 1 pergunta apenas
"""

# --- Workarounds ---
import sys, types
try:
    import _lzma
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

try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except (ImportError, ModuleNotFoundError):
    vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    class _Stub: pass
    vertexai_stub.ChatVertexAI = _Stub
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_stub
# --- End workarounds ---

import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from ragas.dataset_schema import SingleTurnSample
from ragas import EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, LLMContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

from packages.ai_agents.researcher import ResearcherAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    print("\n🛡️  TESTE RÁPIDO — 1 pergunta")
    print("=" * 50)

    # 1. Pipeline RAG
    researcher = ResearcherAgent()
    _orig = researcher.hybrid_search
    researcher.hybrid_search = lambda q, n=4: _orig(q, n)

    query = "O que são dados pessoais sensíveis segundo a LGPD?"
    reference = (
        "Segundo o Art. 5º, II da LGPD, dados pessoais sensíveis são dados sobre "
        "origem racial ou étnica, convicção religiosa, opinião política, filiação a "
        "sindicato ou a organização de caráter religioso, filosófico ou político, "
        "dado referente à saúde ou à vida sexual, dado genético ou biométrico, "
        "quando vinculado a uma pessoa natural."
    )

    print(f"\n📝 Pergunta: {query}")
    contexts = researcher.hybrid_search(query)
    response = researcher.research_and_answer(query)
    print(f"🤖 Resposta: {response.answer[:200]}...")
    print(f"📎 Citações: {response.citations}")

    sample = SingleTurnSample(
        user_input=query,
        response=response.answer,
        retrieved_contexts=contexts,
        reference=reference,
    )

    # 2. RAGAS evaluation
    print("\n⏳ Avaliando com RAGAS usando modelo local (Ollama - mistral-nemo)...")
    local_llm = ChatOpenAI(
        model="mistral-nemo",
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        temperature=0,
        timeout=300,
        max_retries=5,
    )
    evaluator_llm = LangchainLLMWrapper(local_llm)

    faithfulness = Faithfulness(llm=evaluator_llm)
    ctx_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    ctx_recall = LLMContextRecall(llm=evaluator_llm)

    results = evaluate(
        dataset=EvaluationDataset(samples=[sample]),
        metrics=[faithfulness, ctx_precision, ctx_recall],
    )

    # 3. Resultado
    df = results.to_pandas()
    print("\n" + "=" * 60)
    print("📊 RESULTADO DO TESTE RÁPIDO")
    print("=" * 60)
    import math
    for col in ["faithfulness", "context_precision", "context_recall"]:
        if col in df.columns:
            val = df[col].iloc[0]
            if isinstance(val, float) and math.isnan(val):
                print(f"  ⏭️  {col:30s} {'░' * 20} NaN (timeout)")
            else:
                emoji = "✅" if val >= 0.8 else "⚠️" if val >= 0.5 else "❌"
                bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                print(f"  {emoji} {col:30s} {bar} {val:.4f}")
    print("=" * 60)
    print("✅ Teste rápido concluído!")

if __name__ == "__main__":
    main()
