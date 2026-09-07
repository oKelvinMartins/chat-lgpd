import os
import sys
import logging
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import json
from rank_bm25 import BM25Okapi
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document

# Injecting the root of the project to allow absolute imports for packages
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from packages.shared_contracts.schemas import ResearcherResponse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

VECTORDB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vectordb")
CHUNKS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "lgpd_chunks.json")

class ResearcherAgent:
    """
    Researcher Agent: The core RAG synthesis engine.
    Queries ChromaDB and synthesizes an answer referencing exact articles.
    """
    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.model = model
        
        # Initialize VectorDB connection
        self.chroma_client = chromadb.PersistentClient(path=VECTORDB_DIR)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.chroma_client.get_collection(
            name="lgpd_collection",
            embedding_function=self.emb_fn
        )
        
        # Load corpus for BM25 and RRF retrieval
        logging.info("Loading local corpus for BM25 Hybrid Search...")
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)
        
        # Tokenize corpus for BM25
        self.tokenized_corpus = [c["texto_chunk"].lower().split() for c in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.system_prompt = (
            "Você é o Pesquisador Jurídico Principal focado na LGPD.\n"
            "Responda à pergunta do usuário usando ESTRITAMENTE o contexto recuperado abaixo.\n"
            "Se o contexto não for suficiente para uma resposta completa, diga isso (e marque is_fully_answered como falso).\n"
            "Para CADA afirmação feita, você DEVE citar explicitamente o artigo/inciso usado, e preencher o array de 'citations'.\n\n"
            "=== CONTEXTO RECUPERADO ==="
        )

    def hybrid_search(self, query: str, n_results: int = 7) -> list[str]:
        """Queries the vector DB (Semantic) and BM25 (Lexical), applies RRF and reorders context.
        
        Key improvement: Instead of returning individual chunks, we identify the most 
        relevant ARTICLES and return the full article text (parent + all children) for 
        maximum context quality.
        """
        logging.info(f"Running Hybrid Search for: '{query}'")
        
        # Fetch more candidates to improve recall before RRF fusion
        fetch_k = 15
        
        # 1. Semantic Search (ChromaDB)
        chroma_res = self.collection.query(
            query_texts=[query],
            n_results=fetch_k
        )
        
        # 2. Lexical Search (BM25)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]
        
        # 3. Reciprocal Rank Fusion (RRF) - score by ARTICLE, not by chunk
        article_scores = {}
        
        # Add Semantic ranks (k=60)
        if chroma_res and chroma_res.get("ids"):
            for rank, chunk_id in enumerate(chroma_res["ids"][0]):
                # Extract article identifier from chunk_id (e.g., "artigo_14._child_1_284" -> "14.")
                art = chroma_res["metadatas"][0][rank].get("artigo", "")
                if art:
                    article_scores[art] = article_scores.get(art, 0.0) + 1.0 / (60 + rank + 1)
                
        # Add Lexical ranks (k=60)
        for rank, idx in enumerate(top_n_bm25):
            art = self.corpus[idx].get("artigo", "")
            if art:
                article_scores[art] = article_scores.get(art, 0.0) + 1.0 / (60 + rank + 1)
            
        # 4. Sort articles by RRF score and pick top N
        sorted_articles = sorted(article_scores.keys(), key=lambda x: article_scores[x], reverse=True)[:n_results]
        logging.info(f"Top articles by RRF: {sorted_articles}")
        
        # 5. For each top article, assemble the FULL article text (parent + all unique children)
        context_docs = []
        for art_id in sorted_articles:
            # Gather all chunks for this article
            art_chunks = [c for c in self.corpus if c.get("artigo") == art_id]
            
            # Deduplicate by texto_chunk content
            seen_texts = set()
            unique_chunks = []
            for c in art_chunks:
                text = c["texto_chunk"].strip()
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_chunks.append(c)
            
            # Build the full article: parent first, then children in order
            parents = [c for c in unique_chunks if c.get("tipo") == "parent"]
            children = [c for c in unique_chunks if c.get("tipo") == "child"]
            
            full_text = f"[Referência: Art. {art_id}]\n"
            for p in parents:
                full_text += p["texto_chunk"] + "\n"
            for child in children:
                full_text += "  " + child["texto_chunk"] + "\n"
            
            context_docs.append(full_text.strip())
                    
        # 6. Long Context Reorder (Lost in the Middle mitigation)
        reordering = LongContextReorder()
        docs = [Document(page_content=t) for t in context_docs]
        reordered_docs = reordering.transform_documents(docs)
        
        return [d.page_content for d in reordered_docs]

    def research_and_answer(self, query: str) -> ResearcherResponse:
        # Step 1: Retrieval (Hybrid + Reorder)
        context_chunks = self.hybrid_search(query)
        formatted_context = "\n\n---\n\n".join(context_chunks)
        
        logging.info(f"Retrieved {len(context_chunks)} chunks for context.")
        
        # Step 2: Synthesis with Sandwich Prompting (Rules -> Context -> Reminder)
        full_system_prompt = (
            f"{self.system_prompt}\n"
            f"{formatted_context}\n"
            "===========================\n"
            "LEMBRETE: Responda apenas com base no contexto acima. Preencha rigorosamente a lista de 'citations'."
        )
        
        logging.info("Generating response with LLM...")
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": query},
                ],
                response_format=ResearcherResponse,
            )
            
            response = completion.choices[0].message.parsed
            logging.info(f"Answer synthesized! Citations used: {response.citations}")
            return response
            
        except Exception as e:
            logging.error(f"Failed to synthesize answer: {e}")
            raise

if __name__ == "__main__":
    researcher = ResearcherAgent()
    query = "O que são dados sensíveis?"
    print(f"\n[Pesquisando]: {query}")
    res = researcher.research_and_answer(query)
    
    print("\n[Resposta]:", res.answer)
    print("[Citações]:", res.citations)
    print("[Completa?]:", res.is_fully_answered)
