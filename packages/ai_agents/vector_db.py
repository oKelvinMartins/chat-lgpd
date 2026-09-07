import json
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Paths based on the monorepo structure
CHUNKS_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "lgpd_chunks.json")
VECTORDB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vectordb")

def load_chunks():
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}")
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def initialize_vector_db():
    logging.info("Initializing ChromaDB Client...")
    os.makedirs(VECTORDB_DIR, exist_ok=True)
    
    # Initialize persistent ChromaDB local client
    client = chromadb.PersistentClient(path=VECTORDB_DIR)
    
    # Configure Multilingual Embedding Function
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # We create a collection specifically for LGPD with the chosen embedding function
    collection = client.get_or_create_collection(
        name="lgpd_collection",
        embedding_function=emb_fn
    )
    
    chunks = load_chunks()
    logging.info(f"Loaded {len(chunks)} parent chunks from JSON.")
    
    documents = []
    metadatas = []
    ids = []
    
    # This loop assumes a Parent-Child RAG model where we might embed the 
    # paragraphs (children) but retain a reference to the full article (parent).
    # For now, we embed the full text of the article as the baseline.
    for i, chunk in enumerate(chunks):
        # Indexamos o texto específico daquele pedaço (caput ou inciso/parágrafo)
        documents.append(chunk["texto_chunk"])
        
        # Salvamos todos os metadados necessários para Retrieval/Hybrid Search
        meta = {
            "artigo": chunk.get("artigo", ""),
            "tipo": chunk.get("tipo", "unknown")
        }
        
        # Se for um filho, guardamos o ID do pai e o texto completo do pai para injetar no prompt
        if chunk.get("tipo") == "child":
            meta["parent_id"] = chunk.get("parent_id", "")
            # ChromaDB não aceita textos muito grandes ou nulos em metadados se não for string limpa,
            # mas vamos salvar o texto do pai para o Agente ter o contexto
            meta["parent_texto_completo"] = chunk.get("parent_texto_completo", "")
            
        metadatas.append(meta)
        
        # Garantindo ID absolutamente único 
        unique_string = f"{chunk['id']}_{i}"
        ids.append(unique_string)
        
    if collection.count() == 0:
        logging.info("Populating vector database for the first time... this may take a moment.")
        # Note: Chroma provides a default embedding function if none is specified,
        # but in production we'd inject OpenAIEmbeddings or similar here.
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info("Successfully ingested LGPD documents into VectorDB!")
    else:
        logging.info("Vector database already populated. Skipping ingestion.")
        
    return collection

if __name__ == "__main__":
    initialize_vector_db()
