import asyncio
import os
from dotenv import load_dotenv

# Carrega variáveis do arquivo .env raiz
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apps.api_rag.orchestrator import RAGOrchestrator

app = FastAPI(title="LGPD RAG API", description="API Orquestradora do Chatbot LGPD", version="1.0.0")

# Preparação para o Frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Na produção, usar localhost:3000 etc
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializa as instâncias dos Agentes (singleton para a API inteira)
orchestrator = RAGOrchestrator()

class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Recebe a mensagem, orquestra com Agentes (Router -> Researcher -> Critic) + Memória, e retorna a resposta.
    """
    # Usamos to_thread pois o orchestrator.process_message é síncrono e CPU/IO bound (Chroma, Ollama HTTP)
    answer = await asyncio.to_thread(
        orchestrator.process_message,
        request.session_id,
        request.user_id,
        request.message
    )
    return ChatResponse(answer=answer)
