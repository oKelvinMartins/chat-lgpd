from pydantic import BaseModel, Field

class RouterDecision(BaseModel):
    is_lgpd_related: bool = Field(
        description="True se a pergunta do usuário for minimamente relacionada à LGPD, leis de privacidade, dados pessoais, ou uso de IA com dados."
    )
    confidence: float = Field(
        description="Confiança do agente nesta decisão, de 0.0 a 1.0"
    )
    reasoning: str = Field(
        description="Explicação passo a passo do porquê a pergunta foi classificada dessa forma."
    )
    refusal_message: str | None = Field(
        default=None,
        description="Se is_lgpd_related for falso, forneça uma mensagem educada de recusa informando que você só responde sobre LGPD."
    )

class ResearcherResponse(BaseModel):
    answer: str = Field(
        description="Resposta detalhada gerada pelo Agente baseada estritamente no contexto recuperado."
    )
    citations: list[str] = Field(
        description="Lista exata de citações (ex: ['Art. 18', 'Art. 5, § 1º']) usadas para montar a resposta."
    )
    is_fully_answered: bool = Field(
        description="True se o contexto forneceu informações suficientes para responder a pergunta completamente."
    )

class CriticDecision(BaseModel):
    is_faithful: bool = Field(
        description="True se a resposta gerada é estritamente baseada no contexto fornecido, sem inventar fatos ou misturar com conhecimento prévio."
    )
    reasoning: str = Field(
        description="Raciocínio detalhado explicando a análise de fidelidade, apontando qualquer alucinação se encontrada."
    )
    suggested_correction: str | None = Field(
        default=None,
        description="Se is_faithful for falso, forneça uma sugestão de como a resposta deveria ser reescrita para não alucinar."
    )
