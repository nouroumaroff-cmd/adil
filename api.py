"""
api.py — Backend RAG + Web Search pour l'assistant juridique tchadien
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client
import anthropic

SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_SERVICE_KEY"]
OPENAI_KEY     = os.environ["OPENAI_API_KEY"]
ANTHROPIC_KEY  = os.environ["ANTHROPIC_API_KEY"]

EMBED_MODEL    = "text-embedding-3-small"
LLM_MODEL      = "claude-haiku-4-5-20251001"
TOP_K          = 6
MAX_TOKENS     = 1024
MIN_SIMILARITY = 0.3

app      = FastAPI(title="Adil — API Juridique Tchadienne")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
oai      = OpenAI(api_key=OPENAI_KEY)
claude   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    domaine: str = ""
    historique: list = []

class Source(BaseModel):
    source: str
    article: str
    extrait: str
    type: str = "document"

class ReponseRAG(BaseModel):
    reponse: str
    sources: list[Source]
    used_web: bool = False

SYSTEM_RAG = """Tu es Adil, un assistant juridique expert en droit tchadien.
Tu aides les citoyens tchadiens à comprendre leurs droits et les démarches juridiques.

RÈGLES :
- Base-toi PRIORITAIREMENT sur les textes juridiques fournis dans le contexte
- Cite toujours les articles exacts (ex: "Selon l'Art. 307 du Code Pénal...")
- Si l'information n'est pas dans le contexte, dis-le clairement
- Sois clair, concret, et donne des étapes pratiques
- Rappelle qu'un avocat reste conseillé pour les cas complexes
- Réponds en français"""

SYSTEM_WEB = """Tu es Adil, un assistant juridique expert en droit tchadien.
Les textes juridiques locaux ne contiennent pas assez d'informations sur cette question.
Utilise les résultats de recherche web fournis pour donner une réponse utile.
Précise que tu t'appuies sur des sources web et recommande de vérifier auprès d'un professionnel."""

def get_embedding(text: str) -> list[float]:
    resp = oai.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def search_documents(question: str, domaine: str = "") -> list[dict]:
    embedding = get_embedding(question)
    params = {"query_embedding": embedding, "match_count": TOP_K}
    if domaine:
        params["filter_domaine"] = domaine
    result = supabase.rpc("match_documents", params).execute()
    return result.data or []

def build_context(docs: list[dict]) -> tuple[str, float]:
    if not docs:
        return "", 0.0
    max_similarity = max(doc.get("similarity", 0) for doc in docs)
    context = "=== TEXTES JURIDIQUES TCHADIENS ===\n\n"
    for i, doc in enumerate(docs, 1):
        context += f"[{i}] {doc.get('source', '')} — {doc.get('article', '')}\n"
        context += f"{doc.get('content', '')}\n\n"
    return context, max_similarity

def search_web(question: str) -> str:
    """Recherche web via l'outil natif de Claude."""
    try:
        response = claude.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": f"Recherche des informations juridiques précises sur: {question} — droit tchadien, lois du Tchad."
            }]
        )
        web_content = ""
        for block in response.content:
            if hasattr(block, 'text'):
                web_content += block.text + "\n"
        return web_content.strip()
    except Exception as e:
        print(f"Erreur recherche web: {e}")
        return ""

@app.get("/")
def health():
    return {"status": "ok", "service": "Adil API v2 — RAG + Web"}

@app.post("/question", response_model=ReponseRAG)
def poser_question(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    used_web = False

    # 1. Recherche RAG
    docs = search_documents(req.question, req.domaine)
    context, max_similarity = build_context(docs)
    print(f"Similarité max: {max_similarity:.3f}")

    # 2. Si documents insuffisants → recherche web
    web_context = ""
    if max_similarity < MIN_SIMILARITY:
        print("→ Recherche web activée")
        web_context = search_web(req.question)
        used_web = bool(web_context)

    # 3. Prompt final
    if used_web and web_context:
        system = SYSTEM_WEB
        user_content = f"""=== RÉSULTATS WEB ===
{web_context}

=== DOCUMENTS LOCAUX (complément) ===
{context if context else "Aucun document local pertinent."}

---
Question: {req.question}"""
    else:
        system = SYSTEM_RAG
        user_content = f"{context}\n\n---\nQuestion: {req.question}"

    # 4. Historique
    messages = []
    for msg in req.historique[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    # 5. Claude
    response = claude.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
    )
    reponse_text = response.content[0].text

    # 6. Sources
    sources = []
    for doc in docs[:3]:
        sources.append(Source(
            source=doc.get("source", ""),
            article=doc.get("article", ""),
            extrait=doc.get("content", "")[:150] + "...",
            type="document"
        ))
    if used_web:
        sources.append(Source(
            source="🌐 Recherche web",
            article="",
            extrait="Résultats complémentaires via internet",
            type="web"
        ))

    return ReponseRAG(reponse=reponse_text, sources=sources, used_web=used_web)

@app.get("/domaines")
def liste_domaines():
    result = supabase.table("documents").select("source").execute()
    sources = list({r["source"] for r in result.data})
    return {"domaines": sorted(sources)}
