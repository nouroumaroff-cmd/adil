"""
api.py — Backend RAG pour l'assistant juridique tchadien
Déployer sur Render (render.com) — Free tier
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client
import anthropic

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SUPABASE_URL      = os.environ["SUPABASE_URL"]
SUPABASE_KEY      = os.environ["SUPABASE_SERVICE_KEY"]
OPENAI_KEY        = os.environ["OPENAI_API_KEY"]
ANTHROPIC_KEY     = os.environ["ANTHROPIC_API_KEY"]

EMBED_MODEL       = "text-embedding-3-small"
LLM_MODEL         = "claude-haiku-4-5-20251001"   # Le moins cher, très rapide
TOP_K             = 5    # Nombre de chunks récupérés par question
MAX_TOKENS        = 1024

# ─── CLIENTS ──────────────────────────────────────────────────────────────────
app       = FastAPI(title="Adil — API Juridique Tchadienne")
supabase  = create_client(SUPABASE_URL, SUPABASE_KEY)
oai       = OpenAI(api_key=OPENAI_KEY)
claude    = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # En prod, mettre l'URL de ton frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODÈLES ──────────────────────────────────────────────────────────────────
class QuestionRequest(BaseModel):
    question: str
    domaine: str = ""          # ex: "penal", "civil", "travail"
    historique: list = []      # messages précédents [{role, content}]

class Source(BaseModel):
    source: str
    article: str
    extrait: str

class ReponseRAG(BaseModel):
    reponse: str
    sources: list[Source]

# ─── SYSTÈME RAG ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es Adil, un assistant juridique expert en droit tchadien.
Tu aides les citoyens tchadiens à comprendre leurs droits et les démarches juridiques.

RÈGLES STRICTES :
- Réponds UNIQUEMENT en te basant sur les textes juridiques fournis dans le contexte
- Cite toujours les articles exacts (ex: "Selon l'Art. 7 du Code Foncier...")
- Si l'information n'est pas dans le contexte, dis-le clairement
- Utilise un langage clair et accessible, pas de jargon inutile
- Donne des étapes concrètes quand c'est possible
- Rappelle toujours qu'un avocat reste conseillé pour les cas complexes
- Tu peux répondre en français ou en arabe selon la langue de l'utilisateur

FORMAT de réponse :
1. Réponse directe à la question
2. Base légale (articles cités)
3. Étapes pratiques si applicable
4. Avertissement si nécessaire"""


def get_embedding(text: str) -> list[float]:
    """Calcule l'embedding d'une question."""
    resp = oai.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def search_documents(question: str, domaine: str = "", top_k: int = TOP_K) -> list[dict]:
    """Recherche sémantique dans Supabase."""
    embedding = get_embedding(question)

    # Appel de la fonction RPC Supabase (match_documents)
    params = {
        "query_embedding": embedding,
        "match_count": top_k,
    }
    if domaine:
        params["filter_domaine"] = domaine

    result = supabase.rpc("match_documents", params).execute()
    return result.data or []


def build_context(docs: list[dict]) -> str:
    """Construit le contexte à partir des documents trouvés."""
    if not docs:
        return "Aucun document pertinent trouvé."

    context = "=== TEXTES JURIDIQUES PERTINENTS ===\n\n"
    for i, doc in enumerate(docs, 1):
        context += f"[{i}] Source: {doc.get('source', 'Inconnu')}"
        if doc.get("article"):
            context += f" — {doc['article']}"
        context += f"\n{doc.get('content', '')}\n\n"
    return context


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "Adil API Juridique Tchadienne"}


@app.post("/question", response_model=ReponseRAG)
def poser_question(req: QuestionRequest):
    """Endpoint principal : question → RAG → réponse avec sources."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    # 1. Recherche des documents pertinents
    docs = search_documents(req.question, req.domaine)

    # 2. Construction du contexte
    context = build_context(docs)

    # 3. Construction des messages pour Claude
    messages = []

    # Historique de conversation
    for msg in req.historique[-6:]:   # max 6 messages précédents
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Question actuelle avec contexte
    messages.append({
        "role": "user",
        "content": f"{context}\n\n---\nQuestion: {req.question}"
    })

    # 4. Appel Claude Haiku
    response = claude.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    reponse_text = response.content[0].text

    # 5. Formatage des sources
    sources = [
        Source(
            source=doc.get("source", ""),
            article=doc.get("article", ""),
            extrait=doc.get("content", "")[:150] + "...",
        )
        for doc in docs[:3]   # top 3 sources affichées
    ]

    return ReponseRAG(reponse=reponse_text, sources=sources)


@app.get("/domaines")
def liste_domaines():
    """Retourne la liste des domaines juridiques disponibles."""
    result = supabase.table("documents").select("source").execute()
    sources = list({r["source"] for r in result.data})
    return {"domaines": sorted(sources)}
