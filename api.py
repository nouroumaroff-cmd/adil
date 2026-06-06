"""
api.py — Adil : Assistant Juridique Tchadien
Architecture simple : RAG + Claude + Web fallback
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client
import anthropic

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_SERVICE_KEY"]
OPENAI_KEY    = os.environ["OPENAI_API_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

EMBED_MODEL   = "text-embedding-3-small"
LLM_MODEL     = "claude-haiku-4-5-20251001"
TOP_K         = 8
MAX_TOKENS    = 1500

app      = FastAPI(title="Adil — Assistant Juridique Tchadien")
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

SYSTEM = """Tu es ADIL, un assistant juridique expert en droit tchadien. Tu aides les citoyens tchadiens à comprendre leurs droits et les démarches juridiques.

## Règles
- Utilise les textes juridiques fournis en priorité
- Cite les articles exacts : "Selon l'Art. 297 du Code Pénal Tchadien..."
- Si le contexte contient la réponse, utilise-le OBLIGATOIREMENT
- Si le contexte est insuffisant, dis-le et donne quand même une réponse utile basée sur tes connaissances générales du droit tchadien
- Donne toujours des étapes pratiques concrètes
- Sois concis et clair
- Rappelle qu'un avocat reste conseillé pour les cas complexes

## Format
1. Réponse directe
2. Base légale (si disponible)
3. Étapes pratiques
4. Avertissement si nécessaire"""

def get_docs(question: str, domaine: str = "") -> list[dict]:
    emb    = oai.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding
    params = {"query_embedding": emb, "match_count": TOP_K}
    if domaine:
        params["filter_domaine"] = domaine
    result = supabase.rpc("match_documents", params).execute()
    return result.data or []

def build_context(docs: list[dict]) -> tuple[str, float]:
    if not docs:
        return "", 0.0
    max_sim = max(d.get("similarity", 0) for d in docs)
    ctx = "=== TEXTES JURIDIQUES TCHADIENS ===\n\n"
    for i, d in enumerate(docs, 1):
        sim = d.get("similarity", 0)
        ctx += f"[{i}] {d.get('source','')} — {d.get('article','')}\n"
        ctx += f"{d.get('content','')}\n\n"
    return ctx, max_sim

def web_search(question: str) -> str:
    try:
        r = claude.messages.create(
            model=LLM_MODEL,
            max_tokens=800,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": f"Recherche: {question} droit tchadien loi Tchad"}]
        )
        text = ""
        for block in r.content:
            if hasattr(block, 'text'):
                text += block.text
        return text.strip()
    except:
        return ""

@app.get("/")
def health():
    return {"status": "ok", "service": "Adil v3"}

@app.post("/question", response_model=ReponseRAG)
def poser_question(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    # 1. Recherche RAG
    docs    = get_docs(req.question, req.domaine)
    ctx, sim = build_context(docs)
    used_web = False
    web_ctx  = ""

    print(f"Question: {req.question} | Sim max: {sim:.3f}")

    # 2. Web search si similarité faible
    if sim < 0.35:
        print("→ Web search activé")
        web_ctx  = web_search(req.question)
        used_web = bool(web_ctx)

    # 3. Construction du prompt
    if web_ctx:
        context_final = f"{ctx}\n=== RÉSULTATS WEB ===\n{web_ctx}"
    else:
        context_final = ctx

    # 4. Messages
    messages = []
    for msg in req.historique[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"{context_final}\n\n---\nQuestion du citoyen: {req.question}"
    })

    # 5. Claude répond
    response = claude.messages.create(
        model=LLM_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM,
        messages=messages,
    )
    reponse = response.content[0].text

    # 6. Sources (top 3 les plus pertinentes)
    sources = []
    for d in docs[:3]:
        if d.get("similarity", 0) > 0.2:
            sources.append(Source(
                source=d.get("source", ""),
                article=d.get("article", ""),
                extrait=d.get("content", "")[:120] + "...",
                type="document"
            ))

    if used_web:
        sources.append(Source(
            source="🌐 Recherche web",
            article="",
            extrait="Résultats complémentaires via internet",
            type="web"
        ))

    return ReponseRAG(reponse=reponse, sources=sources, used_web=used_web)

@app.get("/domaines")
def liste_domaines():
    result  = supabase.table("documents").select("source").execute()
    sources = list({r["source"] for r in result.data})
    return {"domaines": sorted(sources)}
