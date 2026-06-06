"""
api.py — Adil : Agent IA Juridique Tchadien
Claude agit comme un vrai agent avec outils : RAG + Web Search
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client
import anthropic
import json

SUPABASE_URL  = os.environ["SUPABASE_URL"]
SUPABASE_KEY  = os.environ["SUPABASE_SERVICE_KEY"]
OPENAI_KEY    = os.environ["OPENAI_API_KEY"]
ANTHROPIC_KEY = os.environ["ANTHROPIC_API_KEY"]

EMBED_MODEL   = "text-embedding-3-small"
LLM_MODEL     = "claude-haiku-4-5-20251001"
TOP_K         = 6
MAX_TOKENS    = 2048

app      = FastAPI(title="Adil — Agent Juridique Tchadien")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
oai      = OpenAI(api_key=OPENAI_KEY)
claude   = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODÈLES ──────────────────────────────────────────────────────────────────
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

# ─── PROMPT SYSTÈME AGENT ─────────────────────────────────────────────────────
AGENT_SYSTEM = """Tu es ADIL, un agent juridique expert en droit tchadien. Tu aides les citoyens tchadiens à comprendre leurs droits.

## Ton identité
- Tu es un assistant juridique spécialisé dans le droit tchadien
- Tu as accès aux textes officiels du Tchad (codes, lois, ordonnances)
- Tu peux rechercher sur internet pour compléter tes connaissances

## Tes outils
1. **search_legal_documents** : recherche dans les textes juridiques tchadiens officiels
2. **web_search** : recherche sur internet pour des informations complémentaires

## Ta méthode de travail (agent)
1. TOUJOURS commencer par chercher dans les documents juridiques locaux
2. Si les résultats sont insuffisants ou incomplets, compléter avec une recherche web
3. Synthétiser les informations de toutes les sources
4. Donner une réponse claire, structurée et pratique

## Règles strictes
- Cite TOUJOURS les articles exacts quand disponibles (ex: "Art. 297 du Code Pénal Tchadien")
- Si tu n'as pas l'information, dis-le clairement et oriente vers un avocat
- Donne toujours des étapes pratiques concrètes
- Rappelle que tu n'es pas un avocat et que des cas complexes nécessitent un professionnel
- Réponds en français sauf si l'utilisateur écrit en arabe

## Format de réponse
- Réponse directe et claire
- Base légale (articles cités)
- Étapes pratiques numérotées
- ⚠️ Avertissement si nécessaire"""

# ─── OUTILS DE L'AGENT ────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "search_legal_documents",
        "description": "Recherche dans les textes juridiques officiels tchadiens (codes, lois, ordonnances). Utilise cet outil EN PREMIER pour toute question juridique.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La requête de recherche en français"
                },
                "domaine": {
                    "type": "string",
                    "description": "Domaine juridique optionnel (pénal, civil, travail, foncier, marchés)",
                    "default": ""
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "web_search",
        "description": "Recherche sur internet pour des informations juridiques complémentaires. Utilise uniquement si les documents locaux sont insuffisants.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La requête de recherche"
                }
            },
            "required": ["query"]
        }
    }
]

# ─── FONCTIONS DES OUTILS ─────────────────────────────────────────────────────
def search_legal_documents(query: str, domaine: str = "") -> dict:
    """Recherche vectorielle dans Supabase."""
    try:
        emb = oai.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
        params = {"query_embedding": emb, "match_count": TOP_K}
        if domaine:
            params["filter_domaine"] = domaine
        result = supabase.rpc("match_documents", params).execute()
        docs = result.data or []

        if not docs:
            return {"found": False, "message": "Aucun document pertinent trouvé.", "documents": []}

        max_sim = max(d.get("similarity", 0) for d in docs)
        formatted = []
        for d in docs:
            formatted.append({
                "source": d.get("source", ""),
                "article": d.get("article", ""),
                "content": d.get("content", ""),
                "similarity": round(d.get("similarity", 0), 3)
            })

        return {
            "found": True,
            "max_similarity": round(max_sim, 3),
            "count": len(formatted),
            "documents": formatted
        }
    except Exception as e:
        return {"found": False, "error": str(e), "documents": []}


def do_web_search(query: str) -> dict:
    """Recherche web via Claude."""
    try:
        response = claude.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": f"Recherche des informations juridiques précises sur: {query} — droit tchadien"
            }]
        )
        text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                text += block.text + "\n"
        return {"found": bool(text), "content": text.strip()}
    except Exception as e:
        return {"found": False, "error": str(e), "content": ""}


def execute_tool(name: str, inputs: dict) -> str:
    """Exécute un outil et retourne le résultat en JSON."""
    if name == "search_legal_documents":
        result = search_legal_documents(
            query=inputs.get("query", ""),
            domaine=inputs.get("domaine", "")
        )
    elif name == "web_search":
        result = do_web_search(inputs.get("query", ""))
    else:
        result = {"error": f"Outil inconnu: {name}"}
    return json.dumps(result, ensure_ascii=False)


# ─── BOUCLE AGENT ─────────────────────────────────────────────────────────────
def run_agent(question: str, domaine: str, historique: list) -> tuple[str, list, bool]:
    """
    Boucle agent : Claude utilise ses outils jusqu'à avoir une réponse complète.
    Retourne (réponse, sources, used_web)
    """
    messages = []

    # Historique
    for msg in historique[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # Question initiale avec contexte domaine
    user_content = question
    if domaine:
        user_content = f"[Domaine: {domaine}]\n{question}"
    messages.append({"role": "user", "content": user_content})

    sources = []
    used_web = False
    max_iterations = 4  # Limite les appels d'outils

    for iteration in range(max_iterations):
        response = claude.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            system=AGENT_SYSTEM,
            tools=TOOLS,
            messages=messages
        )

        # Si Claude a terminé
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    final_text += block.text
            return final_text, sources, used_web

        # Si Claude utilise des outils
        if response.stop_reason == "tool_use":
            # Ajoute la réponse de Claude avec les appels d'outils
            messages.append({"role": "assistant", "content": response.content})

            # Exécute chaque outil
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name   = block.name
                    tool_inputs = block.input

                    print(f"  🔧 Outil: {tool_name} | Inputs: {tool_inputs}")

                    # Exécute l'outil
                    result = execute_tool(tool_name, tool_inputs)
                    result_data = json.loads(result)

                    # Collecte les sources
                    if tool_name == "search_legal_documents" and result_data.get("found"):
                        for doc in result_data.get("documents", [])[:3]:
                            if doc.get("similarity", 0) > 0.2:
                                sources.append({
                                    "source": doc.get("source", ""),
                                    "article": doc.get("article", ""),
                                    "extrait": doc.get("content", "")[:150] + "...",
                                    "type": "document"
                                })

                    if tool_name == "web_search" and result_data.get("found"):
                        used_web = True
                        sources.append({
                            "source": "🌐 Recherche web",
                            "article": "",
                            "extrait": "Informations complémentaires via internet",
                            "type": "web"
                        })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Ajoute les résultats des outils
            messages.append({"role": "user", "content": tool_results})

        else:
            # Stop inattendu
            break

    # Fallback si on dépasse les itérations
    final_text = "Je n'ai pas pu trouver une réponse complète. Veuillez consulter un avocat tchadien."
    return final_text, sources, used_web


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "service": "Adil Agent Juridique Tchadien v3"}

@app.post("/question", response_model=ReponseRAG)
def poser_question(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide")

    print(f"\n📩 Question: {req.question}")

    reponse, sources_raw, used_web = run_agent(
        question=req.question,
        domaine=req.domaine,
        historique=req.historique
    )

    sources = [Source(**s) for s in sources_raw]

    return ReponseRAG(reponse=reponse, sources=sources, used_web=used_web)

@app.get("/domaines")
def liste_domaines():
    result = supabase.table("documents").select("source").execute()
    sources = list({r["source"] for r in result.data})
    return {"domaines": sorted(sources)}
