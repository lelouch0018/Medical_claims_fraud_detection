
"""
RAG Agent using OpenRouter API + Stage-2 Analysis
-------------------------------------------------
This agent:
 - Uses `retrieve()` from embeddings_store
 - Uses stage2.analyze_claim_id()
 - Calls OpenRouter LLM for natural-language answers
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
from typing import List, Dict, Any

# Ensure project root is importable
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY. Set it in your .env file.")

# OpenRouter endpoint
import requests

# Local RAG tools
from src.embeddings_store import retrieve
from src.stage2 import analyze_claim_id

# ------------------------------
# Utility: format context
# ------------------------------
def build_context_from_chunks(chunks: List[Dict[str,Any]], max_chars: int = 3500) -> str:
    parts, total = [], 0
    for c in chunks:
        snippet = c.get("text") or c.get("text_preview") or ""
        meta = f"[doc_id={c.get('doc_id')} claim={c.get('claim_id')} dist={c.get('distance'):.3f}] "
        piece = meta + snippet.strip()
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(parts) if parts else "No relevant documents retrieved."

# ------------------------------
# Core RAG Answer Function
# ------------------------------
def answer_with_rag(query: str, model: str = "google/gemini-flash-1.5", k: int = 5):
    # Retrieve chunks
    try:
        chunks = retrieve(query, k=k)
    except Exception as e:
        print("Retriever error:", e)
        chunks = []

    context = build_context_from_chunks(chunks)

    prompt = f"""
You are an insurance-claims analysis assistant.
Use the retrieved documents to answer.

User question:
{query}

Retrieved context:
{context}

If asked about a claim, reference doc_ids when relevant.
"""

    # Call OpenRouter
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You specialize in analyzing insurance claims using RAG."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter Error {response.status_code}: {response.text}")

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    return {
        "question": query,
        "answer": answer,
        "chunks": chunks
    }

# ------------------------------
# Extract claim IDs (optional)
# ------------------------------
def extract_claim_ids(text: str):
    import re
    return re.findall(r"\bC\d{6}\b", text)

# ------------------------------
# Interactive mode
# ------------------------------
if __name__ == "__main__":
    print("RAG Agent Ready. Ask questions or type: analyze C123456")
    while True:
        try:
            q = input("\n> ").strip()
        except KeyboardInterrupt:
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            print("Bye")
            break

        # Direct stage-2 call
        if q.lower().startswith("analyze "):
            cid = q.split()[1]
            print(f"\nRunning stage-2 on {cid}...\n")
            try:
                print(json.dumps(analyze_claim_id(cid), indent=2))
            except Exception as e:
                print("Error:", e)
            continue

        # Otherwise do RAG
        out = answer_with_rag(q)
        print("\n--- RAG Answer ---")
        print(out["answer"])

        # Auto-detect claim IDs
        ids = extract_claim_ids(out["answer"])
        if ids:
            print("\nDetected claim IDs:", ids)
