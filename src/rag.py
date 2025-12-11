import json
from .embeddings_store import retrieve

SUSPICIOUS = ["999","urgent implant","invalid","not covered","external_implant"]

def analyze(claim_row, use_openai=False):
    query = f"Invoice for claim {claim_row['claim_id']} amount {claim_row['amount']}"
    chunks = retrieve(query, k=5)

    score = claim_row.get("stage1_score", 0) * 0.3
    reasons = []

    for c in chunks:
        t = c['text'].lower()
        for kw in SUSPICIOUS:
            if kw in t:
                score += 0.5
                reasons.append(f"keyword:{kw}")

    verdict = "legit"
    if score > 0.6:
        verdict = "suspicious"
    elif score > 0.3:
        verdict = "needs_more_info"

    return {
        "claim_id": claim_row["claim_id"],
        "verdict": verdict,
        "score": score,
        "reasons": reasons,
        "retrieved_docs": [c['doc_id'] for c in chunks]
    }
