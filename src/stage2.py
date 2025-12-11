import os, sys, json, time, numpy as np, pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Imports from project's src
from .config import MODELS_DIR, PROCESSED_DIR, DOCS_CHUNKS_JSON
from .embeddings_store import retrieve

# Define paths using config
PROC_STAGE1 = PROCESSED_DIR / "claims_stage1.parquet"
OUT_PARQUET = PROCESSED_DIR / "review_queue_improved.parquet"
OUT_JSON = PROCESSED_DIR / "review_queue_improved.json"

def analyze_claim_id(claim_id: str, k: int = 5):
    df = pd.read_parquet(PROC_STAGE1)
    # Ensure the DataFrame is not empty and claim_id exists before proceeding
    if df.empty or claim_id not in df['claim_id'].values:
        return {"claim_id": claim_id, "verdict_improved": "not_found", "score": 0.0, "reasons": ["Claim ID not found or no claims processed."]}

    claim_row = df[df['claim_id'] == claim_id].iloc[0]

    all_claims = pd.read_parquet(PROC_STAGE1)
    by_code = all_claims.groupby('procedure_code')['amount']
    q1 = by_code.quantile(0.25)
    q3 = by_code.quantile(0.75)
    iqr = q3 - q1
    proc_q3 = q3.to_dict()
    proc_iqr = iqr.to_dict()

    SUSPICIOUS_KEYWORDS = ["external_urgent_implant","not covered","invalid license","billed hours: 999","duplicate charge","999"]

    def keyword_matches_in_chunks(chunks, keywords):
        count = 0
        for c in chunks:
            txt = c['text'].lower()
            for kw in keywords:
                if kw.lower() in txt:
                    count += 1
        return count

    q = f"Invoice for claim {claim_row['claim_id']} amount {claim_row['amount']} procedure {claim_row['procedure_code']}"
    try:
        chunks = retrieve(q, k=k)
    except Exception as e:
        chunks = []
        print(f"Retrieval error for claim {claim_id}: {e}")

    score = claim_row.get('stage1_score', 0) * 0.25

    reasons = []
    proc = claim_row.get('procedure_code')
    q3_val = proc_q3.get(proc, None)
    iqr_val = proc_iqr.get(proc, None)
    is_amount_outlier = False
    if q3_val is not None and iqr_val is not None and (not np.isnan(q3_val)) and (not np.isnan(iqr_val)):
        threshold = q3_val + 3 * iqr_val
        if claim_row.get('amount', 0) > threshold:
            is_amount_outlier = True
            score += 0.45
            reasons.append("amount_outlier")

    km = keyword_matches_in_chunks(chunks, SUSPICIOUS_KEYWORDS)
    if km > 0:
        score += 0.35 * min(1.0, km)
        reasons.append(f"keyword_matches:{km}")

    if chunks:
        first_dist = float(chunks[0].get('distance', 1.0))
        bonus = max(0.0, 0.15 * (1 - first_dist/(first_dist+1)))
        if bonus > 0.001:
            score += bonus
            reasons.append(f"embed_bonus:{round(bonus,3)}")

    score = float(min(1.0, score))

    if score >= 0.7:
        verdict = "suspicious"
    elif score >= 0.35:
        verdict = "needs_more_info"
    else:
        verdict = "legit"

    return {
        "claim_id": claim_row['claim_id'],
        "provider_id": claim_row['provider_id'],
        "amount": float(claim_row['amount']),
        "stage1_score": int(claim_row['stage1_score']),
        "is_fraud_label": int(claim_row.get('is_fraud_label', 0)),
        "is_amount_outlier": is_amount_outlier,
        "keyword_matches": km,
        "stage2_score_improved": score,
        "verdict_improved": verdict,
        "reasons": reasons,
        "retrieved_docs": [{"doc_id": c['doc_id'], "distance": c['distance'], "text_preview": c['text'][:200]} for c in chunks]
    }

def process_all_candidates():
    df = pd.read_parquet(PROC_STAGE1)
    cands = df[df['stage1_score'] >= 1].copy().reset_index(drop=True)
    if cands.empty:
        res = {"candidates_processed": 0, "results_saved": False}
        print("No candidates found in claims_stage1.parquet.")
        return res

    t0 = time.time()
    results = [analyze_claim_id(row['claim_id']) for _, row in cands.iterrows()]
    t1 = time.time()
    print(f"Processed {len(results)} candidates in {round(t1-t0,2)}s")

    pd.DataFrame(results).to_parquet(OUT_PARQUET, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved improved review queue to:", OUT_PARQUET)
    print("Saved JSON to:", OUT_JSON)

    res_df = pd.DataFrame(results)
    res_df['pred_flag'] = res_df['verdict_improved'].map(lambda v: 1 if v in ("suspicious","needs_more_info") else 0)
    y_true = res_df['is_fraud_label'].astype(int)
    y_pred = res_df['pred_flag'].astype(int)

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n=== Evaluation on synthetic labels ===")
    print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
    print("Confusion matrix (rows: true, cols: pred):")
    print(cm)

    print("\n--- Done. ---")
    return {"candidates_processed": len(results), "results_saved": True, "precision": prec, "recall": rec, "f1": f1}
