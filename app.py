# app.py
from dotenv import load_dotenv
load_dotenv()

import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd

# Project root (adjust if needed)
PROJ_ROOT = os.environ.get("FRAUD_BASE_DIR", "/content/drive/MyDrive/fraud_etl_rag_fastapi")
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

app = FastAPI(title="Fraud ETL + Improved Stage2 API (lazy imports)")

# lightweight UI endpoints (no heavy imports)
class GenerateReq(BaseModel):
    n_claims: int = 1000

@app.get("/health")
def health():
    return {"status": "ok", "proj_root": PROJ_ROOT}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    # Simple minimal UI to test deployment
    html = """
    <!doctype html>
    <html>
      <head><title>Fraud ETL API</title></head>
      <body>
        <h2>Fraud ETL + Stage2 API</h2>
        <p>Endpoints:</p>
        <ul>
          <li><a href="/docs">OpenAPI docs (Swagger)</a></li>
          <li>/health — health check</li>
          <li>/data/generate — POST to generate synthetic data</li>
          <li>/features/compute — POST to run feature computation</li>
          <li>/embeddings/build — POST to build embeddings + FAISS index</li>
          <li>/embeddings/info — GET index info</li>
          <li>/stage2/analyze?claim_id=&lt;id&gt; — GET to analyze a claim</li>
          <li>/candidates/list — GET to list top candidates</li>
        </ul>
      </body>
    </html>
    """
    return HTMLResponse(content=html)

# lazily import heavy modules
def _lazy_imports():
    """
    Import project modules on demand. Returns a dict of available callables.
    If an import fails, the value will be None — endpoints should raise helpful errors.
    """
    out = {}
    try:
        from src.etl import generate_synthetic_data
        out['generate_synthetic_data'] = generate_synthetic_data
    except Exception as e:
        out['generate_synthetic_data'] = None
        out.setdefault("_errors", []).append(f"etl import error: {e}")

    try:
        from src.features import compute_basic_features_and_stage1
        out['compute_basic_features_and_stage1'] = compute_basic_features_and_stage1
    except Exception as e:
        out['compute_basic_features_and_stage1'] = None
        out.setdefault("_errors", []).append(f"features import error: {e}")

    try:
        from src.embeddings_store import build_index, load_index, retrieve
        out['build_index'] = build_index
        out['load_index'] = load_index
        out['retrieve'] = retrieve
    except Exception as e:
        out['build_index'] = out['load_index'] = out['retrieve'] = None
        out.setdefault("_errors", []).append(f"embeddings_store import error: {e}")

    try:
        from src.stage2 import analyze_claim_id
        out['analyze_claim_id'] = analyze_claim_id
    except Exception as e:
        out['analyze_claim_id'] = None
        out.setdefault("_errors", []).append(f"stage2 import error: {e}")

    try:
        from src.docs import prepare_docs_from_raw
        out['prepare_docs_from_raw'] = prepare_docs_from_raw
    except Exception:
        out['prepare_docs_from_raw'] = None

    return out

#endpoints that call project functions (lazy import inside)

@app.post("/data/generate")
def data_generate(req: GenerateReq):
    modules = _lazy_imports()
    gen = modules.get('generate_synthetic_data')
    if gen is None:
        raise HTTPException(status_code=500, detail=f"generate_synthetic_data not available. Errors: {modules.get('_errors')}")
    res = gen(n_claims=req.n_claims)
    return {"status": "generated", **res}

@app.get("/ingest")
def ingest():
    raw = os.path.join(PROJ_ROOT, "data", "raw", "claims.csv")
    if not os.path.exists(raw):
        raise HTTPException(status_code=404, detail="raw claims.csv not found; generate or upload first")
    return {"status":"raw_exists", "path": raw}

@app.post("/features/compute")
def features_compute():
    modules = _lazy_imports()
    fn = modules.get('compute_basic_features_and_stage1')
    if fn is None:
        raise HTTPException(status_code=500, detail=f"compute_basic_features_and_stage1 not available. Errors: {modules.get('_errors')}")
    res = fn()
    return res

@app.post("/embeddings/build")
def embeddings_build():
    modules = _lazy_imports()
    fn = modules.get('build_index')
    if fn is None:
        raise HTTPException(status_code=500, detail=f"build_index not available. Errors: {modules.get('_errors')}")
    res = fn()
    return res

@app.get("/embeddings/info")
def embeddings_info():
    modules = _lazy_imports()
    fn = modules.get('load_index')
    if fn is None:
        raise HTTPException(status_code=500, detail=f"load_index not available. Errors: {modules.get('_errors')}")
    try:
        info = fn()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load_index raised: {e}")
    return info

@app.get("/stage2/analyze")
def stage2_analyze(claim_id: str, k: int = 5):
    modules = _lazy_imports()
    fn = modules.get('analyze_claim_id')
    if fn is None:
        raise HTTPException(status_code=500, detail=f"analyze_claim_id not available. Errors: {modules.get('_errors')}")
    try:
        res = fn(claim_id, k=k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return res

@app.get("/candidates/list")
def candidates_list(limit: int = 200):
    path = os.path.join(PROJ_ROOT, "data", "processed", "claims_stage1.parquet")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Stage1 file missing; run features compute")
    df = pd.read_parquet(path)
    df = df.sort_values("stage1_score", ascending=False).head(limit)
    return {"count": len(df), "candidates": df[['claim_id','stage1_score','provider_id','amount']].to_dict(orient='records')}

# simple startup event to log
@app.on_event("startup")
def _startup_log():
    print("Starting Fraud ETL API (lazy). PROJ_ROOT:", PROJ_ROOT)
