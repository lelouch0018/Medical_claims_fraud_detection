import json
from pathlib import Path
from .config import DOCS_RAW_DIR, DOCS_EXTRACTED_DIR, DOCS_CHUNKS_JSON
DOCS_EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

def prepare_docs_from_raw(metadata_json_path):
    meta = json.load(open(metadata_json_path))

    chunks = []

    for m in meta:
        raw_file = DOCS_RAW_DIR / m['file']
        if raw_file.suffix == ".txt":
            text = open(raw_file, encoding='utf-8').read()
        else:
            text = f"[PLACEHOLDER TEXT for {m['file']}]"

        for i in range(0, len(text), 800):
            chunk = text[i:i+800].strip()
            if chunk:
                chunks.append({
                    "doc_id": f"{m['claim_id']}_{m['file']}_chunk{i}",
                    "claim_id": m['claim_id'],
                    "provider_id": m.get("provider_id"),
                    "text": chunk
                })

    with open(DOCS_CHUNKS_JSON, "w") as f:
        json.dump(chunks, f, indent=2)

    return {"chunks": len(chunks), "saved_to": str(DOCS_CHUNKS_JSON)}
