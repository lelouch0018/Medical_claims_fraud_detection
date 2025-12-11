

#  Fraud ETL + LLM-Powered RAG Analysis System

*A complete end-to-end pipeline for fraud detection and claim analysis using ETL, semantic search, FAISS embeddings, and a FastAPI backend.*

This project was built for a hackathon focused on **reducing delays in medical claims scrutiny** by leveraging **LLMs and Retrieval-Augmented Generation (RAG)**. Instead of manually reviewing hundreds of invoices, the system automates extraction, flagging, and question-answering over claims and supporting documents.

The system performs:

*  **ETL Pipeline** (synthetic claims → cleaned features → Stage-1 numeric fraud scoring)
*  **Unstructured Document Processing** (text + OCR chunks from invoices/images)
*  **FAISS + Embedding Indexing** for semantic similarity search
*  **Stage-2 Heuristic Fraud Engine** (keywords + outliers + embedding signals)
*  **RAG Agent** that lets non-technical reviewers ask natural-language queries
*  **FastAPI backend** serving fraud analysis endpoints and interactive UI

---

#  **Project Structure**

```
fraud_etl_rag_fastapi/
│
├── app.py                     # FastAPI application (main entrypoint)
├── agent.py                   # LLM-powered RAG agent interface
│
├── src/
│   ├── etl.py                 # Synthetic claims generator (Stage-0)
│   ├── features.py            # Stage-1 feature engineering + scoring
│   ├── docs.py                # Invoice extraction, chunking, OCR/text handling
│   ├── embeddings_store.py    # FAISS index builder + loader + retriever
│   ├── stage2.py              # Stage-2 fraud analysis engine (semantic + heuristics)
│   └── __init__.py
│
├── data/
│   ├── raw/                   # raw claims + raw invoices/images
│   └── processed/             # stage1 outputs, review queues, parquet files
│
├── models/
│   ├── docs_metadata.json     # chunked unstructured doc metadata
│   ├── embeddings.npy         # embedding vectors
│   └── faiss_index.idx        # trained FAISS index
│
├── README.md                  # Project documentation
├── .env.example               # Template for environment variables
└── requirements.txt
```

---

#  **High-Level Architecture**

```
                           ┌──────────────────────────┐
                           │    Synthetic Claims       │
                           │     + Raw Documents       │
                           └────────────┬─────────────┘
                                        │
                             (ETL & Preprocessing)
                                        │
               ┌────────────────────────────────────────────────┐
               │                                                │
               ▼                                                ▼
     ┌─────────────────┐                              ┌────────────────────┐
     │ Stage-1 Rules    │                              │ Unstructured Docs  │
     │ (numeric checks) │                              │ Chunking + OCR     │
     └──────┬───────────┘                              └─────────┬──────────┘
            │                                                    │
            ▼                                                    ▼
     ┌─────────────────┐                               ┌───────────────────────┐
     │ stage1_score     │                               │ Embeddings + FAISS    │
     │ candidates list  │                               │ retrieve(top-k)        │
     └──────┬───────────┘                               └─────────┬─────────────┘
            │                                                    │
            └──────────────────┬─────────────────────────────────┘
                               ▼
                      ┌──────────────────────┐
                      │ Stage-2 Fraud Engine │
                      │ - amount outliers    │
                      │ - suspicious keywords │
                      │ - semantic closeness  │
                      └───────────┬──────────┘
                                  ▼
                        ┌────────────────────┐
                        │ LLM RAG Assistant  │
                        │ (OpenRouter LLMs)  │
                        └───────────┬────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ FASTAPI Endpoints    │
                         │ UI + JSON Interface  │
                         └──────────────────────┘
```

---

#  **Setup Instructions**

## **1. Clone the repository**

```bash
git clone https://github.com/<your-username>/fraud-etl-rag-fastapi.git
cd fraud-etl-rag-fastapi
```

---

## **2. Install Python dependencies**

```
pip install -r requirements.txt
```

---

## **3. Create your `.env` file**

Use the provided template:

```
cp .env.example .env
```

Open `.env` and fill in your API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

 *Never commit your real API key to GitHub.*

---

## **4. Set project root path**

FASTAPI and modules rely on a root folder variable.

In your shell:

```
export FRAUD_BASE_DIR=/absolute/path/to/fraud_etl_rag_fastapi
```

Alternatively—users can modify this inside code—
It defaults to:

```python
PROJ_ROOT = os.environ.get("FRAUD_BASE_DIR", os.path.dirname(__file__) + "/..")
```

---

#  **Running the Pipeline**

## **Step 1 — Generate Synthetic Claims**

```
POST /data/generate
```

This creates:

```
data/raw/claims.csv
```

---

## **Step 2 — Compute Stage-1 features**

```
POST /features/compute
```

Produces:

```
data/processed/claims_stage1.parquet
```

---

## **Step 3 — Process Unstructured Documents**

Automatically extracts:

* text from invoices
* OCR from PNGs
* chunks them
* stores metadata

Outputs:

```
models/docs_metadata.json
```

---

## **Step 4 — Build FAISS index**

```
POST /embeddings/build
```

Creates:

* `models/embeddings.npy`
* `models/faiss_index.idx`

---

## **Step 5 — Run Stage-2 Fraud Analysis**

```
GET /stage2/analyze?claim_id=C100123
```

Outputs:

* semantic similarity scores
* suspicious keyword signals
* amount outlier flags
* final stage2_score
* verdict (legit / needs_more_info / suspicious)

---

#  **LLM-Powered Natural Language Querying (RAG Agent)**

The `agent.py` module allows reviewers to ask questions like:

> *"Show me claims where providers billed excessive amounts and suspicious implants appear."*
> *"Explain why claim C100256 looks fraudulent."*

The agent:

1. Retrieves relevant invoice chunks via FAISS
2. Builds contextual prompt
3. Calls OpenRouter (Gemini, GPT-4.1, etc.)
4. Returns explanation + citations

Run interactively:

```
python agent.py
```



#  **Why This System Works**

* ✔ Fully automates early-stage fraud triage
* ✔ Integrates unstructured invoice analysis
* ✔ Uses LLMs to simplify complex audits
* ✔ Detects semantic anomalies beyond simple rules
* ✔ Provides natural language access for auditors
* ✔ Production-ready FastAPI backend



#  **Future Improvements**

* Replace heuristics with a learned stage-2 ML classifier
* Add provider-level anomaly analytics
* Expand UI (React or Streamlit)
* Deploy backend + vector DB on cloud (Railway / Render / GCP)



# Final Note

I built this project to demonstrate how modern **LLMs + ETL + RAG + heuristics** can dramatically reduce turnaround times in healthcare claim reviews. Even with synthetic data, the system is architected exactly like a real-world deployable solution.


