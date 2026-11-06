# Project guide

Start with [README.md](README.md) for installation, database preparation, Docker commands, and limitations.

The application has five simple Python modules:

- `app/process_data.py`: extract PDF/TXT text, create page chunks, embed, and export artifacts.
- `app/retrieval.py`: BM25, semantic search, RRF, reranking, evidence, and prompts.
- `app/generate.py`: Qwen generation, optional rewriting, and bounded retrieval history.
- `app/database.py`: accounts, sessions, conversations, document metadata, and pgvector.
- `app/main.py`: FastAPI routes and startup.

The notebooks are experiments; `retrieve.ipynb` keeps its own functions so they can be edited directly. The API uses the Python modules. Saved notebook outputs are cleared for publication.

## Accounts and persistent chats

The API uses PostgreSQL for accounts, login sessions, conversations, and chat turns.
Copy `.env.example` to `.env` and set your database connection there, or export DATABASE_URL before starting the server. Exported variables override `.env`:

```bash
conda activate llm
export DATABASE_URL='postgresql://handbook:YOUR_DATABASE_PASSWORD@localhost:5433/handbook'
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Replace YOUR_DATABASE_PASSWORD with the PostgreSQL password. URL-encode special characters in credentials. Do not commit credentials.

Use http://127.0.0.1:8000/docs in this order:

1. POST /register: JSON email and password (12–128 characters).
2. POST /login: same JSON; copy access_token. Login tokens expire after one day.
3. Click Authorize in Swagger and paste the token (without adding Bearer).
4. POST /ask: form field question; omit conversation_id to create a new chat automatically.
5. For follow-ups, pass the returned conversation_id to POST /ask with use_history=true.
6. POST /conversations remains optional if you want to create an empty titled chat first.

The server loads the last three saved turns when use_history is true and saves every completed turn, including sources and timings. There is no longer a history_json field. When conversation_id is supplied, it must belong to the authenticated user. Omitting it creates a new owned conversation after answer generation, using the first question as its title.

Existing practice conversations without user_id are preserved but cannot be accessed through authenticated routes. New conversations always have an owner.

Passwords use salted PBKDF2-HMAC-SHA256 hashes. Only hashes of login tokens are stored in the sessions table. The existing temporary-upload behavior remains unchanged.


### Display chats in a web/mobile client

- GET /conversations: Bearer token required. Lists only the logged-in user's chats, most recently active first. Each item has id, title, created_at, updated_at, and turn_count.
- GET /conversations/{conversation_id}: Bearer token required. Returns the owned conversation and its saved turns, oldest first, including answers and sources. This reads saved records; it does not generate answers.
- Both endpoints accept limit (default 50, maximum 100) and offset (default 0). Use has_more to load the next page with offset increased by limit.
- To continue an opened conversation, send its ID to POST /ask.


## pgvector retrieval

The FastAPI startup now creates Retriever(database=app.state.database). It loads the shared lecture corpus from PostgreSQL, not from JSON/NumPy files. Semantic search uses pgvector cosine distance; BM25, reciprocal-rank fusion, and cross-encoder reranking remain in Python.

Tables added: documents, chunks (embedding VECTOR(768)), and embedding_settings. All imported lecture documents are shared corpus data. Private uploads remain temporary per request and are not inserted into shared tables. Their vectors use a connection-local temporary table for combined pgvector search.

To import or update saved embeddings after processing documents:

```bash
python -m app.database
```

Set DATABASE_URL first, as shown above. The importer validates model name, dimensions, normalization, and chunk counts; it reuses existing chunk IDs instead of duplicating them. It does not regenerate embeddings or delete local files. It does not delete stale rows for removed documents; deletion/version management is a future step. Restart the API after updating the corpus so its BM25 cache matches PostgreSQL.

Saved-vector search uses an HNSW cosine index. The code temporarily discourages sequential scans for that query so HNSW can be exercised with the small corpus; this is a learning choice, not a proven speed improvement. Uploaded vectors are searched separately and candidates are merged by cosine similarity. HNSW is approximate and can miss nearest neighbors. The Retriever caches vectors for temporary-upload handling; semantic ranking itself runs in PostgreSQL.

The standalone experiment notebooks and Retriever() without a database argument retain the original local-file mode for comparison. No notebook functions were replaced with imports. ProcessData(database=database) can export files and import them to PostgreSQL; ordinary ProcessData() continues to export files only.

Inspect stored data in pgAdmin:

```sql
SELECT COUNT(*) FROM documents;
SELECT COUNT(*) FROM chunks;
SELECT chunk_id, page, vector_dims(embedding) FROM chunks LIMIT 5;
```

## History and citation limitations

Without rewriting, the retrieval query starts with the current question and adds recent history only while it fits a 256-token budget. Oversized answers are omitted before dropping questions. A current question longer than this history budget is kept by itself, provided it fits the embedding model limit; longer questions are rejected explicitly. Generation receives previous questions and newly retrieved evidence, not previous answers.

Citation validation checks whether source IDs are present and known. It does not check whether each claim is supported by its cited passage. The default reranker threshold is a starting value, not a calibrated guarantee.
