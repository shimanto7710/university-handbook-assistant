# University Handbook Assistant

A conversational RAG application for studying from lecture PDFs and text documents. It retrieves relevant passages, generates an answer, and returns the PDF name and page number so the reader can check the source.

## Why I built it

Before exams, useful information is often spread across several lecture files. I wanted to ask a question across those files, compare the relevant passages, and get one answer without losing track of where the information came from.

I started in notebooks to understand each step, then moved the working pipeline into Python classes and added a FastAPI backend. The project now includes user accounts, saved conversations, optional document uploads, and PostgreSQL storage for both chat records and document embeddings.

An example question is: **“What are the disadvantages of cloud computing?”** A follow-up could be: **“Explain the security concerns in more detail.”** The API retrieves evidence again for each question; it does not treat an earlier generated answer as a source document.

## The pipeline

```text
Preparation
PDF / TXT → extract text → page chunks → E5 embeddings → PostgreSQL + pgvector

Answering
Question + optional history / uploaded file
    → BM25 + vector search
    → reciprocal-rank fusion
    → cross-encoder reranking
    → relevance filtering and prompt budgeting
    → Qwen answer + source metadata
```

The shared lecture corpus used during development contained **579 nonempty page chunks**. The lecture files, generated embeddings, model weights, and database records are excluded from this repository. You can prepare your own PDF or TXT files using the steps below.

## What the application supports

- Search across the shared corpus using keyword and semantic retrieval.
- Upload a PDF or TXT file and search it together with the existing documents.
- Register, log in, start a conversation, and retrieve saved chats.
- Use the last three conversation turns for follow-up retrieval, with optional question rewriting.
- Return source IDs, filenames, page numbers, retrieved text, and reranker scores alongside an answer.
- Report time spent on rewriting, retrieval, reranking, and generation.

Uploads are temporary for one request. Their extracted source text may be saved with the chat, but the uploaded document is not available for retrieval in later requests unless it is uploaded again.

## Decisions and what I learned

**Hybrid search before generation.** Keyword search helps with exact terms, while embeddings find related wording. I combine their rankings with reciprocal-rank fusion, then rerank a candidate pool of 30 passages before selecting up to five for the answer. These are configurable defaults, not benchmark-derived optimal values.

**Page chunks for this dataset.** Keeping a lecture page together makes its source reference easy to follow. I check token lengths before embedding. This works for the lecture material used here, but long pages and text files need splitting; the preparation script rejects over-limit chunks instead of silently truncating them.

**One database for chat and vector storage.** I first worked with local JSON and NumPy artifacts, then added PostgreSQL and pgvector. The API uses HNSW cosine search for stored vectors. A small corpus does not establish a speed advantage: the current code locally discourages sequential scans to practise using the index. Uploaded vectors are searched separately and merged with saved-vector candidates.

**History has a cost.** Simply appending previous answers can make a retrieval query too long. The current version preserves the latest question and adds recent history within a token budget. Optional rewriting creates a standalone search question, but adds another LLM call. The final generation prompt contains previous questions and fresh evidence, not previous answers.

**Fine-tuning was an experiment.** I tried a small LoRA adapter and compared it with the base model. The experiment did not demonstrate a reliable improvement, so the application uses the base model. The notebook is included as learning work, not evidence of improved model quality.

**Source metadata is not proof of correctness.** The prompt asks for citations, and a format check flags missing or unknown source IDs. That check cannot establish whether a cited passage actually supports a claim. This remains an important limitation.

## Stack

| Part | Tools / model |
|---|---|
| Document processing | Python, pypdf, NumPy |
| Embeddings | Sentence Transformers, `intfloat/multilingual-e5-base` (768 dimensions) |
| Keyword retrieval | `rank-bm25` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Generation | PyTorch, Hugging Face Transformers, `Qwen/Qwen2.5-1.5B-Instruct` |
| API | FastAPI, Pydantic |
| Storage | PostgreSQL, pgvector, HNSW |
| Development | Jupyter, Postman, Docker |

Multilingual embeddings are used, but the complete pipeline has not been evaluated across languages.

## Repository layout

```text
app/process_data.py   Extract, chunk, embed, and save artifacts
app/retrieval.py      Hybrid search, reranking, evidence, and prompts
app/generate.py       Generation and conversation handling
app/database.py       SQL storage and vector search
app/main.py           FastAPI routes
notebook/             Learning experiments; independent from the API modules
Dockerfile           Application container
PROJECT_GUIDE.md     Additional database and API notes
```

## Run locally from a fresh clone

Use Python 3.12 and Docker Desktop. Commands below assume macOS/Linux and run from the project root.

### 1. Install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start PostgreSQL

Choose a local database password. Replace the placeholder consistently in both commands. URL-encode special characters in the connection URL.

```bash
docker run --name handbook-db \
  -e POSTGRES_USER=handbook \
  -e POSTGRES_PASSWORD='CHANGE_ME' \
  -e POSTGRES_DB=handbook \
  -p 127.0.0.1:5433:5432 \
  -v handbook_pgdata:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg17

export DATABASE_URL='postgresql://handbook:CHANGE_ME@localhost:5433/handbook'
```

If this container already exists, use `docker start handbook-db`. The volume persists database records across container replacements. Changing the container's password environment variable does not change an existing database password.

### 3. Prepare your own documents and import them

```bash
mkdir -p data/raw
# Put your own PDF or UTF-8 TXT files in data/raw before continuing.
python -m app.process_data
python -m app.database
```

Preparation creates `data/processed/chunks.json`, `embeddings.npy`, and `config.json`. The second command initializes tables and imports these artifacts into PostgreSQL. It requires permission to create the `vector` extension.

Lecture files and generated artifacts are excluded from Git. Use documents you have permission to process. This chunker keeps a nonempty page as one chunk and rejects pages exceeding the embedding model limit; it does not automatically split long pages.

### 4. Start the API

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000/docs after `Application startup complete`. Initial startup downloads the embedding model, reranker, and generator. The API cannot accept requests until all three finish loading. Ensure sufficient disk space and memory.

## Try the API

1. `POST /register`: JSON `{"email":"student@example.com","password":"YOUR_PASSWORD_AT_LEAST_12_CHARS"}`.
2. `POST /login`: the same JSON; copy `access_token`.
3. Use `Authorization: Bearer YOUR_ACCESS_TOKEN` for protected endpoints.
4. `POST /ask`: multipart form with `question`. Omit `conversation_id` for a new conversation.
5. For a follow-up, pass the returned `conversation_id` and `use_history=true`. Optionally set `use_rewriting=true` for a standalone retrieval question; this adds a generation step.
6. Attach an optional PDF/TXT in the multipart `file` field (application limit: 20 MiB).
7. `GET /conversations` lists your chats; `GET /conversations/{conversation_id}` returns saved turns. Both support `limit` and `offset`.

`POST /conversations` can create an empty conversation explicitly, but it is not required before `/ask`.

The answer response contains `answer`, `sources`, `citation_warnings`, `rejected`, `timings`, `conversation_id`, and `turn_id`. Sources are retrieved evidence, not a guarantee that every answer claim is supported.

## Run the API in Docker on a Mac

Complete the database import above first. The application image does not include PDFs, embeddings, notebooks, or database records.

```bash
docker build -t handbook-api .
```

Create a local `.env` file (excluded from Git and Docker builds):

```dotenv
DATABASE_URL=postgresql://handbook:CHANGE_ME@host.docker.internal:5433/handbook
```

Use your actual password and URL-encode it if needed. The API reads the environment supplied by Docker; it does not load `.env` automatically when run directly in Python.

```bash
docker run --name handbook-api \
  -p 127.0.0.1:8001:8000 \
  --env-file .env \
  -v handbook_model_cache:/home/appuser/.cache/huggingface \
  handbook-api
```

Open http://localhost:8001/docs after startup. The model-cache volume avoids downloading weights again on container replacement. This Linux container uses CPU on a Mac; it cannot use Apple's MPS acceleration. Use one worker because each worker would load its own models.

If the container already exists, `docker start -a handbook-api` reuses its old image and configuration. Rebuilding an image does not update an existing container.

For an older cache volume with a permission error, repair only that volume's ownership:

```bash
docker run --rm --user root \
  -v handbook_model_cache:/home/appuser/.cache/huggingface \
  --entrypoint sh handbook-api \
  -c 'chown -R appuser:appuser /home/appuser/.cache/huggingface'
```

For AWS, supply the RDS connection configuration at runtime, migrate or import the corpus, and set up network access and TLS. AWS deployment is planned, not completed.

## Status and remaining work

- Citation-format validation detects missing or unknown source IDs. It does not validate factual support or require a citation on every claim. The model can still hallucinate.
- The default reranker threshold is not calibrated against a held-out test set.
- History without rewriting is token-budgeted, preserving the current question and prioritizing recent turns. Previous answers may influence retrieval but are not included as generation evidence.
- Temporary uploaded documents are not available for retrieval in later requests unless uploaded again.
- No OCR for scanned PDFs. Long pages may require manual splitting.
- HNSW is approximate. The application locally discourages sequential scans to exercise HNSW on a small corpus; no general speedup claim is made.
- The retriever caches the corpus at startup. Restart after imports. Re-importing does not delete stale chunks from shortened or removed documents.
- Generation requests are serialized within one API process. Rate limiting, request-level timeouts, and public-deployment hardening are not implemented.
- Docker build has been attempted locally, but complete container startup and an end-to-end Docker API test remain unverified after disk-space failure.

The next steps are completing the Docker smoke test, adding a small repeatable evaluation set and API tests, and deploying to AWS. This repository is a portfolio project under development. See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for database and conversation details.
