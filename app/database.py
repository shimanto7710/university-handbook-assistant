"""Small PostgreSQL helper for accounts, conversations, and chat turns."""

import hashlib
import hmac
import secrets
import json
import os
from pathlib import Path

import numpy as np
from pgvector.psycopg import register_vector

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        bytes.fromhex(salt),
        600000,
    ).hex()
    return f"pbkdf2_sha256$600000${salt}${digest}"


def verify_password(password, stored_hash):
    algorithm, iterations, salt, expected = stored_hash.split("$")
    if algorithm != "pbkdf2_sha256":
        return False
    actual = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        bytes.fromhex(salt),
        int(iterations),
    ).hex()
    return hmac.compare_digest(actual, expected)


class Database:
    def __init__(self, database_url):
        self.database_url = database_url

    def connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def initialize(self):
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            # Existing practice conversations remain unowned and inaccessible via the API.
            connection.execute("""
                ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id UUID
                    REFERENCES users(id) ON DELETE CASCADE
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user
                ON conversations(user_id)
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 day')
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    question TEXT NOT NULL,
                    search_question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
                    rejected BOOLEAN NOT NULL DEFAULT FALSE,
                    citation_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    timings JSONB NOT NULL DEFAULT '{}'::jsonb,
                    use_history BOOLEAN NOT NULL DEFAULT FALSE,
                    use_rewriting BOOLEAN NOT NULL DEFAULT FALSE,
                    model_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chat_turns_conversation_id
                ON chat_turns(conversation_id, id DESC)
            """)

        self.initialize_vectors()

    def initialize_vectors(self):
        with self.connect() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS embedding_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    embedding_model TEXT NOT NULL,
                    normalized BOOLEAN NOT NULL,
                    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension = 768)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                    filename TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    page INTEGER,
                    text TEXT NOT NULL,
                    source_order INTEGER NOT NULL,
                    embedding VECTOR(768) NOT NULL
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                ON chunks USING hnsw (embedding vector_cosine_ops)
            """)

    def import_saved_vectors(self, processed_dir):
        processed_dir = Path(processed_dir)
        chunks = json.loads((processed_dir / "chunks.json").read_text(encoding="utf-8"))
        embeddings = np.load(processed_dir / "embeddings.npy", allow_pickle=False)
        config = json.loads((processed_dir / "config.json").read_text(encoding="utf-8"))
        if not chunks or embeddings.shape != (len(chunks), 768):
            raise ValueError(
                "Expected nonempty chunks and matching 768-dimensional embeddings."
            )
        if config["embedding_dimension"] != 768 or config["chunk_count"] != len(chunks):
            raise ValueError("Saved configuration does not match the embeddings.")
        if (
            not np.isfinite(embeddings).all()
            or (np.linalg.norm(embeddings, axis=1) == 0).any()
        ):
            raise ValueError("Embeddings must be finite, nonzero vectors.")
        if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
            raise ValueError("Duplicate chunk IDs in the saved data.")
        if config["normalized"] and not np.allclose(
            np.linalg.norm(embeddings, axis=1), 1, atol=1e-4
        ):
            raise ValueError("Embeddings were expected to be normalized.")

        # One transaction: a failed import leaves the existing corpus unchanged.
        with self.connect() as connection:
            register_vector(connection)
            existing = connection.execute(
                "SELECT * FROM embedding_settings WHERE id = 1"
            ).fetchone()
            if existing and (
                existing["embedding_model"] != config["embedding_model"]
                or existing["normalized"] != config["normalized"]
            ):
                raise ValueError(
                    "Do not mix embedding models or normalization settings."
                )
            connection.execute(
                """
                INSERT INTO embedding_settings (id, embedding_model, normalized, embedding_dimension)
                VALUES (1, %s, %s, 768) ON CONFLICT (id) DO NOTHING
            """,
                (config["embedding_model"], config["normalized"]),
            )
            document_ids = {}
            for filename in dict.fromkeys(chunk["filename"] for chunk in chunks):
                row = connection.execute(
                    """
                    INSERT INTO documents (filename) VALUES (%s)
                    ON CONFLICT (filename) DO UPDATE SET filename = EXCLUDED.filename
                    RETURNING id
                """,
                    (filename,),
                ).fetchone()
                document_ids[filename] = row["id"]
            for position, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                connection.execute(
                    """
                    INSERT INTO chunks (chunk_id, document_id, page, text, source_order, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id, page = EXCLUDED.page,
                        text = EXCLUDED.text, source_order = EXCLUDED.source_order,
                        embedding = EXCLUDED.embedding
                """,
                    (
                        chunk["chunk_id"],
                        document_ids[chunk["filename"]],
                        chunk["page"],
                        chunk["text"],
                        position,
                        np.asarray(embedding, dtype=np.float32),
                    ),
                )
        return {"documents": len(document_ids), "chunks": len(chunks)}

    def load_vector_data(self):
        with self.connect() as connection:
            register_vector(connection)
            config = connection.execute(
                "SELECT * FROM embedding_settings WHERE id = 1"
            ).fetchone()
            rows = connection.execute("""
                SELECT c.chunk_id, d.filename, c.page, c.text, c.embedding
                FROM chunks c JOIN documents d ON d.id = c.document_id
                ORDER BY c.source_order, c.chunk_id
            """).fetchall()
        if not config or not rows:
            raise ValueError("Import saved embeddings first: python -m app.database")
        chunks = [
            {
                "chunk_id": row["chunk_id"],
                "filename": row["filename"],
                "page": row["page"],
                "text": row["text"],
            }
            for row in rows
        ]
        embeddings = np.asarray(
            [row["embedding"].to_list() for row in rows], dtype=np.float32
        )
        config.pop("id")
        config["chunk_count"] = len(chunks)
        return chunks, embeddings, config

    def search_vectors(
        self, query_embedding, limit, extra_chunks=None, extra_embeddings=None
    ):
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if (
            query_embedding.shape != (768,)
            or not np.isfinite(query_embedding).all()
            or np.linalg.norm(query_embedding) == 0
        ):
            raise ValueError("Query embedding must have 768 finite values.")
        if limit < 1:
            raise ValueError("Use a positive search limit.")
        if extra_chunks and np.asarray(extra_embeddings).shape != (
            len(extra_chunks),
            768,
        ):
            raise ValueError("Upload chunks and embeddings do not match.")
        with self.connect() as connection:
            register_vector(connection)
            # Give HNSW enough candidates for the requested result count.
            connection.execute(
                "SELECT set_config('hnsw.ef_search', %s, true)",
                (str(max(40, limit)),),
            )
            # Prefer HNSW even for this small corpus; this setting is transaction-local.
            previous_seqscan = connection.execute("SHOW enable_seqscan").fetchone()["enable_seqscan"]
            connection.execute("SET LOCAL enable_seqscan = off")
            saved_matches = connection.execute(
                """
                SELECT chunk_id, 1 - (embedding <=> %s) AS semantic_score
                FROM chunks
                ORDER BY embedding <=> %s LIMIT %s
                """,
                (query_embedding, query_embedding, limit),
            ).fetchall()
            connection.execute(
                "SELECT set_config('enable_seqscan', %s, true)", (previous_seqscan,)
            )
            if not extra_chunks:
                return saved_matches

            # Uploaded vectors exist only for this request.
            connection.execute("""
                CREATE TEMP TABLE upload_vectors (
                    chunk_id TEXT PRIMARY KEY, source_order INTEGER, embedding VECTOR(768)
                ) ON COMMIT DROP
            """)
            for position, (chunk, embedding) in enumerate(
                zip(extra_chunks, extra_embeddings)
            ):
                connection.execute(
                    "INSERT INTO upload_vectors VALUES (%s, %s, %s)",
                    (chunk["chunk_id"], position, np.asarray(embedding, dtype=np.float32)),
                )
            upload_matches = connection.execute(
                """
                SELECT chunk_id, 1 - (embedding <=> %s) AS semantic_score
                FROM upload_vectors
                ORDER BY embedding <=> %s, source_order, chunk_id LIMIT %s
                """,
                (query_embedding, query_embedding, limit),
            ).fetchall()

            # Both searches use cosine similarity, so their scores are comparable.
            matches = saved_matches + upload_matches
            matches.sort(key=lambda match: match["semantic_score"], reverse=True)
            return matches[:limit]

    def register(self, email, password):
        password_hash = hash_password(password)
        with self.connect() as connection:
            return connection.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id, email",
                (email, password_hash),
            ).fetchone()

    def login(self, email, password):
        with self.connect() as connection:
            user = connection.execute(
                "SELECT id, email, password_hash FROM users WHERE email = %s",
                (email,),
            ).fetchone()
        if user is None:
            # Do comparable password work even if the email is not registered.
            hash_password(password)
            return None
        if not verify_password(password, user["password_hash"]):
            return None
        return {"id": user["id"], "email": user["email"]}

    def create_session(self, user_id):
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO sessions (token_hash, user_id) VALUES (%s, %s)",
                (token_hash, user_id),
            )
        return token

    def get_user(self, token):
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT users.id, users.email FROM users
                JOIN sessions ON sessions.user_id = users.id
                WHERE sessions.token_hash = %s AND sessions.expires_at > NOW()
            """,
                (token_hash,),
            ).fetchone()

    def create_conversation(self, user_id, title):
        with self.connect() as connection:
            return connection.execute(
                """
                INSERT INTO conversations (user_id, title) VALUES (%s, %s)
                RETURNING id, title, created_at
            """,
                (user_id, title),
            ).fetchone()

    def list_conversations(self, user_id, limit=50, offset=0):
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT c.id, c.title, c.created_at,
                       COALESCE(MAX(t.created_at), c.created_at) AS updated_at,
                       COUNT(t.id)::int AS turn_count
                FROM conversations c
                LEFT JOIN chat_turns t ON t.conversation_id = c.id
                WHERE c.user_id = %s
                GROUP BY c.id
                ORDER BY updated_at DESC, c.id DESC
                LIMIT %s OFFSET %s
            """,
                (user_id, limit, offset),
            ).fetchall()

    def get_conversation(self, conversation_id, user_id):
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT id, title, created_at FROM conversations
                WHERE id = %s AND user_id = %s
            """,
                (conversation_id, user_id),
            ).fetchone()

    def get_turns(self, conversation_id, user_id, limit=50, offset=0):
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT t.id, t.question, t.search_question, t.answer,
                       t.sources, t.citation_warnings, t.rejected,
                       t.timings, t.use_history, t.use_rewriting,
                       t.model_name, t.created_at
                FROM chat_turns t
                JOIN conversations c ON c.id = t.conversation_id
                WHERE c.id = %s AND c.user_id = %s
                ORDER BY t.id ASC
                LIMIT %s OFFSET %s
            """,
                (conversation_id, user_id, limit, offset),
            ).fetchall()

    def owns_conversation(self, conversation_id, user_id):
        with self.connect() as connection:
            return (
                connection.execute(
                    "SELECT id FROM conversations WHERE id = %s AND user_id = %s",
                    (conversation_id, user_id),
                ).fetchone()
                is not None
            )

    def get_history(self, conversation_id, user_id):
        with self.connect() as connection:
            turns = connection.execute(
                """
                SELECT t.question, t.answer FROM chat_turns t
                JOIN conversations c ON c.id = t.conversation_id
                WHERE c.id = %s AND c.user_id = %s
                ORDER BY t.id DESC LIMIT 3
            """,
                (conversation_id, user_id),
            ).fetchall()
        return list(reversed(turns))

    def save_turn(self, conversation_id, user_id, response, use_history, use_rewriting):
        with self.connect() as connection:
            row = connection.execute(
                """
                INSERT INTO chat_turns (
                    conversation_id, question, search_question, answer, sources,
                    rejected, citation_warnings, timings, use_history, use_rewriting, model_name
                )
                SELECT c.id, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                FROM conversations c WHERE c.id = %s AND c.user_id = %s
                RETURNING id
            """,
                (
                    response["question"],
                    response["search_question"],
                    response["answer"],
                    Jsonb(response["sources"]),
                    response["rejected"],
                    Jsonb(response["citation_warnings"]),
                    Jsonb(response["timings"]),
                    use_history,
                    use_rewriting,
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    conversation_id,
                    user_id,
                ),
            ).fetchone()
        if row is None:
            raise ValueError("Conversation not found.")
        return row["id"]


if __name__ == "__main__":
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL first.")
    database = Database(database_url)
    database.initialize()
    processed_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    print("Imported into pgvector:", database.import_saved_vectors(processed_dir))
