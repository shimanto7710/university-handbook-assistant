"""Run from the project root: python -m uvicorn app.main:app --port 8000."""

from contextlib import asynccontextmanager
import math
import os
from uuid import UUID
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from fastapi import Query, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel, EmailStr, Field, SecretStr
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from psycopg.errors import UniqueViolation
from pypdf.errors import PdfReadError

from .database import Database
from .generate import AnswerGenerator
from .retrieval import Retriever


class AccountInput(BaseModel):
    email: EmailStr = Field(max_length=320)
    password: SecretStr = Field(min_length=12, max_length=128)


class ConversationInput(BaseModel):
    title: str = Field(min_length=1, max_length=200)


@asynccontextmanager
async def lifespan(app):
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Set DATABASE_URL before starting the API.")
    app.state.database = Database(database_url)
    app.state.database.initialize()
    # Load once at startup, rather than once per question.
    retriever = Retriever(database=app.state.database)
    app.state.generator = AnswerGenerator(retriever)
    app.state.model_lock = Lock()
    yield
    del app.state.generator


app = FastAPI(title="University Handbook Assistant", lifespan=lifespan)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


bearer = HTTPBearer(auto_error=False)


def current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
):
    user = None
    if credentials is not None:
        user = request.app.state.database.get_user(credentials.credentials)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired login token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@app.post("/register", status_code=201)
def register(account: AccountInput, request: Request):
    email = str(account.email).strip().lower()
    try:
        user = request.app.state.database.register(
            email, account.password.get_secret_value()
        )
    except UniqueViolation as error:
        raise HTTPException(
            status_code=409, detail="Email already registered."
        ) from error
    return {"user": user}


@app.post("/login")
def login(account: AccountInput, request: Request):
    user = request.app.state.database.login(
        str(account.email).strip().lower(),
        account.password.get_secret_value(),
    )
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = request.app.state.database.create_session(user["id"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 86400,
        "user": user,
    }


@app.post("/conversations", status_code=201)
def create_conversation(
    body: ConversationInput,
    request: Request,
    user: dict = Depends(current_user),
):
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Enter a conversation title.")
    return request.app.state.database.create_conversation(user["id"], title)


@app.get("/conversations")
def list_conversations(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: dict = Depends(current_user),
):
    rows = request.app.state.database.list_conversations(user["id"], limit + 1, offset)
    return {
        "conversations": rows[:limit],
        "limit": limit,
        "offset": offset,
        "has_more": len(rows) > limit,
    }


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: UUID,
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: dict = Depends(current_user),
):
    database = request.app.state.database
    conversation = database.get_conversation(conversation_id, user["id"])
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    turns = database.get_turns(conversation_id, user["id"], limit + 1, offset)
    return {
        "conversation": conversation,
        "turns": turns[:limit],
        "limit": limit,
        "offset": offset,
        "has_more": len(turns) > limit,
    }


@app.post("/ask")
def ask(
    request: Request,
    question: str = Form(..., min_length=1, max_length=2000),
    conversation_id: UUID | None = Form(None),
    file: UploadFile | None = File(None),
    min_score: float = Form(0.0),
    top_k: int = Form(5, ge=1, le=10),
    candidate_k: int = Form(30, ge=1, le=100),
    max_new_tokens: int = Form(300, ge=1, le=1024),
    use_history: bool = Form(False),
    use_rewriting: bool = Form(False),
    user: dict = Depends(current_user),
):
    if not question.strip():
        raise HTTPException(status_code=422, detail="Enter a question.")
    if not math.isfinite(min_score):
        raise HTTPException(status_code=422, detail="min_score must be finite.")

    database = request.app.state.database
    if conversation_id is not None and not database.owns_conversation(
        conversation_id, user["id"],
    ):
        raise HTTPException(status_code=404, detail="Conversation not found.")

    try:
        # Uploads exist only for this request and are always cleaned up.
        with TemporaryDirectory(prefix="handbook-upload-") as upload_dir:
            file_path = None
            if file is not None:
                filename = Path((file.filename or "").replace("\\", "/")).name
                if Path(filename).suffix.lower() not in {".pdf", ".txt"}:
                    raise HTTPException(
                        status_code=400, detail="Upload a PDF or TXT file."
                    )
                content = file.file.read(MAX_UPLOAD_BYTES + 1)
                if len(content) > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail="Maximum upload size is 20 MiB."
                    )
                if not content:
                    raise HTTPException(
                        status_code=400, detail="The uploaded file is empty."
                    )
                file_path = Path(upload_dir) / filename
                file_path.write_bytes(content)

            # Avoid concurrent generation on the shared local models.
            with request.app.state.model_lock:
                history = (
                    database.get_history(conversation_id, user["id"])
                    if use_history and conversation_id is not None
                    else []
                )
                response = request.app.state.generator.ask_question(
                    question=question,
                    file_path=file_path,
                    min_score=min_score,
                    top_k=top_k,
                    candidate_k=candidate_k,
                    max_new_tokens=max_new_tokens,
                    use_history=use_history,
                    history=history,
                    use_rewriting=use_rewriting,
                    debug=False,
                )
                if conversation_id is None:
                    title = " ".join(question.split())[:100]
                    conversation = database.create_conversation(user["id"], title)
                    conversation_id = conversation["id"]
                turn_id = database.save_turn(
                    conversation_id,
                    user["id"],
                    response,
                    use_history,
                    use_rewriting,
                )
            response["conversation_id"] = str(conversation_id)
            response["turn_id"] = turn_id
            response["history"] = history
            return response
    except (ValueError, PdfReadError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        if file is not None:
            file.file.close()
