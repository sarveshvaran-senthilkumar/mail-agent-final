"""
MongoDB access layer.
Owns: find_or_create_user, insert_document, is_processed/mark_processed,
      get_last_timestamp/update_last_timestamp (timestamp cursor for polling).
"""
from typing import Optional

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from src.config import settings
from src.logger import log_event
from src.models import Document, ProcessedMessage, ResumeAnalysis, RunCursor, User

_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.mongo_uri)
    return _client


def get_db() -> Database:
    return get_client()[settings.mongo_db_name]


def users_collection() -> Collection:
    return get_db()["users"]


def documents_collection() -> Collection:
    return get_db()["documents"]


def processed_messages_collection() -> Collection:
    return get_db()["processed_messages"]


def run_cursor_collection() -> Collection:
    return get_db()["run_cursor"]


def resume_analyses_collection() -> Collection:
    return get_db()["resume_analyses"]


# --------------------------------------------------------------------------
# User
# --------------------------------------------------------------------------

def find_or_create_user(email: str) -> User:
    """
    Upsert-by-email. Returns the resulting User with its Mongo _id populated.
    """
    coll = users_collection()
    existing = coll.find_one({"email": email})

    if existing:
        log_event("user_resolved", created=False, email=email)
        existing["_id"] = str(existing["_id"])
        return User(**existing)

    user = User(email=email)
    doc = user.model_dump(by_alias=True, exclude={"id"})
    result = coll.insert_one(doc)

    log_event("user_resolved", created=True, email=email, user_id=str(result.inserted_id))
    user.id = str(result.inserted_id)
    return user


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

def insert_document(document: Document) -> str:
    coll = documents_collection()
    doc = document.model_dump(by_alias=True, exclude={"id"})
    result = coll.insert_one(doc)

    log_event(
        "metadata_inserted",
        document_id=str(result.inserted_id),
        user_id=document.user_id,
        doc_type=document.doc_type,
        folder=document.folder,
    )
    return str(result.inserted_id)


def insert_resume_analysis(analysis: ResumeAnalysis) -> str:
    """
    Persists the full resume sub-classification result (all detected stacks
    with their experience bands) into the resume_analyses collection.
    """
    coll = resume_analyses_collection()
    result = coll.insert_one(analysis.model_dump())
    log_event(
        "resume_analysis_inserted",
        analysis_id=str(result.inserted_id),
        filename=analysis.filename,
        stacks_found=len(analysis.stacks),
    )
    return str(result.inserted_id)


# --------------------------------------------------------------------------
# Idempotency
# --------------------------------------------------------------------------

def is_processed(message_id: str) -> bool:
    """
    Idempotency check — prevents reprocessing the same Gmail message
    if it somehow appears in a listing twice.
    """
    return processed_messages_collection().find_one({"message_id": message_id}) is not None


def mark_processed(
    message_id: str,
    internal_date_ms: int,
    status: str,
    error: Optional[str] = None,
) -> None:
    record = ProcessedMessage(
        message_id=message_id,
        internal_date_ms=internal_date_ms,
        status=status,
        error=error,
    )
    processed_messages_collection().update_one(
        {"message_id": message_id},
        {"$set": record.model_dump()},
        upsert=True,
    )


# --------------------------------------------------------------------------
# Timestamp cursor (replaces is:unread query)
# --------------------------------------------------------------------------

def get_last_timestamp() -> int:
    """
    Returns the internalDate (epoch ms) to use as the lower bound for listing mail.

    Priority:
      1. If a run_cursor exists in MongoDB (from a previous run) → use it.
      2. Else if settings.gmail_start_date is set → parse it and use as floor.
      3. Else → return 0 (fetch all mail, no floor).

    The start-date floor only applies once — after the first successful run the
    cursor is written and gmail_start_date is never consulted again.
    """
    from datetime import datetime, timezone

    doc = run_cursor_collection().find_one({"key": "last_internal_date_ms"})
    if doc is not None:
        value = doc["value"]
        log_event("cursor_loaded", last_internal_date_ms=value)
        return value

    # No cursor yet — check for a configured start date
    start_date_str = settings.gmail_start_date.strip()
    if start_date_str:
        try:
            dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            start_ms = int(dt.timestamp() * 1000)
            log_event("cursor_not_found", detail="using gmail_start_date", start_date=start_date_str, start_ms=start_ms)
            return start_ms
        except ValueError:
            log_event("cursor_start_date_invalid", level="warning", value=start_date_str, detail="falling back to 0")

    log_event("cursor_not_found", detail="no start date configured — fetching all mail")
    return 0


def update_last_timestamp(ms: int) -> None:
    """
    Persists the highest internalDate seen in the current run so the next
    run only fetches mail newer than this point.
    """
    cursor = RunCursor(value=ms)
    run_cursor_collection().update_one(
        {"key": cursor.key},
        {"$set": cursor.model_dump()},
        upsert=True,
    )
    log_event("cursor_updated", last_internal_date_ms=ms)
