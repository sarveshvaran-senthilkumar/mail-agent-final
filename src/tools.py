"""
Gmail I/O (OAuth2, list, fetch, attachment download) + local disk storage
+ RAG text extraction from saved files.

Auth flow:
  - First run: opens a browser consent screen (InstalledAppFlow), then writes
    settings.gmail_token_file so subsequent runs are silent (refresh token).
  - Requires a Google Cloud OAuth client (Desktop app) credentials.json,
    see settings.gmail_client_secret_file.

Polling strategy:
  - list_new_messages() accepts after_ms (epoch milliseconds from the run cursor).
  - Converts to epoch seconds for Gmail's after: query operator.
  - On first run after_ms=0 so all mail with attachments is fetched.
"""
import base64
from datetime import datetime, timezone
from email.utils import parseaddr
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from src.config import settings
from src.logger import log_event


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def get_gmail_service():
    """
    Returns an authenticated Gmail API service object.
    Reuses the cached token if present and valid; otherwise runs the
    interactive OAuth2 consent flow once and caches the result.
    """
    creds: Optional[Credentials] = None
    token_path = Path(settings.gmail_token_file)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), settings.gmail_scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                log_event("auth_refreshed")
            except Exception as e:
                log_event("token_refresh_failed", level="warning", error=str(e))
                creds = None

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(
                settings.gmail_client_secret_file, settings.gmail_scopes
            )
            creds = flow.run_local_server(port=0)
            log_event("auth_consent_completed")

        token_path.write_text(creds.to_json())

    try:
        service = build("gmail", "v1", credentials=creds)
        log_event("auth_success")
        return service
    except Exception as e:
        log_event("auth_failed", level="error", error=str(e))
        raise


# --------------------------------------------------------------------------
# List / fetch
# --------------------------------------------------------------------------

def list_new_messages(service, after_ms: int = 0) -> list[dict]:
    """
    Returns messages with attachments that arrived on or after after_ms (epoch ms).
    after_ms=0 on the first run fetches all mail with attachments.
    Gmail's after: operator requires YYYY/MM/DD date format.
    """
    if after_ms > 0:
        dt = datetime.fromtimestamp(after_ms / 1000, tz=timezone.utc)
        date_str = dt.strftime("%Y/%m/%d")
        query = f"has:attachment after:{date_str}"
    else:
        query = "has:attachment"

    response = service.users().messages().list(userId="me", q=query).execute()
    messages = response.get("messages", [])
    log_event("messages_found", count=len(messages), query=query, after_ms=after_ms)
    return messages


def fetch_message_content(service, message_id: str) -> dict:
    """
    Returns the raw Gmail API message resource (format="full").
    internalDate, headers, and attachment metadata all live in here.
    """
    message = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    log_event("message_fetched", message_id=message_id)
    return message


def get_internal_date_ms(message: dict) -> int:
    """
    Extracts Gmail internalDate as an integer epoch milliseconds value.
    This is the millisecond-precision timestamp of when Gmail received the message.
    """
    return int(message.get("internalDate", 0))


def parse_email_headers(message: dict) -> dict:
    """
    Extracts subject / from / to from a Gmail message resource.
    Used only for user resolution (from_email) — not passed to the LLM.
    """
    headers = message.get("payload", {}).get("headers", [])
    lookup = {h["name"].lower(): h["value"] for h in headers}

    sender_name, sender_email = parseaddr(lookup.get("from", ""))

    return {
        "subject": lookup.get("subject", ""),
        "from_email": sender_email,
        "from_name": sender_name,
        "to": lookup.get("to", ""),
    }


def get_message_body_text(message: dict) -> str:
    """
    Best-effort plain-text body extraction (walks multipart payloads).
    Consolidated into the user prompt alongside attachment text for LLM classification.
    """
    def _walk(part) -> str:
        mime_type = part.get("mimeType", "")
        body = part.get("body", {})

        if mime_type == "text/plain" and "data" in body:
            return decode_attachment(body["data"]).decode("utf-8", errors="replace")

        for sub_part in part.get("parts", []) or []:
            text = _walk(sub_part)
            if text:
                return text
        return ""

    return _walk(message.get("payload", {}))


def list_attachments(message: dict) -> list[dict]:
    """
    Returns [{"filename": ..., "attachment_id": ..., "mime_type": ...}, ...]
    for every part that carries an attachmentId.
    """
    found = []

    def _walk(part):
        body = part.get("body", {})
        if body.get("attachmentId") and part.get("filename"):
            found.append({
                "filename": part["filename"],
                "attachment_id": body["attachmentId"],
                "mime_type": part.get("mimeType", "application/octet-stream"),
            })
        for sub_part in part.get("parts", []) or []:
            _walk(sub_part)

    _walk(message.get("payload", {}))
    return found


def get_attachment_bytes(service, message_id: str, attachment_id: str) -> bytes:
    attachment = service.users().messages().attachments().get(
        userId="me", messageId=message_id, id=attachment_id
    ).execute()
    return decode_attachment(attachment["data"])


def decode_attachment(data_b64url: str) -> bytes:
    return base64.urlsafe_b64decode(data_b64url.encode("utf-8"))


# --------------------------------------------------------------------------
# Storage — inbox (staging) and category (final)
# --------------------------------------------------------------------------

def save_to_inbox(content: bytes, filename: str) -> str:
    """
    Saves raw attachment bytes to the inbox/ staging folder before any
    processing. The file stays here permanently as the original copy.
    Returns the absolute path as a string.
    """
    target_dir = settings.inbox_path
    path = target_dir / filename
    stem, suffix = path.stem, path.suffix
    counter = 1
    while path.exists():
        path = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    path.write_bytes(content)
    log_event("attachment_staged", path=str(path.resolve()), filename=filename)
    return str(path.resolve())


def save_file(content: bytes, folder: str, filename: str) -> str:
    """
    Saves attachment bytes under settings.storage_root/<folder>/<filename>.
    Returns the absolute path as a string. Deconflicts filename collisions
    by appending a numeric suffix rather than overwriting.
    """
    target_dir = settings.storage_root_path / folder
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / filename
    stem, suffix = path.stem, path.suffix
    counter = 1
    while path.exists():
        path = target_dir / f"{stem}_{counter}{suffix}"
        counter += 1

    path.write_bytes(content)
    log_event("attachment_stored", path=str(path.resolve()), folder=folder, filename=filename)
    return str(path.resolve())


def save_reason_file(inbox_filename: str, reasoning_text: str) -> str:
    """
    Saves a single consolidated .txt reasoning report into settings.reason_path (storage/reason/).
    Uses the exact inbox filename stem (e.g. storage/reason/alex_cv_1_reason.txt).
    """
    target_dir = settings.reason_path
    stem = Path(inbox_filename).stem
    reason_filename = f"{stem}_reason.txt"
    path = target_dir / reason_filename

    counter = 1
    while path.exists():
        path = target_dir / f"{stem}_reason_{counter}.txt"
        counter += 1

    path.write_text(reasoning_text, encoding="utf-8")
    log_event("reason_file_saved", path=str(path.resolve()), folder="reason", filename=reason_filename)
    return str(path.resolve())


# --------------------------------------------------------------------------
# RAG — text extraction from saved files
# --------------------------------------------------------------------------

def extract_text(file_path: str) -> str:
    """
    Extracts plain text from a locally saved attachment for LLM classification.
    Supports: .pdf (pdfplumber), .docx (python-docx), .txt (plain read).
    All other types (images, zip, xlsx, etc.) return "" which triggers the
    misc/ fallback in the agent.
    Extracts up to settings.resume_extracted_text_max_chars (15,000 chars) for full context.
    """
    return _extract_text_with_limit(file_path, settings.resume_extracted_text_max_chars)


def extract_resume_text(file_path: str) -> str:
    """Backward-compatible alias for extract_text."""
    return extract_text(file_path)


def _extract_text_with_limit(file_path: str, max_chars: int) -> str:
    """Internal: extract text from a file and cap at max_chars."""
    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext == ".pdf":
            text = _extract_pdf(path)
        elif ext == ".docx":
            text = _extract_docx(path)
        elif ext == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            log_event(
                "extraction_skipped",
                level="warning",
                filename=path.name,
                ext=ext,
                reason="unsupported_type",
            )
            return ""
    except Exception as e:
        log_event("extraction_failed", level="warning", filename=path.name, error=str(e))
        return ""

    truncated = text[:max_chars]
    if not text.strip():
        log_event(
            "text_extraction_empty",
            level="warning",
            filename=path.name,
            ext=ext,
            detail="0 characters extracted from file — if PDF, file may be a scanned image without vector text layer",
        )
    else:
        log_event(
            "text_extracted",
            filename=path.name,
            chars_extracted=len(text),
            chars_sent=len(truncated),
            limit=max_chars,
        )
    return truncated


def _extract_pdf(path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


def _extract_docx(path: Path) -> str:
    from docx import Document as DocxDocument
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
