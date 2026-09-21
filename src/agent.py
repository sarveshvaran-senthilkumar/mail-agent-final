"""
Orchestration.

Part 1 (deterministic, once per run):
    auth -> load timestamp cursor -> list new messages (after cursor).

Part 2 (Single-pass RAG + LLM engine, per message):
    fetch -> get internalDate -> resolve user ->
    download attachment -> save to inbox/ ->
    extract text (RAG) -> classify (LLM returns target_folders) ->
    for each target folder: save copy -> insert metadata -> save reasoning report.

Part 3 (after all messages):
    update timestamp cursor to newest internalDate seen this run.

Error handling wraps each message individually so one bad message never
aborts the run; auth/listing failures are run-level and do abort.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src import database
from src import tools
from src.llm import classify
from src.logger import log_event
from src.models import Document, ResumeAnalysis, StackExperience
from src.utils import safe_filename


@dataclass
class RunSummary:
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[dict] = field(default_factory=list)


def run() -> RunSummary:
    summary = RunSummary()
    log_event("run_started")

    # --- Part 1: auth + timestamp-based message listing ---
    try:
        service = tools.get_gmail_service()
        after_ms = database.get_last_timestamp()
        messages = tools.list_new_messages(service, after_ms=after_ms)
    except Exception as e:
        log_event("run_aborted", level="error", error=str(e))
        raise

    # Track the highest internalDate seen this run so we can advance the cursor
    max_internal_date_ms = after_ms

    # --- Part 2: RAG + LLM-assisted, per message ---
    for message_ref in messages:
        message_id = message_ref["id"]

        if database.is_processed(message_id):
            log_event("message_skipped", message_id=message_id, reason="already_processed")
            summary.skipped += 1
            continue

        try:
            internal_date_ms, processed = _process_message(service, message_id, after_ms=after_ms)
            if not processed:
                summary.skipped += 1
                continue

            database.mark_processed(message_id, internal_date_ms=internal_date_ms, status="success")
            log_event("message_processed", message_id=message_id, internal_date_ms=internal_date_ms)
            summary.processed += 1

            if internal_date_ms > max_internal_date_ms:
                max_internal_date_ms = internal_date_ms

        except Exception as e:
            # Fetch internalDate best-effort so the cursor still advances past failed msgs
            try:
                msg = tools.fetch_message_content(service, message_id)
                ts = tools.get_internal_date_ms(msg)
            except Exception:
                ts = 0
            database.mark_processed(message_id, internal_date_ms=ts, status="failed", error=str(e))
            log_event("message_failed", level="error", message_id=message_id, error=str(e))
            summary.failed += 1
            summary.errors.append({"message_id": message_id, "error": str(e)})
            continue  # never let one bad message abort the run

    # --- Part 3: advance the cursor ---
    if max_internal_date_ms > after_ms:
        database.update_last_timestamp(max_internal_date_ms)

    log_event(
        "run_completed",
        processed=summary.processed,
        failed=summary.failed,
        skipped=summary.skipped,
    )
    return summary


def _process_message(service, message_id: str, after_ms: int = 0) -> tuple[int, bool]:
    """
    Processes one Gmail message through the single-pass inbox-first RAG pipeline.
    Returns (internal_date_ms, processed_boolean).
    """
    message = tools.fetch_message_content(service, message_id)
    internal_date_ms = tools.get_internal_date_ms(message)

    if after_ms > 0 and internal_date_ms < after_ms:
        log_event(
            "message_skipped",
            message_id=message_id,
            reason="older_than_start_cursor",
            internal_date_ms=internal_date_ms,
            after_ms=after_ms,
        )
        return internal_date_ms, False

    headers = tools.parse_email_headers(message)
    subject = headers.get("subject", "")
    body_text = tools.get_message_body_text(message)

    sender_email = headers.get("from_email", "").strip() or "unknown_sender@domain.local"
    user = database.find_or_create_user(sender_email)

    attachments = tools.list_attachments(message)
    if not attachments:
        log_event("message_no_attachments", message_id=message_id)
        return internal_date_ms, True

    for attachment_meta in attachments:
        raw_filename = attachment_meta["filename"]
        filename = safe_filename(raw_filename)

        # Step a: download attachment bytes from Gmail
        content = tools.get_attachment_bytes(service, message_id, attachment_meta["attachment_id"])

        # Step b: save to inbox/ staging folder — original copy, never modified
        inbox_file_path = tools.save_to_inbox(content, filename)
        inbox_filename = Path(inbox_file_path).name

        # Step c: extract text from the saved file (RAG, full context)
        extracted_text = tools.extract_text(inbox_file_path)

        # Step d: single-pass LLM classification & target folder generation
        classification = classify(
            subject=subject,
            body=body_text,
            filename=inbox_filename,
            extracted_text=extracted_text,
        )

        # Step e: file copies to all LLM-selected target folders
        for folder in classification.target_folders:
            stored_path = tools.save_file(content, folder, inbox_filename)

            document = Document(
                user_id=user.id,
                message_id=message_id,
                filename=inbox_filename,
                doc_type=classification.doc_type,
                folder=folder,
                storage_path=stored_path,
                inbox_path=inbox_file_path,
                llm_tag=f"qwen:{classification.confidence:.2f}",
            )
            database.insert_document(document)

        # Step f: if doc_type is "resume", persist sub-classification analysis to resume_analyses collection
        if classification.doc_type.lower() == "resume":
            detected_stacks = []
            for folder in classification.target_folders:
                parts = folder.split("/")
                if len(parts) == 2:
                    role_slug, level_str = parts[0], parts[1]
                elif len(parts) >= 3 and parts[0] == "resumes":
                    role_slug, level_str = parts[1], parts[2]
                else:
                    continue

                detected_stacks.append(
                    StackExperience(
                        stack=role_slug,
                        years=0.0,
                        experience_band=level_str,
                        confidence=classification.confidence,
                        evidence=classification.reasoning or "",
                    )
                )
            analysis = ResumeAnalysis(
                message_id=message_id,
                filename=inbox_filename,
                stacks=detected_stacks,
            )
            database.insert_resume_analysis(analysis)

        # Step g: save ONE single consolidated reasoning report in storage/reason/
        all_folders_str = ", ".join(classification.target_folders)
        email_received_dt = datetime.fromtimestamp(internal_date_ms / 1000, tz=timezone.utc).isoformat() if internal_date_ms else "N/A"
        processed_dt = datetime.now(timezone.utc).isoformat()

        reason_text = (
            f"CLASSIFICATION REASONING REPORT\n"
            f"===============================\n"
            f"Target File: {inbox_filename}\n"
            f"Email Received Date: {email_received_dt} (Epoch MS: {internal_date_ms})\n"
            f"Processed Timestamp: {processed_dt}\n"
            f"Doc Type: {classification.doc_type}\n"
            f"Target Folders ({len(classification.target_folders)}): {all_folders_str}\n"
            f"Confidence Score: {classification.confidence:.2f}\n"
            f"LLM Reasoning: {classification.reasoning}\n\n"
            f"CONSOLIDATED CONTEXT USED:\n"
            f"- Email Subject: {subject or 'N/A'}\n"
            f"- Email Body:\n{body_text or 'N/A'}\n"
        )
        tools.save_reason_file(inbox_filename, reason_text)

    return internal_date_ms, True


if __name__ == "__main__":
    run()
