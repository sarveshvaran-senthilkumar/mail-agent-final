"""
Structured logging. Every module calls log_event() at each meaningful step
(auth, fetch, classify, store, insert, error) rather than logging only at the end.
"""
import json
import logging
import sys
from datetime import datetime, timezone

from src.config import settings

_logger = logging.getLogger("mail_agent")
_logger.setLevel(settings.log_level)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_handler)
_logger.propagate = False


def log_event(event: str, level: str = "info", **context) -> None:
    """
    Emit a single structured log line.

    Example:
        log_event("attachment_stored", path="/storage/invoices/foo.pdf", message_id="18c...")
        log_event("message_failed", level="error", message_id="18c...", step="llm_classify", error=str(e))
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **context,
    }
    line = json.dumps(record, default=str)

    log_fn = getattr(_logger, level.lower(), _logger.info)
    log_fn(line)