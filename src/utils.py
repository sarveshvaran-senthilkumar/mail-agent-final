"""
Generic helpers only — nothing Gmail- or Mongo-specific here.
Domain-specific parsing (headers, attachments) lives in tools.py.
"""
import functools
import time
from typing import Callable, TypeVar

from src.logger import log_event

T = TypeVar("T")


def retry(times: int = 3, delay_seconds: float = 1.0, backoff: float = 2.0):
    """
    Simple retry decorator with exponential backoff. Use on flaky I/O calls
    (e.g. LLM or Gmail requests) where a transient failure shouldn't kill
    the whole run.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0
            current_delay = delay_seconds
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= times:
                        raise
                    log_event(
                        "retrying",
                        level="warning",
                        function=func.__name__,
                        attempt=attempt,
                        error=str(e),
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator


def safe_filename(name: str) -> str:
    """Strips characters that are awkward on most filesystems."""
    keep = "-_.() "
    return "".join(c for c in name if c.isalnum() or c in keep).strip() or "unnamed"