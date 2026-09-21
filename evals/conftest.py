"""
Shared fixtures. Mocks Gmail, Mongo, and the LLM so tests never hit
real external services.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mongomock
import pytest


@pytest.fixture(autouse=True)
def use_temp_storage(tmp_path, monkeypatch):
    """Redirects storage_root, inbox_folder, and reason_folder to a temporary directory during tests."""
    from src.config import settings
    monkeypatch.setattr(settings, "storage_root", str(tmp_path))
    monkeypatch.setattr(settings, "inbox_folder", str(tmp_path / "inbox"))
    monkeypatch.setattr(settings, "reason_folder", str(tmp_path / "reason"))


@pytest.fixture(autouse=True)
def mock_mongo_client(monkeypatch):
    import src.database as database
    client = mongomock.MongoClient()
    monkeypatch.setattr(database, "_client", client)
    monkeypatch.setattr(database, "get_client", lambda: client)
    return client


@pytest.fixture
def sample_gmail_message():
    """A minimal Gmail API message resource with one PDF attachment."""
    return {
        "id": "msg-123",
        "threadId": "thread-123",
        "internalDate": "1800000000000",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Invoice #4521"},
                {"name": "From", "value": "Vendor <vendor@example.com>"},
            ],
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": "SGVsbG8gd29ybGQ="},  # "Hello world"
                },
                {
                    "filename": "invoice.pdf",
                    "mimeType": "application/pdf",
                    "body": {"attachmentId": "att-456"},
                },
            ],
        },
    }


@pytest.fixture
def mock_llm_classify(monkeypatch):
    import src.agent as agent
    import src.llm as llm
    from src.models import LLMClassification

    def _fake_classify(subject, body, filename, extracted_text=""):
        return LLMClassification(doc_type="invoice", target_folders=["invoices"], folder="invoices", confidence=0.92)

    monkeypatch.setattr(llm, "classify", _fake_classify)
    monkeypatch.setattr(agent, "classify", _fake_classify)
    return _fake_classify

