from unittest.mock import MagicMock

import src.agent as agent


def test_run_processes_message_end_to_end(monkeypatch, mock_mongo_client, sample_gmail_message, mock_llm_classify):
    fake_service = MagicMock()
    monkeypatch.setattr(agent.tools, "get_gmail_service", lambda: fake_service)
    monkeypatch.setattr(agent.tools, "list_new_messages", lambda service, **kwargs: [{"id": "msg-123"}])
    monkeypatch.setattr(agent.tools, "fetch_message_content", lambda service, mid: sample_gmail_message)
    monkeypatch.setattr(agent.tools, "get_attachment_bytes", lambda service, mid, aid: b"%PDF-1.4 fake")
    monkeypatch.setattr(agent.tools, "save_file", lambda content, folder, filename: f"/storage/{folder}/{filename}")

    summary = agent.run()

    assert summary.processed == 1
    assert summary.failed == 0


def test_run_isolates_a_failing_message(monkeypatch, mock_mongo_client, sample_gmail_message, mock_llm_classify):
    """One message's attachment download fails -> run continues, failure is recorded."""
    fake_service = MagicMock()
    monkeypatch.setattr(agent.tools, "get_gmail_service", lambda: fake_service)
    monkeypatch.setattr(
        agent.tools, "list_new_messages", lambda service, **kwargs: [{"id": "msg-bad"}, {"id": "msg-good"}]
    )
    monkeypatch.setattr(agent.tools, "fetch_message_content", lambda service, mid: sample_gmail_message)
    monkeypatch.setattr(agent.tools, "save_file", lambda content, folder, filename: f"/storage/{folder}/{filename}")

    def flaky_get_attachment(service, mid, aid):
        if mid == "msg-bad":
            raise RuntimeError("attachment download failed")
        return b"%PDF-1.4 fake"

    monkeypatch.setattr(agent.tools, "get_attachment_bytes", flaky_get_attachment)

    summary = agent.run()

    assert summary.processed == 1
    assert summary.failed == 1
    assert summary.errors[0]["message_id"] == "msg-bad"


def test_run_skips_already_processed_messages(monkeypatch, mock_mongo_client):
    import src.database as database
    database.mark_processed("msg-123", internal_date_ms=0, status="success")

    fake_service = MagicMock()
    monkeypatch.setattr(agent.tools, "get_gmail_service", lambda: fake_service)
    monkeypatch.setattr(agent.tools, "list_new_messages", lambda service, **kwargs: [{"id": "msg-123"}])

    summary = agent.run()

    assert summary.skipped == 1
    assert summary.processed == 0


def test_run_skips_messages_older_than_start_date(monkeypatch, mock_mongo_client, sample_gmail_message):
    fake_service = MagicMock()
    old_message = dict(sample_gmail_message)
    old_message["id"] = "msg-old"
    old_message["internalDate"] = "100000"  # Far in the past (1970)

    monkeypatch.setattr(agent.tools, "get_gmail_service", lambda: fake_service)
    monkeypatch.setattr(agent.tools, "list_new_messages", lambda service, **kwargs: [{"id": "msg-old"}])
    monkeypatch.setattr(agent.tools, "fetch_message_content", lambda service, mid: old_message)

    summary = agent.run()

    assert summary.skipped == 1
    assert summary.processed == 0


def test_run_aborts_on_auth_failure(monkeypatch):
    def broken_auth():
        raise RuntimeError("bad credentials")

    monkeypatch.setattr(agent.tools, "get_gmail_service", broken_auth)

    try:
        agent.run()
        assert False, "expected run() to raise on auth failure"
    except RuntimeError:
        pass


def test_run_files_resume_to_multiple_target_folders(monkeypatch, mock_mongo_client, sample_gmail_message):
    fake_service = MagicMock()
    resume_message = dict(sample_gmail_message)
    resume_message["id"] = "msg-resume-999"

    monkeypatch.setattr(agent.tools, "get_gmail_service", lambda: fake_service)
    monkeypatch.setattr(agent.tools, "list_new_messages", lambda service, **kwargs: [{"id": "msg-resume-999"}])
    monkeypatch.setattr(agent.tools, "fetch_message_content", lambda service, mid: resume_message)
    monkeypatch.setattr(agent.tools, "get_attachment_bytes", lambda service, mid, aid: b"%PDF-1.4 fake")

    saved_folders = []
    def fake_save_file(content, folder, filename):
        saved_folders.append(folder)
        return f"/storage/{folder}/{filename}"

    monkeypatch.setattr(agent.tools, "save_file", fake_save_file)

    from src.models import LLMClassification
    def fake_resume_classify(subject, body, filename, extracted_text=""):
        return LLMClassification(
            doc_type="resume",
            target_folders=["java-developer/level-3", "ai-ml-engineer/level-1"],
            confidence=0.88,
            reasoning="Senior Java and AI engineer",
        )

    monkeypatch.setattr(agent, "classify", fake_resume_classify)

    summary = agent.run()

    assert summary.processed == 1
    assert summary.failed == 0
    assert saved_folders == ["java-developer/level-3", "ai-ml-engineer/level-1"]

