import src.tools as tools


def test_parse_email_headers(sample_gmail_message):
    headers = tools.parse_email_headers(sample_gmail_message)
    assert headers["subject"] == "Invoice #4521"
    assert headers["from_email"] == "vendor@example.com"


def test_get_message_body_text(sample_gmail_message):
    body = tools.get_message_body_text(sample_gmail_message)
    assert body == "Hello world"


def test_list_attachments(sample_gmail_message):
    attachments = tools.list_attachments(sample_gmail_message)
    assert len(attachments) == 1
    assert attachments[0]["filename"] == "invoice.pdf"
    assert attachments[0]["attachment_id"] == "att-456"


def test_save_file_writes_and_dedupes(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "storage_root", str(tmp_path))

    path1 = tools.save_file(b"content-a", "invoices", "file.pdf")
    path2 = tools.save_file(b"content-b", "invoices", "file.pdf")

    assert path1 != path2
    assert "file.pdf" in path1
    assert "file_1.pdf" in path2
