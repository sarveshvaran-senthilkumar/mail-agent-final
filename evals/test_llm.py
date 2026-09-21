from unittest.mock import MagicMock

import src.llm as llm


def _mock_openai_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


def test_classify_happy_path(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_openai_response(
        '{"doc_type": "invoice", "target_folders": ["invoices"], "confidence": 0.85, "reasoning": "looks like an invoice"}'
    )
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    result = llm.classify("Invoice #123", "Please find attached", "invoice.pdf", "INVOICE #123 total $500")

    assert result.doc_type == "invoice"
    assert result.target_folders == ["invoices"]
    assert result.confidence == 0.85


def test_classify_falls_back_on_malformed_json(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _mock_openai_response("not json at all")
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    result = llm.classify("subj", "body", "file.pdf", "some text")

    assert result.doc_type == "unclassified"
    assert result.target_folders == ["misc"]


def test_classify_falls_back_on_api_error(monkeypatch):
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = Exception("connection refused")
    monkeypatch.setattr(llm, "get_client", lambda: fake_client)

    result = llm.classify("subj", "body", "file.pdf", "some text")

    assert result.doc_type == "unclassified"


def test_safe_parse_strips_markdown_fences():
    parsed = llm._safe_parse('```json\n{"doc_type": "x", "folder": "y", "confidence": 0.5}\n```')
    assert parsed == {"doc_type": "x", "folder": "y", "confidence": 0.5}


def test_safe_parse_repairs_truncated_reasoning_string():
    truncated_text = (
        '{\n'
        '  "doc_type": "resume",\n'
        '  "target_folders": [\n'
        '    "resumes/data-engineering/level-5",\n'
        '    "resumes/ai/level-5"\n'
        '  ],\n'
        '  "confidence": 0.95,\n'
        '  "reasoning": "The document is a resume for Vishu Kalier. Active in Eternal 12 month'
    )
    parsed = llm._safe_parse(truncated_text)
    assert parsed is not None
    assert parsed["doc_type"] == "resume"
    assert parsed["target_folders"] == ["resumes/data-engineering/level-5", "resumes/ai/level-5"]
    assert parsed["confidence"] == 0.95
