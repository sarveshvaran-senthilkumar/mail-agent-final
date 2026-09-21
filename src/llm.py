"""
Wraps the on-prem Qwen deployment, which exposes an OpenAI-compatible
/v1/chat/completions endpoint — so we reuse the official `openai` SDK
pointed at the internal base_url instead of hand-rolling HTTP calls.

Single-pass pipeline: classify() passes email metadata + extracted document text
to Qwen, which returns doc_type, target_folders, confidence, and reasoning in one call.
"""
import json
import re
from typing import Optional

from openai import OpenAI

from src.config import settings
from src.logger import log_event
from src.models import LLMClassification
from prompts.prompt import CLASSIFICATION_SYSTEM_PROMPT, build_classification_prompt

_client: Optional[OpenAI] = None

# Fallback used when the model output can not be parsed/validated — keeps the
# pipeline moving instead of raising and dropping the whole message.
_FALLBACK = LLMClassification(
    doc_type="unclassified",
    target_folders=["misc"],
    folder="misc",
    confidence=0.0,
    reasoning="LLM response could not be parsed",
)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=settings.qwen_base_url,
            api_key=settings.qwen_api_key,
            timeout=settings.qwen_timeout_seconds,
        )
    return _client


def classify(
    subject: str,
    body: str,
    filename: str,
    extracted_text: str,
) -> LLMClassification:
    """
    Calls the on-prem Qwen model with consolidated email metadata + extracted document text.
    Never raises on malformed model output — falls back to a safe default.
    """
    # Short-circuit: if extraction returned nothing, skip the LLM call entirely
    if not extracted_text.strip():
        log_event(
            "llm_skipped",
            level="warning",
            filename=filename,
            reason="empty_extracted_text",
        )
        return LLMClassification(
            doc_type="other",
            target_folders=["misc"],
            folder="misc",
            confidence=0.0,
            reasoning="No text could be extracted from this file type",
        )

    client = get_client()
    user_prompt = build_classification_prompt(
        subject=subject,
        body=body,
        filename=filename,
        extracted_text=extracted_text,
    )

    try:
        response = client.chat.completions.create(
            model=settings.qwen_model,
            messages=[
                {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        msg = response.choices[0].message
        content = msg.content or getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        if not content.strip():
            log_event("llm_empty_response", level="warning", filename=filename)
            return _FALLBACK
        raw_text = content.strip()
    except Exception as e:
        log_event("llm_call_failed", level="error", filename=filename, error=str(e))
        return _FALLBACK

    parsed = _safe_parse(raw_text)
    if parsed is None:
        log_event("llm_parse_failed", level="warning", raw_text=raw_text[:500])
        return _FALLBACK

    try:
        classification = LLMClassification(**parsed)
    except Exception as e:
        log_event("llm_validation_failed", level="warning", error=str(e), raw=parsed)
        return _FALLBACK

    log_event(
        "llm_classified",
        filename=filename,
        doc_type=classification.doc_type,
        target_folders=classification.target_folders,
        confidence=classification.confidence,
    )
    return classification


def _safe_parse(raw_text: str) -> Optional[dict]:
    """
    Models sometimes wrap JSON in markdown fences, append text, include unescaped newlines,
    or truncate trailing reasoning strings before closing braces.
    Strip fences defensively, auto-repair truncated trailing quotes/braces, and use regex recovery.
    """
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            pass

    # Auto-repair truncated JSON strings (e.g. trailing reasoning string cut off before closing quote/brace)
    for suffix in ['"}', '"}]}', '"]}', '}']:
        try:
            return json.loads(text + suffix, strict=False)
        except json.JSONDecodeError:
            pass

    # Regex recovery: extract doc_type and target_folders even if trailing reasoning is truncated
    doc_type_match = re.search(r'"doc_type"\s*:\s*"([^"]+)"', text)
    target_folders_match = re.search(r'"target_folders"\s*:\s*(\[[^\]]+\])', text, re.DOTALL)
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', text, re.DOTALL)

    if doc_type_match and target_folders_match:
        try:
            doc_type = doc_type_match.group(1)
            target_folders = json.loads(target_folders_match.group(1), strict=False)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.8
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "Partial JSON extracted"
            return {
                "doc_type": doc_type,
                "target_folders": target_folders,
                "confidence": confidence,
                "reasoning": reasoning[:200],
            }
        except Exception:
            pass

    return None
