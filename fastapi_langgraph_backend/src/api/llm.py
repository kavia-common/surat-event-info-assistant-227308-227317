from __future__ import annotations

import logging
from typing import List, Optional

from .config import get_settings
from .schemas import SourceItem

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM cannot be used due to missing configuration."""


def _format_sources_for_prompt(sources: List[SourceItem]) -> str:
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"{i}. {s.title} — {s.url}")
    return "\n".join(lines)


# PUBLIC_INTERFACE
def generate_email_reply(
    *,
    body: str,
    subject: Optional[str],
    sender_email: Optional[str],
    sources: List[SourceItem],
    intent_label: str,
) -> tuple[str, str]:
    """
    Generate a concise, friendly email-style reply given the user body and web sources.

    Returns:
        (reply_text, model_name)
    """
    settings = get_settings()
    if not settings.openai_api_key:
        raise LLMConfigurationError(
            "OPENAI_API_KEY is not set. Configure OPENAI_API_KEY (and optionally OPENAI_MODEL) to enable replies."
        )

    # Lazy import so the app still boots without OpenAI installed (though requirements include it).
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.openai_api_key)

    src_block = _format_sources_for_prompt(sources)

    # Keep a small, strict instruction set to reduce hallucinations and enforce citations.
    system_prompt = (
        "You are a helpful assistant that drafts short email replies about recent events in Surat, India.\n"
        "Rules:\n"
        "- Use ONLY the provided sources for factual claims.\n"
        "- Provide 2–4 concise bullet items (or fewer if not enough sources), each with a clickable link.\n"
        "- Be friendly and professional.\n"
        "- If sources are insufficient or unclear, apologize briefly and suggest what to ask next.\n"
        "- Avoid unsafe content, hate, harassment, or explicit content. If asked, refuse politely.\n"
        "- Do NOT invent dates, venues, or event details that are not in sources.\n"
    )

    user_prompt = (
        f"Sender: {sender_email or 'unknown'}\n"
        f"Subject: {subject or ''}\n"
        f"Detected intent: {intent_label}\n\n"
        f"User email body:\n{body}\n\n"
        f"Sources:\n{src_block if src_block else '(no sources found)'}\n\n"
        "Now draft the reply. Format:\n"
        "Greeting line,\n"
        "2–4 bullet items with links,\n"
        "Closing line.\n"
    )

    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            timeout=settings.openai_timeout_s,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("LLM returned an empty response.")
        return text, settings.openai_model
    except Exception:
        logger.exception("LLM generation failed.")
        raise
