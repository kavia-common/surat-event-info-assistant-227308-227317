from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph  # type: ignore

from .config import get_settings
from .llm import LLMConfigurationError, generate_email_reply
from .schemas import SourceItem
from .search import search_recent_surat_events

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    body: str
    subject: Optional[str]
    sender_email: Optional[str]

    intent: str
    sources: List[SourceItem]
    reply_text: str
    model: str
    search_backend: str


def _basic_intent_classifier(text: str) -> str:
    """Lightweight heuristic intent classifier to avoid needing an LLM just for routing."""
    t = text.lower()
    if any(k in t for k in ["festival", "garba", "navratri", "cultural", "music", "concert", "art", "exhibition"]):
        return "cultural_events"
    if any(k in t for k in ["tech", "startup", "hackathon", "conference", "meetup", "workshop"]):
        return "tech_events"
    if any(k in t for k in ["sports", "marathon", "cricket", "football", "tournament"]):
        return "sports_events"
    if any(k in t for k in ["business", "expo", "trade fair", "summit", "industry"]):
        return "business_events"
    return "general_events"


def _contains_unsafe_request(text: str) -> bool:
    """
    Minimal safety gate. We keep this conservative:
    if user requests clearly unsafe/illegal/explicit content, we refuse.
    """
    t = text.lower()
    unsafe_markers = [
        "how to make a bomb",
        "buy drugs",
        "child sexual",
        "explicit sexual",
        "kill",
        "terrorist",
    ]
    return any(m in t for m in unsafe_markers)


def _strip_excess_whitespace(s: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def node_classify_intent(state: GraphState) -> Dict[str, Any]:
    body = state.get("body", "")
    if _contains_unsafe_request(body):
        return {
            "intent": "unsafe_request",
            "reply_text": (
                "Hi,\n\n"
                "Sorry, I can’t help with that request. If you’re looking for information about public events in Surat, "
                "tell me what kind of events you’re interested in (culture, tech, business, sports) and your preferred dates.\n\n"
                "Regards,\nSurat Event Info Assistant"
            ),
            "sources": [],
            "model": "none",
            "search_backend": "none",
        }
    return {"intent": _basic_intent_classifier(body)}


def node_search(state: GraphState) -> Dict[str, Any]:
    intent = state.get("intent", "general_events")
    body = state.get("body", "")

    # Provide an intent-specific query to improve relevance.
    query = {
        "cultural_events": "cultural events in Surat",
        "tech_events": "tech meetups workshops in Surat",
        "sports_events": "sports events tournaments in Surat",
        "business_events": "expos trade fairs business events in Surat",
        "general_events": "recent events in Surat",
        "unsafe_request": "recent events in Surat",
    }.get(intent, "recent events in Surat")

    # Blend user text lightly, but keep query stable.
    query = f"{query}. User request: {body[:280]}"

    sources, backend = search_recent_surat_events(query)
    return {"sources": sources, "search_backend": backend}


def node_synthesize(state: GraphState) -> Dict[str, Any]:
    settings = get_settings()

    # If we already have a refusal reply, keep it.
    if state.get("intent") == "unsafe_request" and state.get("reply_text"):
        return {
            "reply_text": state["reply_text"],
            "model": state.get("model", "none"),
        }

    sources: List[SourceItem] = state.get("sources", []) or []

    # Guardrail: if not enough sources, produce a helpful apology without calling the LLM.
    if len(sources) < 1:
        reply = (
            "Hi,\n\n"
            "I couldn’t find enough reliable recent information about events in Surat right now. "
            "If you share the type of event (culture/tech/business/sports) and a date range (e.g., this weekend), "
            "I can try again.\n\n"
            "Regards,\nSurat Event Info Assistant"
        )
        return {"reply_text": reply, "model": "none"}

    # Limit sources passed to the LLM to reduce token usage and keep output concise.
    sources_for_llm = sources[: max(settings.max_reply_items, 4)]

    try:
        reply_text, model = generate_email_reply(
            body=state.get("body", ""),
            subject=state.get("subject"),
            sender_email=state.get("sender_email"),
            sources=sources_for_llm,
            intent_label=state.get("intent", "general_events"),
        )
        return {"reply_text": _strip_excess_whitespace(reply_text), "model": model}
    except LLMConfigurationError as e:
        # Developer-friendly message, but still safe for users.
        logger.warning("LLM not configured: %s", str(e))
        reply = (
            "Hi,\n\n"
            "I found some sources, but the reply generator is not configured on this server yet. "
            "An administrator needs to set OPENAI_API_KEY.\n\n"
            "Sources I found:\n"
            + "\n".join([f"- {s.title}: {s.url}" for s in sources[:4]])
            + "\n\nRegards,\nSurat Event Info Assistant"
        )
        return {"reply_text": reply, "model": "none"}
    except Exception:
        logger.exception("Synthesis failed.")
        raise


# PUBLIC_INTERFACE
def build_reply_graph():
    """Build and return the LangGraph workflow used by the /reply endpoint."""
    g = StateGraph(GraphState)
    g.add_node("classify_intent", node_classify_intent)
    g.add_node("search", node_search)
    g.add_node("synthesize", node_synthesize)

    g.set_entry_point("classify_intent")

    # If unsafe, skip search and go straight to synthesize (which returns the refusal response).
    def route_after_classify(state: GraphState) -> str:
        if state.get("intent") == "unsafe_request":
            return "synthesize"
        return "search"

    g.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {"search": "search", "synthesize": "synthesize"},
    )
    g.add_edge("search", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()
