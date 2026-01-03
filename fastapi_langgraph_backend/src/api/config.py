"""
Central configuration for the FastAPI + LangGraph backend.

All settings are loaded from environment variables (optionally via python-dotenv)
to keep secrets out of source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

# Load .env if present (non-fatal if missing)
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_list_csv(name: str, default: List[str]) -> List[str]:
    val = os.getenv(name)
    if val is None:
        return default
    items = [s.strip() for s in val.split(",")]
    return [s for s in items if s]


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_title: str
    app_description: str
    app_version: str

    # API
    api_prefix: str

    # CORS
    cors_allow_origins: List[str]

    # LLM
    openai_api_key: Optional[str]
    openai_model: str
    openai_timeout_s: float

    # Search
    tavily_api_key: Optional[str]
    search_max_results: int

    # Safety / behavior
    max_reply_items: int
    min_reply_items: int

    # Observability
    log_level: str


# PUBLIC_INTERFACE
def get_settings() -> Settings:
    """Return the current application settings (env-configured)."""
    return Settings(
        app_title=os.getenv("APP_TITLE", "Surat Event Info Assistant API"),
        app_description=os.getenv(
            "APP_DESCRIPTION",
            "API that accepts email-like queries about recent Surat events, performs web search, "
            "and returns an AI-generated reply with brief citations.",
        ),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        api_prefix=os.getenv("API_PREFIX", "/api/v1"),
        cors_allow_origins=_get_list_csv("CORS_ALLOW_ORIGINS", ["*"]),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_timeout_s=float(os.getenv("OPENAI_TIMEOUT_S", "30")),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        search_max_results=int(os.getenv("SEARCH_MAX_RESULTS", "6")),
        max_reply_items=int(os.getenv("MAX_REPLY_ITEMS", "4")),
        min_reply_items=int(os.getenv("MIN_REPLY_ITEMS", "2")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
