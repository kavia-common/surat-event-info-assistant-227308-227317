from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ReplyRequest(BaseModel):
    """Incoming email-like query used to generate a reply."""

    sender_email: Optional[str] = Field(
        default=None,
        description="Optional sender email address (used for personalization only).",
        examples=["someone@example.com"],
    )
    subject: Optional[str] = Field(
        default=None,
        description="Optional email subject line.",
        examples=["Any recent events in Surat this week?"],
    )
    body: str = Field(
        ...,
        min_length=1,
        description="Email body containing the user's question/request.",
        examples=["Hi, can you share recent cultural or tech events happening in Surat?"],
    )


class SourceItem(BaseModel):
    """A single citation/source link used by the assistant."""

    title: str = Field(..., description="Title for the cited source.")
    url: str = Field(..., description="URL for the cited source.")


class ReplyResponse(BaseModel):
    """Response payload for an AI-generated email reply."""

    reply_text: str = Field(..., description="AI-generated reply text suitable for email.")
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="List of sources used to generate the reply (2–4 recommended).",
    )
    model: str = Field(..., description="The LLM model identifier used to generate the reply.")
    latency_ms: int = Field(..., ge=0, description="End-to-end latency for the pipeline in milliseconds.")


class ErrorResponse(BaseModel):
    """Structured error response returned by the API."""

    error: str = Field(..., description="Short error type identifier.", examples=["configuration_error"])
    message: str = Field(..., description="Human-friendly error message.")
    request_id: Optional[str] = Field(default=None, description="Optional request correlation id.")
