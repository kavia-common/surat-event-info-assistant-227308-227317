import logging
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings
from .graph import build_reply_graph
from .schemas import ErrorResponse, ReplyRequest, ReplyResponse

settings = get_settings()

# Basic logging setup (uvicorn will also configure loggers; this ensures our modules log consistently).
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

openapi_tags = [
    {"name": "Health", "description": "Service health and diagnostics."},
    {"name": "Replies", "description": "Generate email-style replies about recent Surat events."},
]

app = FastAPI(
    title=settings.app_title,
    description=settings.app_description,
    version=settings.app_version,
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_reply_graph = build_reply_graph()


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "latency_ms": elapsed_ms,
            },
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = request.headers.get("x-request-id")
    # If detail already matches our structured error schema, pass it through.
    if isinstance(exc.detail, dict) and "error" in exc.detail and "message" in exc.detail:
        detail = dict(exc.detail)
        detail.setdefault("request_id", request_id)
        return JSONResponse(status_code=exc.status_code, content=detail)
    payload = ErrorResponse(
        error="http_error",
        message=str(exc.detail),
        request_id=request_id,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("x-request-id")
    logger.exception(
        "unhandled_exception",
        extra={"request_id": request_id, "path": request.url.path, "method": request.method},
    )
    payload = ErrorResponse(
        error="internal_error",
        message="An unexpected error occurred while processing the request.",
        request_id=request_id,
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.get("/", tags=["Health"], summary="Health Check", operation_id="health_check__get")
# PUBLIC_INTERFACE
def health_check() -> Dict[str, Any]:
    """Health check endpoint used by monitoring and deployment verification."""
    return {"message": "Healthy"}


@app.get(
    f"{settings.api_prefix}/docs/websocket",
    tags=["Health"],
    summary="Real-time usage notes (none)",
    operation_id="docs_websocket__get",
)
# PUBLIC_INTERFACE
def websocket_docs() -> Dict[str, str]:
    """This service does not expose WebSocket endpoints. This route exists for documentation parity."""
    return {"note": "No WebSocket endpoints are exposed by this service."}


@app.post(
    f"{settings.api_prefix}/reply",
    response_model=ReplyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    tags=["Replies"],
    summary="Generate an email reply about recent Surat events",
    description=(
        "Accepts an email-like query (sender, subject, body), performs web search for recent Surat events, "
        "and returns an AI-generated concise reply with 2–4 brief citations."
    ),
    operation_id="reply__post",
)
# PUBLIC_INTERFACE
def create_reply(payload: ReplyRequest, request: Request) -> ReplyResponse:
    """
    Generate an email-style reply about recent events in Surat.

    Parameters:
        payload: Email-like request payload containing body (required) and optional subject/sender.
        request: FastAPI request (used for request id correlation).

    Returns:
        ReplyResponse with reply_text, sources, model, and latency_ms.
    """
    request_id = request.headers.get("x-request-id")
    start = time.perf_counter()

    if not payload.body or not payload.body.strip():
        err = ErrorResponse(
            error="validation_error",
            message="Field 'body' must be a non-empty string.",
            request_id=request_id,
        )
        raise HTTPException(status_code=400, detail=err.model_dump())

    try:
        state = {
            "body": payload.body.strip(),
            "subject": payload.subject,
            "sender_email": payload.sender_email,
        }
        result = _reply_graph.invoke(state)
        latency_ms = int((time.perf_counter() - start) * 1000)

        sources = result.get("sources", []) or []
        reply_text = (result.get("reply_text") or "").strip()
        model = result.get("model") or "none"

        if not reply_text:
            reply_text = (
                "Hi,\n\nSorry — I couldn’t generate a reply right now. Please try again with more details.\n\nRegards,\n"
                "Surat Event Info Assistant"
            )

        return ReplyResponse(
            reply_text=reply_text,
            sources=sources[: settings.max_reply_items],
            model=model,
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.exception("reply_pipeline_failed", extra={"request_id": request_id})
        err = ErrorResponse(
            error="pipeline_error",
            message=str(e) if str(e) else "Reply pipeline failed.",
            request_id=request_id,
        )
        raise HTTPException(status_code=500, detail=err.model_dump())
