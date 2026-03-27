from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from inference import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMP,
    DEFAULT_TOP_P,
    InferenceResources,
    generate_response,
    get_model_repo_id,
    load_inference_resources,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS, gt=0)
    temp: float = Field(default=DEFAULT_TEMP, gt=0)
    top_p: float = Field(default=DEFAULT_TOP_P, gt=0, le=1.0)


class ChatResponse(BaseModel):
    response: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load inference resources when the FastAPI app starts."""
    app.state.resources = load_inference_resources(get_model_repo_id())
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the single-page app shell."""
    return FileResponse(INDEX_FILE)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """Run chat generation for the submitted conversation."""
    resources: InferenceResources | None = getattr(request.app.state, "resources", None)
    if resources is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    response = await run_in_threadpool(
        generate_response,
        messages=[message.model_dump() for message in payload.messages],
        resources=resources,
        max_new_tokens=payload.max_new_tokens,
        temp=payload.temp,
        top_p=payload.top_p,
    )
    return ChatResponse(response=response)


def main() -> None:
    """Launch the serving app with Uvicorn."""
    uvicorn.run(app)


if __name__ == "__main__":
    main()
