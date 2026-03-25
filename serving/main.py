from contextlib import asynccontextmanager
import os
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import uvicorn

from serving.inference import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMP,
    DEFAULT_TOP_P,
    InferenceResources,
    generate_response,
    get_model_repo_id,
    load_inference_resources,
)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
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
    app.state.resources = load_inference_resources(get_model_repo_id())
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    resources: InferenceResources | None = getattr(request.app.state, "resources", None)
    if resources is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    response = generate_response(
        messages=[message.model_dump() for message in payload.messages],
        resources=resources,
        max_new_tokens=payload.max_new_tokens,
        temp=payload.temp,
        top_p=payload.top_p,
    )
    return ChatResponse(response=response)


def main() -> None:
    uvicorn.run(
        "serving.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
    )


if __name__ == "__main__":
    main()
