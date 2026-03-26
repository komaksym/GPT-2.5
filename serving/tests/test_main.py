from fastapi.testclient import TestClient

from serving import main as serving_main
from serving.inference import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMP, DEFAULT_TOP_P


def test_root_serves_app_shell(monkeypatch):
    """Serve the app shell from the root route."""
    monkeypatch.setattr(
        serving_main, "load_inference_resources", lambda repo_id: object()
    )

    with TestClient(serving_main.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "GPT 2.5 Interface" in response.text
    assert (
        "Modern, improved reproduction of the prominent GPT 2 (124M)" in response.text
    )
    assert (
        "https://cdn.tailwindcss.com?plugins=forms,container-queries" in response.text
    )
    assert 'href="/static/app.css"' in response.text
    assert 'src="/static/app.js"' in response.text
    assert 'id="message-input"' in response.text
    assert 'id="send-button"' in response.text
    assert 'id="new-chat-button"' in response.text
    assert 'id="chat-view"' in response.text
    assert 'id="chat-messages"' in response.text
    assert 'data-prompt="Explain quantum computing in simple terms"' in response.text
    assert 'data-prompt="Write a Python script for data cleaning"' in response.text


def test_static_assets_are_served(monkeypatch):
    """Serve the static CSS and JS assets."""
    monkeypatch.setattr(
        serving_main, "load_inference_resources", lambda repo_id: object()
    )

    with TestClient(serving_main.app) as client:
        css_response = client.get("/static/app.css")
        js_response = client.get("/static/app.js")

    assert css_response.status_code == 200
    assert css_response.headers["content-type"].startswith("text/css")

    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]
    assert "You are a helpful assistant developed by Koma Labs." in js_response.text


def test_chat_returns_generated_response(monkeypatch):
    """Return the generated text from the chat endpoint."""
    captured = {}
    resources = object()

    monkeypatch.setattr(
        serving_main, "load_inference_resources", lambda repo_id: resources
    )

    def fake_generate_response(
        *,
        messages,
        resources,
        max_new_tokens,
        temp,
        top_p,
    ):
        """Capture chat generation inputs and return a canned response."""
        captured["messages"] = messages
        captured["resources"] = resources
        captured["max_new_tokens"] = max_new_tokens
        captured["temp"] = temp
        captured["top_p"] = top_p
        return "Hello from GPT 2.5"

    monkeypatch.setattr(serving_main, "generate_response", fake_generate_response)

    with TestClient(serving_main.app) as client:
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Say hello."},
                ]
            },
        )

    assert response.status_code == 200
    assert response.json() == {"response": "Hello from GPT 2.5"}
    assert captured == {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Say hello."},
        ],
        "resources": resources,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "temp": DEFAULT_TEMP,
        "top_p": DEFAULT_TOP_P,
    }


def test_chat_runs_generation_in_threadpool(monkeypatch):
    """Run chat generation inside FastAPI's threadpool helper."""
    resources = object()
    captured = {}

    monkeypatch.setattr(
        serving_main, "load_inference_resources", lambda repo_id: resources
    )

    def fake_generate_response(**kwargs):
        """Provide a callable identity for the threadpool assertion."""
        return "unused"

    async def fake_run_in_threadpool(func, *args, **kwargs):
        """Capture threadpool dispatch details and return a canned response."""
        captured["func"] = func
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "Hello from the threadpool"

    monkeypatch.setattr(serving_main, "generate_response", fake_generate_response)
    monkeypatch.setattr(serving_main, "run_in_threadpool", fake_run_in_threadpool)

    with TestClient(serving_main.app) as client:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 200
    assert response.json() == {"response": "Hello from the threadpool"}
    assert captured == {
        "func": fake_generate_response,
        "args": (),
        "kwargs": {
            "messages": [{"role": "user", "content": "Hello"}],
            "resources": resources,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
            "temp": DEFAULT_TEMP,
            "top_p": DEFAULT_TOP_P,
        },
    }


def test_chat_returns_503_when_model_is_unavailable(monkeypatch):
    """Return a 503 when startup failed to load the model."""
    monkeypatch.setattr(serving_main, "load_inference_resources", lambda repo_id: None)

    with TestClient(serving_main.app) as client:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Model is not loaded."}
