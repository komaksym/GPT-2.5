from fastapi.testclient import TestClient

try:
    import main as serving_main
    from inference import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMP, DEFAULT_TOP_P
except ModuleNotFoundError:
    from serving import main as serving_main
    from serving.inference import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMP, DEFAULT_TOP_P


def test_root_serves_app_shell(monkeypatch):
    monkeypatch.setattr(
        serving_main, "load_inference_resources", lambda repo_id: object()
    )

    with TestClient(serving_main.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "GPT 2.5" in response.text
    assert "/static/app.css" in response.text
    assert "/static/app.js" in response.text


def test_static_assets_are_served(monkeypatch):
    monkeypatch.setattr(
        serving_main, "load_inference_resources", lambda repo_id: object()
    )

    with TestClient(serving_main.app) as client:
        css_response = client.get("/static/app.css")
        js_response = client.get("/static/app.js")

    assert css_response.status_code == 200
    assert css_response.headers["content-type"].startswith("text/css")
    assert "editorial intelligence" not in css_response.text.lower()

    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]
    assert "sendMessage" in js_response.text


def test_chat_returns_generated_response(monkeypatch):
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


def test_chat_returns_503_when_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(serving_main, "load_inference_resources", lambda repo_id: None)

    with TestClient(serving_main.app) as client:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Model is not loaded."}
