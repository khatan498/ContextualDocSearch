"""Tests for app.api.body_limit.

The middleware sits in front of a tiny echo app, so the tests can see exactly
what reaches the app — the body replay has to be byte-for-byte faithful.
"""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.body_limit import MAX_BODY_BYTES, BodySizeLimitMiddleware

LIMIT = 100


def make_client() -> tuple[TestClient, list[bytes]]:
    """An echo app behind the middleware, and a record of what reached it."""
    received: list[bytes] = []
    app = FastAPI()
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=LIMIT)

    @app.post("/echo")
    async def echo(request: Request) -> dict:
        body = await request.body()
        received.append(body)
        return {"length": len(body)}

    @app.get("/ping")
    def ping() -> dict:
        return {"ok": True}

    return TestClient(app), received


def chunks(total: int, size: int = 10):
    """Yield `total` bytes in pieces. httpx sends a generator body chunked,
    with no Content-Length — the case a header check alone would miss."""
    for start in range(0, total, size):
        yield b"x" * min(size, total - start)


class TestWithinTheLimit:
    @pytest.mark.parametrize("size", [0, 1, LIMIT])
    def test_body_reaches_the_app_intact(self, size: int) -> None:
        client, received = make_client()
        body = bytes(range(256))[:size] if size <= 256 else b"x" * size

        response = client.post("/echo", content=body)

        assert response.status_code == 200
        assert received == [body]

    def test_chunked_body_is_reassembled(self) -> None:
        client, received = make_client()

        response = client.post("/echo", content=chunks(LIMIT))

        assert response.status_code == 200
        assert received == [b"x" * LIMIT]

    def test_requests_without_a_body_are_untouched(self) -> None:
        client, _ = make_client()

        assert client.get("/ping").json() == {"ok": True}


class TestOverTheLimit:
    def test_declared_length_over_the_limit_is_413(self) -> None:
        client, received = make_client()

        response = client.post("/echo", content=b"x" * (LIMIT + 1))

        assert response.status_code == 413
        assert str(LIMIT) in response.json()["detail"]
        assert received == []

    def test_chunked_body_over_the_limit_is_413(self) -> None:
        # No Content-Length: the middleware must count as the body arrives.
        client, received = make_client()

        response = client.post("/echo", content=chunks(LIMIT * 5))

        assert response.status_code == 413
        assert received == []


class TestDefaultLimit:
    def test_the_largest_valid_search_fits(self) -> None:
        # 500 characters, each an astral-plane character written as a JSON
        # surrogate-pair escape: 12 bytes each, the worst case for a valid query.
        import json

        body = json.dumps({"query": "\U0001F600" * 500, "top_k": 100})

        assert len(body.encode()) < MAX_BODY_BYTES


class TestAtTheProtocolLevel:
    """Drives the middleware with scripted ASGI messages.

    TestClient hands the app even a chunked upload as a single message, so
    reassembling a body that really arrives in pieces — and refusing one
    without reading it — can only be checked by calling the middleware
    directly. `asyncio.run` runs one coroutine to completion, which is all a
    test needs from an event loop.
    """

    @staticmethod
    def run(messages: list[dict], headers: list[tuple[bytes, bytes]] | None = None):
        """Send `messages` through the middleware; report what happened."""
        import asyncio

        pending = list(messages)
        receive_calls = 0
        sent: list[dict] = []
        app_received: list[bytes] = []

        async def receive() -> dict:
            nonlocal receive_calls
            receive_calls += 1
            return pending.pop(0) if pending else {"type": "http.disconnect"}

        async def send(message: dict) -> None:
            sent.append(message)

        async def app(scope, app_receive, app_send) -> None:
            body = b""
            while True:
                message = await app_receive()
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
            app_received.append(body)
            await app_send({"type": "http.response.start", "status": 200, "headers": []})
            await app_send({"type": "http.response.body", "body": b"ok"})

        scope = {"type": "http", "method": "POST", "path": "/", "headers": headers or []}
        asyncio.run(BodySizeLimitMiddleware(app, max_bytes=LIMIT)(scope, receive, send))
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        return status, app_received, receive_calls

    def test_a_body_in_several_messages_is_reassembled_in_order(self) -> None:
        messages = [
            {"type": "http.request", "body": b"first-", "more_body": True},
            {"type": "http.request", "body": b"second-", "more_body": True},
            {"type": "http.request", "body": b"third", "more_body": False},
        ]

        status, app_received, _ = self.run(messages)

        assert status == 200
        assert app_received == [b"first-second-third"]

    def test_over_the_limit_across_messages_is_refused(self) -> None:
        # Each piece is within the limit; only the running total is not.
        piece = {"type": "http.request", "body": b"x" * 40, "more_body": True}
        last = {"type": "http.request", "body": b"x" * 40, "more_body": False}

        status, app_received, _ = self.run([piece, piece, last])

        assert status == 413
        assert app_received == []

    def test_a_declared_length_over_the_limit_is_refused_unread(self) -> None:
        # The header alone decides: not one message of the body is read.
        status, app_received, receive_calls = self.run(
            [{"type": "http.request", "body": b"x", "more_body": False}],
            headers=[(b"content-length", str(LIMIT + 1).encode())],
        )

        assert status == 413
        assert app_received == []
        assert receive_calls == 0

    def test_a_malformed_declared_length_falls_back_to_counting(self) -> None:
        status, app_received, _ = self.run(
            [{"type": "http.request", "body": b"small", "more_body": False}],
            headers=[(b"content-length", b"not-a-number")],
        )

        assert status == 200
        assert app_received == [b"small"]
