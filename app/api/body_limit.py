"""Refuse request bodies larger than the API could ever need.

Neither uvicorn nor Starlette caps the size of a request body: left alone, the
server reads whatever it is sent into memory before validation even runs.
Measured before this existed: a 20 MB body was read in full just to be told
its query was too long.

This is *ASGI middleware* — the Python web-server interface uvicorn and
FastAPI share. It plays the role of ASP.NET Core middleware (a component that
sees every request before the endpoint does), but at the raw protocol level:
it receives the request as a stream of messages and can answer on its own
without ever calling the app behind it.
"""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# The largest valid search is a 500-character query. Even written entirely as
# JSON \\uXXXX escapes of astral characters it stays near 6 KB, so 16 KiB
# rejects nothing legitimate while bounding what any request can cost.
MAX_BODY_BYTES = 16 * 1024


class BodySizeLimitMiddleware:
    """Answer 413 to any request whose body exceeds ``max_bytes``.

    Two checks, because a client can send a body two ways:

    - With a ``Content-Length`` header: refused immediately, unread.
    - Chunked, with no declared length: counted as it arrives, and refused
      the moment it passes the limit — never buffered beyond it.

    A body within the limit is buffered and handed to the app unchanged.

    Args:
        app: The application behind this middleware.
        max_bytes: Largest body accepted.
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    # `async def __call__` makes an instance callable as an awaitable function,
    # which is exactly the shape ASGI expects of an application.
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Lifespan and websocket traffic carry no request body.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        declared = _declared_length(scope)
        if declared is not None and declared > self.max_bytes:
            await self._refuse(scope, receive, send)
            return

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # the client went away; nobody is left to answer
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.max_bytes:
                await self._refuse(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        delivered = False

        # The app expects to read the body through `receive`, which this
        # middleware has already drained, so hand it a replacement that
        # returns the buffered body once and then defers to the real one
        # (which from then on only reports a disconnect).
        async def replay() -> Message:
            # `nonlocal` lets the inner function rebind a variable of the
            # enclosing one — C# closures capture variables by reference
            # anyway, so there is no direct equivalent to write there.
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)

    async def _refuse(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            {"detail": f"Request body too large: the limit is {self.max_bytes} bytes."},
            status_code=413,
        )
        await response(scope, receive, send)


def _declared_length(scope: Scope) -> int | None:
    """The request's Content-Length, or None if absent or unreadable.

    ASGI delivers headers as a list of (name, value) byte pairs, names
    lowercased.
    """
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None  # malformed; the body is still counted as it arrives
    return None
