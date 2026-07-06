import ssl
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection


@dataclass(frozen=True, slots=True)
class RealtimeBackendConnector:
    """Knows how to (re)open the backend realtime websocket.

    Injected into ``RealTimeStreaming`` so the proxy loop can re-create the
    backend connection (session resumption) without depending on the
    provider-specific handler that built the URL/headers/SSL config.
    """

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    ssl_context: "ssl.SSLContext | bool | None" = None
    open_timeout: Optional[float] = None
    max_attempts: int = 1

    async def connect(self) -> "ClientConnection":
        """Open the backend realtime websocket, retrying a hung open handshake.

        The upstream Live handshake (e.g. Gemini Live) intermittently hangs on
        open; waiting longer never recovers a hung attempt, but a fresh attempt
        almost always connects in ~1s. So bound each attempt with ``open_timeout``
        and retry, instead of surfacing one slow handshake to the caller as a
        fatal 1011. A bounded attempt that timed out already spaced out the
        retry, so no extra backoff is needed. Deterministic rejections (auth /
        handshake status) are not retried.
        """
        import websockets
        import websockets.exceptions

        # Handshake-status rejections are deterministic (auth / 4xx): retrying
        # cannot help and the caller must see the upstream status, not a generic
        # 1011. websockets <15 raises InvalidStatusCode, >=15 raises InvalidStatus.
        deterministic_errors = tuple(
            exc
            for exc in (
                getattr(websockets.exceptions, "InvalidStatus", None),
                getattr(websockets.exceptions, "InvalidStatusCode", None),
            )
            if exc is not None
        )
        connect_kwargs: dict = {
            "additional_headers": dict(self.headers),
            "max_size": REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
            "ssl": self.ssl_context,
        }
        if self.open_timeout is not None:
            connect_kwargs["open_timeout"] = self.open_timeout
        last_exc: Optional[BaseException] = None
        for _ in range(self.max_attempts):
            try:
                return await websockets.connect(self.url, **connect_kwargs)
            except deterministic_errors:
                raise
            except (
                TimeoutError,
                OSError,
                websockets.exceptions.WebSocketException,
            ) as e:
                last_exc = e
        assert last_exc is not None  # loop only exits via return or a captured exc
        raise last_exc
