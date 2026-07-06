from unittest.mock import AsyncMock, patch

import pytest

from litellm.litellm_core_utils.realtime_backend_connector import RealtimeBackendConnector


@pytest.mark.asyncio
async def test_connect_returns_backend_ws_and_passes_config():
    connector = RealtimeBackendConnector(
        url="wss://example.com/ws",
        headers={"Authorization": "Bearer sk-test"},
        ssl_context=None,
        open_timeout=8.0,
        max_attempts=3,
    )
    fake_ws = object()
    with patch("websockets.connect", new=AsyncMock(return_value=fake_ws)) as mock_connect:
        result = await connector.connect()

    assert result is fake_ws
    assert mock_connect.call_count == 1
    kwargs = mock_connect.call_args.kwargs
    assert kwargs["additional_headers"] == {"Authorization": "Bearer sk-test"}
    assert kwargs["open_timeout"] == 8.0
    assert mock_connect.call_args.args == ("wss://example.com/ws",)


@pytest.mark.asyncio
async def test_connect_retries_hung_handshake_then_succeeds():
    connector = RealtimeBackendConnector(url="wss://example.com/ws", open_timeout=0.1, max_attempts=3)
    fake_ws = object()
    with patch(
        "websockets.connect",
        new=AsyncMock(side_effect=[TimeoutError(), TimeoutError(), fake_ws]),
    ) as mock_connect:
        result = await connector.connect()

    assert result is fake_ws
    assert mock_connect.call_count == 3


@pytest.mark.asyncio
async def test_connect_raises_after_exhausting_attempts():
    connector = RealtimeBackendConnector(url="wss://example.com/ws", max_attempts=2)
    with patch("websockets.connect", new=AsyncMock(side_effect=TimeoutError())) as mock_connect:
        with pytest.raises(TimeoutError):
            await connector.connect()

    assert mock_connect.call_count == 2


@pytest.mark.asyncio
async def test_connect_does_not_retry_deterministic_handshake_rejection():
    import websockets.exceptions

    rejection_cls = getattr(websockets.exceptions, "InvalidStatus", None) or websockets.exceptions.InvalidStatusCode
    rejection = rejection_cls.__new__(rejection_cls)
    connector = RealtimeBackendConnector(url="wss://example.com/ws", max_attempts=3)
    with patch("websockets.connect", new=AsyncMock(side_effect=rejection)) as mock_connect:
        with pytest.raises(rejection_cls):
            await connector.connect()

    assert mock_connect.call_count == 1


@pytest.mark.asyncio
async def test_connect_omits_open_timeout_when_not_configured():
    connector = RealtimeBackendConnector(url="wss://example.com/ws")
    with patch("websockets.connect", new=AsyncMock(return_value=object())) as mock_connect:
        await connector.connect()

    assert "open_timeout" not in mock_connect.call_args.kwargs
