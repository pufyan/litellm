"""
Tests for RealtimeSessionRegistry.

These verify that active realtime WebSocket sessions are tracked, drained within
a bounded window on shutdown, and force-closed when the window elapses — the
behaviour that keeps a Kubernetes rolling update from dropping in-flight
realtime sessions with an abrupt reset.
"""

import asyncio

import pytest

from litellm.proxy.shutdown.realtime_session_registry import (
    DEFAULT_WS_DRAIN_TIMEOUT,
    RealtimeSessionRegistry,
)


@pytest.fixture(autouse=True)
def _reset():
    RealtimeSessionRegistry.reset()
    yield
    RealtimeSessionRegistry.reset()


class FakeSession:
    def __init__(self, raise_on_close: bool = False) -> None:
        self.closed = False
        self._raise_on_close = raise_on_close

    async def shutdown_close(self) -> None:
        if self._raise_on_close:
            raise RuntimeError("close failed")
        self.closed = True


def test_empty_by_default():
    assert RealtimeSessionRegistry.count() == 0


def test_register_and_unregister_change_count():
    a, b = FakeSession(), FakeSession()
    RealtimeSessionRegistry.register(a)
    RealtimeSessionRegistry.register(b)
    assert RealtimeSessionRegistry.count() == 2
    RealtimeSessionRegistry.unregister(a)
    assert RealtimeSessionRegistry.count() == 1


def test_register_is_idempotent():
    a = FakeSession()
    RealtimeSessionRegistry.register(a)
    RealtimeSessionRegistry.register(a)
    assert RealtimeSessionRegistry.count() == 1


def test_get_ws_drain_timeout_default(monkeypatch):
    monkeypatch.delenv("WS_DRAIN_TIMEOUT", raising=False)
    assert RealtimeSessionRegistry.get_ws_drain_timeout() == DEFAULT_WS_DRAIN_TIMEOUT
    assert DEFAULT_WS_DRAIN_TIMEOUT == 600.0


def test_get_ws_drain_timeout_from_env(monkeypatch):
    monkeypatch.setenv("WS_DRAIN_TIMEOUT", "12.5")
    assert RealtimeSessionRegistry.get_ws_drain_timeout() == 12.5


def test_get_ws_drain_timeout_malformed_falls_back(monkeypatch):
    monkeypatch.setenv("WS_DRAIN_TIMEOUT", "not-a-number")
    assert RealtimeSessionRegistry.get_ws_drain_timeout() == DEFAULT_WS_DRAIN_TIMEOUT


@pytest.mark.asyncio
async def test_drain_returns_immediately_when_no_sessions():
    drained = await RealtimeSessionRegistry.drain(timeout=5.0)
    assert drained == 0


@pytest.mark.asyncio
async def test_drain_completes_early_when_sessions_leave():
    a = FakeSession()
    RealtimeSessionRegistry.register(a)

    async def leave_soon():
        await asyncio.sleep(0.2)
        RealtimeSessionRegistry.unregister(a)

    task = asyncio.create_task(leave_soon())
    start = asyncio.get_event_loop().time()
    drained = await RealtimeSessionRegistry.drain(timeout=10.0, poll_interval=0.05)
    elapsed = asyncio.get_event_loop().time() - start
    await task

    assert drained == 1
    assert elapsed < 5.0


@pytest.mark.asyncio
async def test_drain_respects_timeout_when_sessions_remain():
    RealtimeSessionRegistry.register(FakeSession())
    start = asyncio.get_event_loop().time()
    drained = await RealtimeSessionRegistry.drain(timeout=0.3, poll_interval=0.05)
    elapsed = asyncio.get_event_loop().time() - start

    assert drained == 0
    assert RealtimeSessionRegistry.count() == 1
    assert elapsed >= 0.3


@pytest.mark.asyncio
async def test_force_close_all_closes_every_session():
    a, b = FakeSession(), FakeSession()
    RealtimeSessionRegistry.register(a)
    RealtimeSessionRegistry.register(b)

    closed = await RealtimeSessionRegistry.force_close_all()

    assert closed == 2
    assert a.closed and b.closed


@pytest.mark.asyncio
async def test_force_close_all_survives_a_failing_session():
    good = FakeSession()
    bad = FakeSession(raise_on_close=True)
    RealtimeSessionRegistry.register(good)
    RealtimeSessionRegistry.register(bad)

    closed = await RealtimeSessionRegistry.force_close_all()

    assert closed == 2
    assert good.closed is True


import signal


def test_not_draining_by_default():
    assert RealtimeSessionRegistry.is_draining() is False


def test_on_signal_with_no_sessions_delegates_immediately():
    called = {"n": 0}
    RealtimeSessionRegistry._prev_handlers = {signal.SIGTERM: lambda s, f: called.__setitem__("n", called["n"] + 1)}
    loop = asyncio.new_event_loop()
    try:
        RealtimeSessionRegistry._on_signal(signal.SIGTERM, loop)
    finally:
        loop.close()
    assert RealtimeSessionRegistry.is_draining() is True
    assert called["n"] == 1


def test_second_signal_delegates_without_new_drain():
    called = {"n": 0}
    RealtimeSessionRegistry._prev_handlers = {signal.SIGTERM: lambda s, f: called.__setitem__("n", called["n"] + 1)}
    RealtimeSessionRegistry.register(FakeSession())
    RealtimeSessionRegistry._draining = True
    loop = asyncio.new_event_loop()
    try:
        RealtimeSessionRegistry._on_signal(signal.SIGTERM, loop)
    finally:
        loop.close()
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_drain_then_delegate_closes_sessions_and_delegates():
    called = {"n": 0}
    RealtimeSessionRegistry._prev_handlers = {signal.SIGTERM: lambda s, f: called.__setitem__("n", called["n"] + 1)}
    a = FakeSession()
    RealtimeSessionRegistry.register(a)

    async def leave_soon():
        await asyncio.sleep(0.1)
        RealtimeSessionRegistry.unregister(a)

    task = asyncio.create_task(leave_soon())
    await RealtimeSessionRegistry._drain_then_delegate(signal.SIGTERM)
    await task

    assert called["n"] == 1


@pytest.mark.asyncio
async def test_drain_then_delegate_force_closes_on_timeout(monkeypatch):
    monkeypatch.setenv("WS_DRAIN_TIMEOUT", "0.2")
    called = {"n": 0}
    RealtimeSessionRegistry._prev_handlers = {signal.SIGTERM: lambda s, f: called.__setitem__("n", called["n"] + 1)}
    a = FakeSession()
    RealtimeSessionRegistry.register(a)

    await RealtimeSessionRegistry._drain_then_delegate(signal.SIGTERM)

    assert a.closed is True
    assert called["n"] == 1


def test_delegate_calls_prev_with_signal_signal_signature():
    """uvicorn's Server.handle_exit is installed via signal.signal and requires
    (signum, frame); a zero-arg call raised TypeError and left the server
    running. Guard the two-arg contract."""
    received = {}
    RealtimeSessionRegistry._prev_handlers = {
        signal.SIGTERM: lambda s, f: received.update(sig=s, frame=f)
    }
    RealtimeSessionRegistry._delegate_to_prev(signal.SIGTERM)
    assert received == {"sig": signal.SIGTERM, "frame": None}
