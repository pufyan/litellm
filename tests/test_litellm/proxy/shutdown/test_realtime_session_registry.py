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
