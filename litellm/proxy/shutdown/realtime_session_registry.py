"""
Process-scoped registry of active realtime WebSocket sessions, used to drain
and gracefully close them on ``SIGTERM`` (Kubernetes rolling update, scale-down,
liveness kill).

``InFlightRequestsMiddleware`` only counts HTTP requests (it explicitly skips
non-http ASGI scopes), so long-lived realtime WebSocket sessions are invisible
to ``GracefulShutdownManager.wait_for_drain``. Without this registry a rolling
update tears the process down while realtime sessions are still mid-turn,
dropping them with an abrupt TCP reset instead of a protocol close.

Realtime sessions are long-lived by nature (minutes to hours), so unlike HTTP
requests they can not be drained "to completion". On SIGTERM/SIGINT a signal
handler (installed at startup, see ``install_signal_drain``) lets existing
sessions run to their natural end within ``WS_DRAIN_TIMEOUT`` (default 600s),
then force-closes whatever remains with WebSocket code 1012 (Service Restart)
so clients know to reconnect — at which point the pod is already out of the
Service endpoints and the reconnect lands on a live pod. The handler runs
before uvicorn's own shutdown, which would otherwise close the sockets first
and leave the drain nothing to wait on.

The drain window only takes effect if the pod is given time to use it: set
``terminationGracePeriodSeconds`` above ``WS_DRAIN_TIMEOUT`` (plus headroom for
HTTP drain and dependency teardown), otherwise Kubernetes sends SIGKILL
mid-drain and the graceful path never runs.

State is class-level and therefore scoped to a single uvicorn worker process,
matching the granularity of the other shutdown primitives.
"""

import asyncio
import os
import signal
from typing import TYPE_CHECKING, Callable

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming


DEFAULT_WS_DRAIN_TIMEOUT = 600.0
_DRAIN_POLL_INTERVAL = 0.5
_DRAIN_LOG_INTERVAL = 5.0
_DRAIN_SIGNALS = (signal.SIGTERM, signal.SIGINT)


class RealtimeSessionRegistry:
    """
    Tracks active realtime sessions on this worker so shutdown can drain them
    within a bounded window and then force-close the stragglers.
    """

    _sessions: "frozenset[RealTimeStreaming]" = frozenset()
    _draining: bool = False
    _prev_handlers: "dict[int, Callable[[], None]]" = {}

    @classmethod
    def register(cls, session: "RealTimeStreaming") -> None:
        cls._sessions = cls._sessions | {session}
        verbose_proxy_logger.debug("realtime_session_registered active_sessions=%s", len(cls._sessions))

    @classmethod
    def unregister(cls, session: "RealTimeStreaming") -> None:
        cls._sessions = cls._sessions - {session}
        verbose_proxy_logger.debug("realtime_session_unregistered active_sessions=%s", len(cls._sessions))

    @classmethod
    def count(cls) -> int:
        return len(cls._sessions)

    @classmethod
    def get_ws_drain_timeout(cls) -> float:
        raw = os.getenv("WS_DRAIN_TIMEOUT")
        if raw is None:
            return DEFAULT_WS_DRAIN_TIMEOUT
        try:
            return float(raw)
        except (TypeError, ValueError):
            verbose_proxy_logger.warning(
                "WS_DRAIN_TIMEOUT=%r is not a number; using default %ss",
                raw,
                DEFAULT_WS_DRAIN_TIMEOUT,
            )
            return DEFAULT_WS_DRAIN_TIMEOUT

    @classmethod
    async def drain(
        cls,
        timeout: float,
        poll_interval: float = _DRAIN_POLL_INTERVAL,
        log_interval: float = _DRAIN_LOG_INTERVAL,
    ) -> int:
        initial = cls.count()
        verbose_proxy_logger.info(
            "realtime_drain_started active_sessions=%s timeout_s=%s",
            initial,
            timeout,
        )
        if initial == 0:
            return 0

        elapsed = 0.0
        since_log = 0.0
        while cls.count() > 0 and elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            since_log += poll_interval
            if since_log >= log_interval:
                verbose_proxy_logger.info(
                    "realtime_drain_waiting active_sessions=%s elapsed_s=%.1f",
                    cls.count(),
                    elapsed,
                )
                since_log = 0.0

        remaining = cls.count()
        if remaining == 0:
            verbose_proxy_logger.info(
                "realtime_drain_complete drained_sessions=%s elapsed_s=%.1f",
                initial,
                elapsed,
            )
        else:
            verbose_proxy_logger.warning(
                "realtime_drain_timeout active_sessions=%s elapsed_s=%.1f timeout_s=%s — force-closing stragglers",
                remaining,
                elapsed,
                timeout,
            )
        return initial - remaining

    @classmethod
    async def force_close_all(cls) -> int:
        sessions = cls._sessions
        if not sessions:
            return 0

        results = await asyncio.gather(
            *(session.shutdown_close() for session in sessions),
            return_exceptions=True,
        )
        failures = tuple(r for r in results if isinstance(r, BaseException))
        for failure in failures:
            verbose_proxy_logger.debug("realtime force-close failed: %s", failure)
        verbose_proxy_logger.info(
            "realtime_force_closed sessions=%s failures=%s",
            len(sessions),
            len(failures),
        )
        return len(sessions)

    @classmethod
    def install_signal_drain(cls, loop: "asyncio.AbstractEventLoop") -> None:
        """Drain realtime sessions on SIGTERM/SIGINT before the server tears down.

        uvicorn closes active WebSockets as part of its own shutdown, which runs
        before the ASGI lifespan shutdown. Draining from lifespan therefore always
        sees an empty registry. Instead we intercept the signal first: mark the
        worker draining (new connections are rejected), let existing sessions run
        to their natural end within WS_DRAIN_TIMEOUT, force-close the stragglers,
        then delegate to uvicorn's original handler so its normal shutdown
        proceeds. The first signal drains; a second one delegates immediately so
        an operator can still force an early exit.
        """
        for signum in _DRAIN_SIGNALS:
            try:
                prev = signal.getsignal(signum)
                if callable(prev):
                    cls._prev_handlers[signum] = prev  # type: ignore[assignment]
                loop.add_signal_handler(signum, cls._on_signal, signum, loop)
            except (NotImplementedError, RuntimeError, ValueError) as e:
                verbose_proxy_logger.warning("realtime drain signal handler not installed for %s: %s", signum, e)

    @classmethod
    def _delegate_to_prev(cls, signum: int) -> None:
        prev = cls._prev_handlers.get(signum)
        if prev is None:
            return
        # uvicorn installs Server.handle_exit via signal.signal, so it expects
        # the (signum, frame) signature of a signal.signal handler, not the
        # zero-arg callback shape of loop.add_signal_handler.
        prev(signum, None)

    @classmethod
    def _on_signal(cls, signum: int, loop: "asyncio.AbstractEventLoop") -> None:
        if cls._draining:
            cls._delegate_to_prev(signum)
            return
        cls._draining = True
        if cls.count() == 0:
            cls._delegate_to_prev(signum)
            return
        loop.create_task(cls._drain_then_delegate(signum))

    @classmethod
    async def _drain_then_delegate(cls, signum: int) -> None:
        try:
            await cls.drain(timeout=cls.get_ws_drain_timeout())
            await cls.force_close_all()
        finally:
            cls._delegate_to_prev(signum)

    @classmethod
    def is_draining(cls) -> bool:
        return cls._draining

    @classmethod
    def reset(cls) -> None:
        cls._sessions = frozenset()
        cls._draining = False
        cls._prev_handlers = {}
