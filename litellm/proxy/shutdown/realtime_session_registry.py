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
requests they can not be drained "to completion". The lifespan shutdown lets
them run to their natural end within ``WS_DRAIN_TIMEOUT`` (default 600s), then
force-closes whatever remains with WebSocket code 1012 (Service Restart) so
clients know to reconnect — at which point the pod is already out of the
Service endpoints and the reconnect lands on a live pod.

The drain window only takes effect if the pod is given time to use it: set
``terminationGracePeriodSeconds`` above ``WS_DRAIN_TIMEOUT`` (plus headroom for
HTTP drain and dependency teardown), otherwise Kubernetes sends SIGKILL
mid-drain and the graceful path never runs.

State is class-level and therefore scoped to a single uvicorn worker process,
matching the granularity of the other shutdown primitives.
"""

import asyncio
import os
from typing import TYPE_CHECKING

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming


DEFAULT_WS_DRAIN_TIMEOUT = 600.0
_DRAIN_POLL_INTERVAL = 0.5
_DRAIN_LOG_INTERVAL = 5.0


class RealtimeSessionRegistry:
    """
    Tracks active realtime sessions on this worker so shutdown can drain them
    within a bounded window and then force-close the stragglers.
    """

    _sessions: "frozenset[RealTimeStreaming]" = frozenset()

    @classmethod
    def register(cls, session: "RealTimeStreaming") -> None:
        cls._sessions = cls._sessions | {session}

    @classmethod
    def unregister(cls, session: "RealTimeStreaming") -> None:
        cls._sessions = cls._sessions - {session}

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
        if initial == 0:
            return 0

        verbose_proxy_logger.info(
            "realtime_drain_started active_sessions=%s timeout_s=%s",
            initial,
            timeout,
        )

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
    def reset(cls) -> None:
        cls._sessions = frozenset()
