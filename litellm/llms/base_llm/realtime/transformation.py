from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional, Protocol

import httpx
from typing_extensions import Self

from litellm.types.llms.openai import OpenAIRealtimeStreamSessionEvents
from litellm.types.realtime import (
    RealtimeGoAwayNotice,
    RealtimeInputAudioTranscriptionUsage,
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
    RealtimeResumptionState,
    RealtimeTranscriptEntry,
)

from ..chat.transformation import BaseLLMException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class RealtimeBackend(Protocol):
    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def send(self, message: str | bytes) -> None: ...

    async def recv(self, decode: bool | None = None) -> str | bytes: ...

    async def close(self) -> None: ...


class BaseRealtimeConfig(ABC):
    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
    ) -> dict:
        pass

    @abstractmethod
    def get_complete_url(self, api_base: str | None, model: str, api_key: str | None = None) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    @abstractmethod
    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: str | None = None,
    ) -> Sequence[str | bytes]:
        pass

    async def pace_backend_send(self, message: bytes) -> None:
        return None

    def is_setup_message(self, msg_obj: dict) -> bool:
        return False

    def is_content_message(self, msg_obj: dict) -> bool:
        return False

    def requires_session_configuration(
        self,
    ) -> bool:  # initial configuration message sent to setup the realtime session
        return False

    def session_configuration_request(self, model: str) -> str | None:  # message sent to setup the realtime session
        return None

    def unbilled_usage_on_session_close(self, model: str) -> RealtimeInputAudioTranscriptionUsage | None:
        return None

    async def open_backend(self, url: str, headers: Mapping[str, str]) -> RealtimeBackend | None:
        return None

    def transform_session_created_event(
        self,
        model: str,
        logging_session_id: str,
        session_configuration_request: str | None = None,
    ) -> Mapping[str, object] | OpenAIRealtimeStreamSessionEvents | None:
        """
        Optional hook for providers that defer session setup until client `session.update`.

        Return an OpenAI-compatible `session.created` payload when the proxy should
        emit a synthetic event immediately after backend websocket connection.
        """
        return None

    def supports_session_resumption(self) -> bool:
        """Whether the provider backend can resume a live session on a fresh
        websocket (e.g. Gemini Live's sessionResumption handle)."""
        return False

    def resumption_event_markers(self) -> "tuple[str, ...]":
        """Cheap substring markers for resumption/goAway service frames. A raw
        backend frame containing none of them is skipped without JSON parsing
        (audio deltas dominate backend traffic). Empty tuple disables the
        fast-path and every frame is parsed."""
        return ()

    def extract_resumption_state(self, event: dict) -> Optional[RealtimeResumptionState]:
        """Return updated resumption state when ``event`` carries a new resumption
        token/handle from the backend; None otherwise."""
        return None

    def extract_go_away(self, event: dict) -> Optional[RealtimeGoAwayNotice]:
        """Return a notice when ``event`` announces the backend will close the
        connection soon (e.g. Gemini Live ``goAway``); None otherwise."""
        return None

    def build_resume_session_request(
        self, state: RealtimeResumptionState, original_session_request: Optional[str]
    ) -> Optional[str]:
        """Build the first message for a re-opened backend socket that resumes the
        prior session from ``state``. Default: re-send the original setup (fresh
        session, prior server-side context is lost)."""
        return original_session_request

    def build_history_replay_messages(self, entries: "tuple[RealtimeTranscriptEntry, ...]") -> "Optional[list[str]]":
        """Build provider-native messages that inject the accumulated conversation
        transcript into a freshly restarted session (used when no native
        resumption token is available). Return None when the provider cannot
        replay history; the reconnect then proceeds with an empty context."""
        return None

    def merge_setup_for_reconnect(
        self, original_session_request: Optional[str], rejected_session_update: str
    ) -> Optional[str]:
        """Fold a ``session.update`` the backend cannot accept mid-session
        (e.g. Gemini/Vertex Live, which reject any ``setup`` after the first)
        into a new setup request for a reconnect.

        Return the merged setup to make ``_send_to_backend`` reconnect the
        backend socket and resume with it instead of silently dropping the
        update. Return ``None`` (the default) to keep the existing drop
        behavior for providers that have not opted into this, or don't need it
        because they accept ``session.update`` at any point in the session."""
        return None

    @abstractmethod
    def transform_realtime_response(
        self,
        message: str | bytes,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:  # message sent to setup the realtime session
        """
        Keep this state less - leave the state management (e.g. tracking current_output_item_id, current_response_id, current_conversation_id, current_delta_chunks) to the caller.
        """
