"""Canonical realtime session schema.

The provider-independent contract clients send to ``/v1/realtime``; see
``docs/realtime_session_contract.md``. Field names, types and meanings live
here; which backend honors which field does not. Each provider implementation
resolves every field into map / adapt-to-model / drop.
"""

from typing import Dict, List, Literal, Optional, Union

from typing_extensions import TypedDict

TurnDetectionType = Literal["server_vad", "semantic_vad"]
Sensitivity = Literal["high", "medium", "low"]
Eagerness = Literal["low", "medium", "high", "auto"]
TurnCoverage = Literal["activity_only", "all_input", "audio_activity_and_all_video"]
ThinkingLevel = Literal["minimal", "low", "medium", "high"]
MediaResolution = Literal["low", "medium", "high"]
Modality = Literal["audio", "text"]
NoiseReductionType = Literal["near_field", "far_field"]


class CanonicalTurnDetection(TypedDict, total=False):
    """When a user turn ends and how the model reacts to it."""

    type: Optional[TurnDetectionType]
    threshold: float
    prefix_padding_ms: int
    silence_duration_ms: int
    idle_timeout_ms: int
    start_sensitivity: Sensitivity
    end_sensitivity: Sensitivity
    eagerness: Eagerness
    create_response: bool
    interrupt_response: bool
    turn_coverage: TurnCoverage


class CanonicalNoiseReduction(TypedDict):
    type: NoiseReductionType


class CanonicalTranscription(TypedDict, total=False):
    """An empty dict means "enable with backend defaults"."""

    model: str
    language: str
    prompt: str


class CanonicalSlidingWindow(TypedDict, total=False):
    target_tokens: int


class CanonicalContextWindowCompression(TypedDict, total=False):
    """Keep the context window from overflowing, by whatever mechanism the
    backend provides (compression, summarization or truncation)."""

    sliding_window: CanonicalSlidingWindow
    trigger_tokens: int
    target_tokens: int


class CanonicalSessionResumption(TypedDict, total=False):
    enabled: bool


class CanonicalRealtimeSession(TypedDict, total=False):
    """The full canonical ``session`` object of a ``session.update`` event.

    Every field is optional: omission means "use the backend default", which is
    distinct from sending an explicit value.
    """

    modalities: List[Modality]
    instructions: str
    voice: str
    language: str
    tools: List[Dict[str, object]]
    tool_choice: Union[str, Dict[str, object]]
    max_response_output_tokens: Union[int, Literal["inf"]]

    input_audio_format: Union[str, Dict[str, object]]
    output_audio_format: Union[str, Dict[str, object]]
    output_audio_speed: float
    input_audio_transcription: CanonicalTranscription
    output_audio_transcription: CanonicalTranscription
    input_audio_noise_reduction: Optional[CanonicalNoiseReduction]
    transcription_keyterms: List[str]

    turn_detection: Optional[CanonicalTurnDetection]

    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    frequency_penalty: float
    stop_sequences: List[str]
    candidate_count: int

    thinking_budget: int
    thinking_level: ThinkingLevel
    include_thoughts: bool

    context_window_compression: CanonicalContextWindowCompression
    session_resumption: CanonicalSessionResumption
    media_resolution: MediaResolution


CANONICAL_SESSION_KEYS: frozenset[str] = frozenset(CanonicalRealtimeSession.__annotations__)

CANONICAL_TURN_DETECTION_KEYS: frozenset[str] = frozenset(CanonicalTurnDetection.__annotations__)
