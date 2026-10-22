import hashlib
import json
import math
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

OPENROUTER_PROVIDER: Final = "openrouter"
OPENROUTER_KEY_PREFIX: Final = f"{OPENROUTER_PROVIDER}/"
_DIGEST_LENGTH: Final = 12


class _Pricing(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    prompt: str | None = None
    completion: str | None = None
    image: str | None = None
    input_cache_read: str | None = None
    input_cache_write: str | None = None


class _Architecture(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()


class _TopProvider(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    max_completion_tokens: int | None = None


class OpenRouterModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    context_length: int | None = None
    architecture: _Architecture = _Architecture()
    pricing: _Pricing = _Pricing()
    top_provider: _TopProvider = _TopProvider()
    supported_parameters: tuple[str, ...] = ()


_MODEL_ADAPTER: Final = TypeAdapter(OpenRouterModel)

CatalogEntry = dict[str, object]  # mutable-ok: catalog entries are plain dicts in litellm.model_cost


def _parse_model(raw: object) -> OpenRouterModel | None:
    try:
        return _MODEL_ADAPTER.validate_python(raw)
    except ValidationError:
        return None


def parse_openrouter_models(payload: object) -> tuple[OpenRouterModel, ...]:
    data: Final = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return ()
    return tuple(model for model in (_parse_model(raw) for raw in data) if model is not None)


def _price(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed: Final = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _capabilities(model: OpenRouterModel) -> CatalogEntry:
    params: Final = frozenset(model.supported_parameters)
    inputs: Final = frozenset(model.architecture.input_modalities)
    flags: Final = {
        "supports_function_calling": "tools" in params,
        "supports_tool_choice": "tool_choice" in params,
        "supports_response_schema": bool(params & {"response_format", "structured_outputs"}),
        "supports_reasoning": "reasoning" in params,
        "supports_vision": "image" in inputs,
        "supports_pdf_input": "file" in inputs,
        "supports_audio_input": "audio" in inputs,
        "supports_prompt_caching": _price(model.pricing.input_cache_read) is not None,
    }
    return {name: True for name, enabled in flags.items() if enabled}


def _costs(model: OpenRouterModel) -> CatalogEntry:
    prices: Final = {
        "input_cost_per_token": _price(model.pricing.prompt),
        "output_cost_per_token": _price(model.pricing.completion),
        "cache_read_input_token_cost": _price(model.pricing.input_cache_read),
        "cache_creation_input_token_cost": _price(model.pricing.input_cache_write),
        "input_cost_per_image": _price(model.pricing.image),
    }
    return {name: value for name, value in prices.items() if value is not None}


def _limits(model: OpenRouterModel) -> CatalogEntry:
    limits: Final = {
        "max_tokens": model.context_length,
        "max_input_tokens": model.context_length,
        "max_output_tokens": model.top_provider.max_completion_tokens,
    }
    return {name: value for name, value in limits.items() if value is not None}


def _mode(model: OpenRouterModel) -> str:
    return "image_generation" if model.architecture.output_modalities == ("image",) else "chat"


def build_openrouter_entry(model: OpenRouterModel, source_url: str) -> CatalogEntry | None:
    costs: Final = _costs(model)
    if "input_cost_per_token" not in costs or "output_cost_per_token" not in costs:
        return None
    return {
        "litellm_provider": OPENROUTER_PROVIDER,
        "mode": _mode(model),
        "source": source_url,
        **costs,
        **_limits(model),
        **_capabilities(model),
    }


def build_openrouter_entries(
    payload: object, source_url: str
) -> dict[str, CatalogEntry]:  # mutable-ok: merged into the catalog
    built: Final = (
        (f"{OPENROUTER_KEY_PREFIX}{model.id}", build_openrouter_entry(model, source_url))
        for model in parse_openrouter_models(payload)
    )
    return {key: entry for key, entry in built if entry is not None}


def entries_digest(entries: dict[str, CatalogEntry]) -> str:
    body: Final = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()[:_DIGEST_LENGTH]
