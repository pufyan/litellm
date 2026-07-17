# LiteLLM Realtime session contract

This is the single client-facing contract for the LiteLLM proxy `/v1/realtime` WebSocket. Clients send one provider-independent schema, and the proxy maps it to each backend (OpenAI, Azure, xAI, Gemini, Vertex AI, Bedrock Nova Sonic) internally. You do not send provider-specific shapes.

## Connecting

Open a WebSocket to the proxy realtime endpoint with the model you want:

```
wss://<proxy-host>/v1/realtime?model=<model_name>
Authorization: Bearer <litellm_virtual_key>
```

Do not send the `OpenAI-Beta: realtime=v1` header. Without it the proxy runs in GA mode and applies the canonical -> provider remap. Sending that header forces legacy beta passthrough for OpenAI-compatible backends and disables the remap, so keep it off unless you specifically need beta event names.

## Canonical `session.update`

Configure the session by sending a `session.update` event. Use the flat schema below. The proxy translates every field to the active provider.

```json
{
  "type": "session.update",
  "session": {
    "modalities": ["audio"],
    "instructions": "You are a helpful voice assistant",
    "voice": "marin",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "input_audio_transcription": { "model": "whisper-1" },
    "turn_detection": { "type": "server_vad", "threshold": 0.5 },
    "tools": [],
    "tool_choice": "auto",
    "temperature": 0.8,
    "max_response_output_tokens": 4096
  }
}
```

### Field reference

| Field | Type | Notes |
|---|---|---|
| `modalities` | `["audio"]` \| `["text"]` \| `["audio","text"]` | GA collapses combined to a single output modality; audio mode still delivers transcripts via events |
| `instructions` | string | System prompt for the session |
| `voice` | string | Provider-specific voice id. For OpenAI GA use `marin` / `cedar` (also alloy, ash, ballad, coral, echo, sage, shimmer, verse) |
| `input_audio_format` | string \| object | e.g. `pcm16`, `g711_ulaw` |
| `output_audio_format` | string \| object | e.g. `pcm16`, `g711_ulaw` |
| `input_audio_transcription` | object | e.g. `{ "model": "whisper-1" }` |
| `turn_detection` | object | e.g. `{ "type": "server_vad", "threshold": 0.5 }` |
| `tools` / `tool_choice` | array / string | Function tools and selection policy |
| `temperature` | number | Sampling temperature |
| `max_response_output_tokens` | int \| `"inf"` | Max output tokens per assistant response, 1..4096 or `"inf"` |

The `voice` and `max_response_output_tokens` values cannot be changed once the model has emitted audio in a session. Set them on the first `session.update`.

## Provider support matrix

The proxy owns the mapping. This table records how complete each backend family is against the canonical contract as of this document.

| Canonical field | OpenAI / Azure / xAI | Gemini / Vertex AI | Bedrock Nova Sonic |
|---|---|---|---|
| `voice` | yes | yes | yes |
| `modalities` | yes | yes | fixed to `["text","audio"]` |
| `instructions` | yes | yes | yes |
| `input_audio_format` | yes | fixed format | yes |
| `output_audio_format` | yes | fixed format | yes |
| `input_audio_transcription` | yes | yes | not mapped |
| `turn_detection` | yes | yes | not mapped |
| `tools` / `tool_choice` | yes | yes | yes |
| `temperature` | yes | yes | yes |
| `max_response_output_tokens` | yes | yes | yes |

Bedrock Nova Sonic is the least complete: it ignores `turn_detection` and `input_audio_transcription` and hardcodes modalities. Everything else is uniform across all three families. If a backend does not map a field, it is dropped rather than forwarded, so a canonical payload never breaks a session.

## Where the mapping lives

| Family | Providers | Mapping code |
|---|---|---|
| OpenAI-compatible | `openai`, `azure`, `xai` | `litellm/litellm_core_utils/realtime_streaming.py` (`_remap_beta_session_to_ga`) plus per-provider event normalizers |
| Gemini | `gemini`, `vertex_ai` | `litellm/llms/gemini/realtime/transformation.py` (Vertex extends `GeminiRealtimeConfig`) |
| Bedrock | `bedrock` | `litellm/llms/bedrock/realtime/transformation.py` |

When you add a provider or a new canonical field, the rule is to extend the mapping inside the provider implementation (or the shared remap for OpenAI-compatible backends), never to push provider-specific shapes onto clients.
