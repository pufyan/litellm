# LiteLLM Realtime session contract

This is the single client-facing contract for the LiteLLM proxy `/v1/realtime` WebSocket. Clients send one provider-independent schema, and the proxy maps it to each backend (OpenAI, Azure, xAI, Gemini, Vertex AI, Bedrock Nova Sonic) internally. You do not send provider-specific shapes.

## Design model (read this first if you maintain the mapping)

The contract is a **union superset** built on the OpenAI realtime session schema:

1. The base shape is the OpenAI flat session. Field names and value shapes follow OpenAI wherever OpenAI has an equivalent.
2. Fields that only some providers support are added flat at the same `session` level (no nested `provider_params` namespace). Example: `top_p`, `top_k`, `context_window_compression` come from Gemini but live next to the OpenAI fields.
3. Each provider implementation owns its own mapping. It maps the canonical fields it understands, applies its own defaults for anything the client omitted, and silently drops any canonical field it does not support.

Consequences you must respect:

- A canonical payload never breaks a session. Unsupported fields are dropped, not forwarded, so no backend errors on a field it does not know.
- The drop is silent. There is no error and no echo telling the client a field was ignored, so the contract and the support matrix below are the source of truth for what actually takes effect on a given provider.
- Omitted fields are handled by whether the backend requires them, never by a litellm opinion:
  - Optional field the client did not send: not forwarded at all, so the backend applies its own native default.
  - Field the backend requires but the client did not send: filled with that provider's native default value, because the request would be rejected otherwise. This is the only case where an implementation writes a default, and the value must match the provider's documented native default, not an arbitrary litellm choice.

The distinction matters: litellm must not hardcode a default for an optional inference parameter (that would silently override the vendor default and drift when the vendor changes it). Hardcoded defaults are allowed only for fields the provider's protocol makes mandatory.

This is intentionally more flexible than a lowest-common-denominator contract: clients get the full expressive range of every backend from one schema, at the cost of per-provider support gaps that must be documented rather than hidden.

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
    "top_p": 0.95,
    "top_k": 40,
    "max_response_output_tokens": 4096,
    "context_window_compression": { "sliding_window": {} }
  }
}
```

### Field reference

The `Origin` column records where the field came from. `OpenAI` fields are the base schema; `union` fields are provider extensions added flat per the design model above.

| Field | Origin | Type | Notes |
|---|---|---|---|
| `modalities` | OpenAI | `["audio"]` \| `["text"]` \| `["audio","text"]` | GA collapses combined to a single output modality; audio mode still delivers transcripts via events |
| `instructions` | OpenAI | string | System prompt for the session |
| `voice` | OpenAI | string | Provider-specific voice id. For OpenAI GA use `marin` / `cedar` (also alloy, ash, ballad, coral, echo, sage, shimmer, verse) |
| `input_audio_format` | OpenAI | string \| object | e.g. `pcm16`, `g711_ulaw` |
| `output_audio_format` | OpenAI | string \| object | e.g. `pcm16`, `g711_ulaw` |
| `input_audio_transcription` | OpenAI | object | e.g. `{ "model": "whisper-1" }` |
| `turn_detection` | OpenAI | object | e.g. `{ "type": "server_vad", "threshold": 0.5 }` |
| `tools` / `tool_choice` | OpenAI | array / string | Function tools and selection policy |
| `max_response_output_tokens` | OpenAI | int \| `"inf"` | Max output tokens per assistant response, 1..4096 or `"inf"` |
| `temperature` | union | number | Sampling temperature. Not part of the OpenAI GA session schema; honored by Gemini and Bedrock, dropped for OpenAI |
| `top_p` | union | number | Nucleus sampling. Gemini and Bedrock only; dropped for OpenAI |
| `top_k` | union | int | Top-k sampling. Gemini only; dropped for OpenAI and Bedrock |
| `context_window_compression` | union | object | Context-window management. See the aliasing note below; behavior differs per provider |

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
| `max_response_output_tokens` | yes | yes | yes |
| `temperature` | dropped | yes | yes |
| `top_p` | dropped | yes | yes |
| `top_k` | dropped | yes | dropped |
| `context_window_compression` | as `truncation` | yes (compression) | dropped |

Bedrock Nova Sonic is the least complete: it ignores `turn_detection` and `input_audio_transcription` and hardcodes modalities. If a backend does not map a field, it is dropped rather than forwarded, so a canonical payload never breaks a session.

## Where the mapping lives

| Family | Providers | Mapping code |
|---|---|---|
| OpenAI-compatible | `openai`, `azure`, `xai` | `litellm/litellm_core_utils/realtime_streaming.py` (`_remap_beta_session_to_ga`) plus per-provider event normalizers |
| Gemini | `gemini`, `vertex_ai` | `litellm/llms/gemini/realtime/transformation.py` (Vertex extends `GeminiRealtimeConfig`) |
| Bedrock | `bedrock` | `litellm/llms/bedrock/realtime/transformation.py` |

When you add a provider or a new canonical field, the rule is to extend the mapping inside the provider implementation (or the shared remap for OpenAI-compatible backends), never to push provider-specific shapes onto clients.

## Semantic aliasing policy

### Hard invariant: one name, one meaning

The canonical namespace must stay collision-free. A given field name means exactly one thing across every provider. It is never acceptable to have two providers read the same canonical key as two different concepts, or to give the same key two different value semantics per provider.

Concretely, this forbids:

- Reusing an existing canonical name for a new, unrelated concept on a new provider. Pick a new name instead.
- A canonical field whose value shape or unit changes depending on which provider is active (for example one provider reading `temperature` as 0..1 and another as 0..2 under the same name). If the meaning cannot be made identical, use two separate names.

A canonical field may vary in *how completely* a provider honors it (documented in "Provider-specific behavior"), but never in *what it means*. If two provider fields cannot share one meaning, they must not share one name. This invariant is what makes the union contract safe to reason about; everything else in this policy exists to protect it.

### Unifying fields with different names

Some providers express the same user intent under different field names (`max_output_tokens` vs `maxOutputTokens` vs `maxTokens`; `voice` vs `voiceId`). Those are safe to unify under one canonical name, and the proxy already does.

The trap is fields that share a topic but not behavior. The rule for deciding whether to merge two provider fields into one canonical field:

> Unify two provider fields under one canonical field only if, for a **typical** value, the user's intent and the observable result match on every target provider. Divergence at edge values is acceptable but must be listed in "Provider-specific behavior" below. If only the topic matches while the base behavior differs, keep them as separate fields.

The principle is unify the intent, not the JSON key. A canonical field is a contract about meaning; each provider interprets that meaning in its own terms and degrades documented-ly for values it cannot honor.

Worked example, `context_window_compression`. OpenAI's native `truncation` (an enum-like strategy that drops old context) and Gemini's `contextWindowCompression` (sliding-window compression / summarization) address the same topic, keeping the context window from overflowing, with different behavior. They are unified under the canonical `context_window_compression` by intent, with these documented degradations:

- OpenAI: maps the canonical value onto native `truncation`. Exact drop-by-count strategies are honored where OpenAI supports them.
- Gemini / Vertex: enables `contextWindowCompression`. This compresses rather than hard-drops context, so a request for exact truncation becomes compression, not deletion.
- Bedrock Nova Sonic: no equivalent; dropped.

Because the drop and the compression-vs-truncation difference are both silent at runtime, they are documented here so clients can reason about cross-provider behavior.

## Provider-specific behavior

Deviations from a naive reading of the contract, per provider. Keep this list in sync with the mappings.

- Bedrock Nova Sonic: ignores `turn_detection` and `input_audio_transcription`; hardcodes modalities to `["text","audio"]`; ignores `top_k`. The `sessionStart.inferenceConfiguration` block (`maxTokens`, `topP`, `temperature`) and `promptStart.audioOutputConfiguration.voiceId` are required-by-protocol: the AWS bidirectional-stream event schema presents them as part of the fixed event structure with no optional marker, so the implementation always sends them and falls back to the values from the AWS documentation examples (`maxTokens: 1024`, `topP: 0.9`, `temperature: 0.7`, `voiceId: "matthew"`) when the client omits them. This is the sanctioned required-by-protocol default, not a litellm opinion. Source: https://docs.aws.amazon.com/nova/latest/userguide/input-events.html
- OpenAI GA: has no session-level `temperature` / `top_p` / `top_k`; these union fields are dropped. `context_window_compression` is applied as native `truncation`.
- Gemini / Vertex AI: audio formats are fixed by the Live model rather than taken from `input_audio_format` / `output_audio_format`. `context_window_compression` compresses context rather than hard-truncating it.

## Rules for contributors

- Client-facing surface is this contract only. Never require a client to send a provider-native shape.
- All mapping lives in the provider implementation (or the shared OpenAI-compatible remap). Adding a provider means adding its mapping there, not changing the client contract.
- A field the provider cannot honor is dropped silently; it must not be forwarded to the backend, and it must be recorded in the support matrix and, if its behavior is surprising, in "Provider-specific behavior".
- Do not hardcode a default for an optional field. If the client omits an optional field, do not send it and let the backend default apply. Write a default only for a field the provider's protocol makes mandatory, and use the provider's native default value.
- Before adding or unifying a field, enforce the one-name-one-meaning invariant: never reuse a canonical name for a different concept and never let a name's value semantics change per provider. If two things cannot share one meaning, give them two names.
- Before unifying a new pair of fields, apply the semantic aliasing rule above. When in doubt, keep them separate.
- Any new union field must land in the field reference (with `Origin: union`), the support matrix, and, if it degrades unevenly, the provider-specific section.
