# LiteLLM Realtime session contract

This is the single client-facing contract for the LiteLLM proxy `/v1/realtime` WebSocket. Clients send one provider-independent schema, and the proxy maps it to each backend (OpenAI, Azure, xAI, Gemini, Vertex AI, Bedrock Nova Sonic) internally. You do not send provider-specific shapes.

## Design model (read this first if you maintain the mapping)

The contract is a **union superset** built on the OpenAI realtime session schema:

1. The base shape is the OpenAI flat session. Field names and value shapes follow OpenAI wherever OpenAI has an equivalent.
2. Fields that only some providers support are added flat at the same `session` level (no nested `provider_params` namespace). Example: `top_p`, `top_k`, `context_window_compression` come from Gemini but live next to the OpenAI fields.
3. Each provider implementation owns its own mapping. It maps the canonical fields it understands, silently drops any canonical field it does not support, and follows the omitted-field rule below rather than inventing defaults.

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

The model is selected by the `model` query parameter on the URL, not by a `session.model` field. A `session.model` value is optional and, for OpenAI-compatible backends, must be one the GA schema accepts; other providers ignore it.

Do not send the `OpenAI-Beta: realtime=v1` header. Without it the proxy runs in GA mode and applies the canonical -> provider remap described here. Sending that header forces legacy beta passthrough for OpenAI-compatible backends and disables the GA remap and the non-GA-field drop, so the canonical union fields (`temperature`, `top_p`, `top_k`, `context_window_compression`) would then leak to the OpenAI backend and be rejected. Keep the header off unless you specifically need legacy beta event names and are only sending beta-valid fields.

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

One field, one address. The flat canonical schema above is the **only** inbound form; every canonical field has exactly one client-facing name and location. Provider-native aliases sent by a client are silently dropped, never merged:

- a GA-nested `audio` block (`audio.output.voice`, `audio.input.turn_detection`, `audio.input.transcription`, `audio.*.format`) — dropped; the proxy builds the GA `audio` object itself from the flat fields
- GA-renamed keys `output_modalities` and `max_output_tokens` — dropped; use `modalities` and `max_response_output_tokens`
- any provider-native key of any other backend (Gemini/Bedrock shapes) — dropped by the same rule

Clients written directly against the OpenAI GA session shape must switch their `session.update` payloads to the flat canonical form when talking to the proxy; this is a deliberate breaking guarantee that keeps the contract unambiguous. Outbound is unchanged: server events (including the `session.created` / `session.updated` echoes) follow the canonical GA event vocabulary, so configuration reads come from the GA-shaped echo while configuration writes use only the flat canonical schema.

## Nested structures have no safety net

This is the single most important operational rule of the contract. The GA remap and its allowlist drop only clean the **top level** of `session`. They rename and drop top-level keys, but they do not descend into nested objects. A stray provider-specific key inside a nested structure is forwarded verbatim, and a backend that does not recognize it rejects the whole `session.update`, taking the system prompt and tool config down with it.

So the leniency is asymmetric: top-level union fields are forgiven (silently dropped), but every nested structure must be sent in strict canonical / OpenAI form. Send nested objects as if you were talking to OpenAI GA directly.

Nested risk map, highest risk first:

1. `session.tools[].parameters` (JSON Schema). The worst offender.
   - Type case: Gemini emits `"STRING"` / `"OBJECT"` (uppercase); OpenAI wants `"string"` / `"object"`.
   - Recursion: the type appears at any depth under `properties.*`, `items`, `additionalProperties`.
   - Gemini-only noise: keys like `behavior: "BLOCKING"` are rejected by OpenAI.
   - Tool shape itself: Gemini `{ "functionDeclarations": [...] }` vs OpenAI flat `{ "type": "function", "name", "parameters" }`.
2. `session.turn_detection`. Nested `start_sensitivity` / `end_sensitivity` are Gemini-only and leak to OpenAI. Send only `type`, `prefix_padding_ms`, `silence_duration_ms`.
3. `session.input_audio_transcription`. OpenAI requires a non-empty `{ "model": "..." }`; a Gemini-style empty `{}` is rejected.
4. `session.voice`. OpenAI wants a plain string; a Gemini-style `{ "name", "language_code" }` object breaks it.
5. `session.context_window_compression`. Nested plus camelCase: Gemini-native `slidingWindow` / `targetTokens` vs canonical `sliding_window` / `target_tokens`.

Who owes the fix: normalizing nested structures is the **provider implementation's responsibility**, not the client's. The contract stays one canonical schema; every provider-specific nested transformation (type case, foreign keys, shape differences) belongs inside that provider's mapping code.

Implementation status: the safety net now exists on every provider path, built on the shared provider-neutral module `litellm/litellm_core_utils/realtime_schema_normalization.py`:

- OpenAI / Azure / xAI: the GA remap normalizes `tools[].parameters` recursively (lowercases schema types at any depth, strips provider-only keys like `behavior`, flattens `functionDeclarations` and chat-style tool shapes), strips non-GA `turn_detection` keys per VAD type, drops empty `transcription` configs, and collapses dict voices to the GA string. Applies to both flat beta fields and GA-nested client input.
- Bedrock: the tool transform runs the same canonicalization before serializing for Nova Sonic (this also fixed GA flat tools, which previously produced empty tool names).
- Gemini / Vertex: outbound tools go through `_build_vertex_schema` (recursive); canonical snake_case `context_window_compression` is converted to Gemini's camelCase (`sliding_window` -> `slidingWindow`).

The risk map above remains the checklist for extending the normalizer when providers add new nested surface.

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
| `context_window_compression` | dropped (see note) | yes (compression) | dropped |

Bedrock Nova Sonic is the least complete: it ignores `turn_detection` and `input_audio_transcription` and hardcodes modalities. If a backend does not map a field, it is dropped rather than forwarded, so a canonical payload never breaks a session.

## Where the mapping lives

| Family | Providers | Mapping code |
|---|---|---|
| OpenAI-compatible | `openai`, `azure`, `xai` | `litellm/litellm_core_utils/realtime_streaming.py` (`_remap_beta_session_to_ga`) plus per-provider event normalizers (e.g. `litellm/llms/xai/realtime/transformation.py`, `XAIRealtimeNormalizer`). `_remap_beta_session_to_ga` only runs on this passthrough path (`provider_config is None`) — its `GA_SESSION_ALLOWED_KEYS` allowlist is the OpenAI GA schema and would otherwise strip union fields (`temperature`, `top_p`, `top_k`, `context_window_compression`) before they ever reach Gemini's or Bedrock's own mapping code. |
| Gemini | `gemini`, `vertex_ai` | `litellm/llms/gemini/realtime/transformation.py` (Vertex extends `GeminiRealtimeConfig`) |
| Bedrock | `bedrock` | `litellm/llms/bedrock/realtime/transformation.py` |
| Shared outbound correlation | `xai`, `gemini`/`vertex_ai`, `bedrock` | `litellm/litellm_core_utils/realtime_correlation/` — the `(response_id, item_id, output_index, content_index)` index/lifecycle machinery all three call into so they don't each reimplement it (and diverge) independently. See its own README for the full event-by-event contract. Not used by `openai`/`azure`, which need no reconstruction (see "Server events" below). |

When you add a provider or a new canonical field, the rule is to extend the mapping inside the provider implementation (or the shared remap for OpenAI-compatible backends), never to push provider-specific shapes onto clients.

How each family drops unsupported fields:

- OpenAI-compatible: after `_remap_beta_session_to_ga` rewrites the flat beta fields into their GA nested form, it keeps only keys in `GA_SESSION_ALLOWED_KEYS`. That allowlist is derived at import time from the installed openai SDK (`RealtimeSessionCreateRequest.model_fields`) with a hardcoded fallback, and a test asserts the two stay in sync so the allowlist cannot silently drift when the SDK updates. Anything outside the GA schema (the union extensions) is dropped here.
- Gemini / Vertex: `get_supported_openai_params` is the allowlist; `map_openai_params` only maps listed keys and ignores the rest.
- Bedrock: maps a fixed subset of canonical fields and ignores the rest by construction.

## Server events: the outbound contract

The contract is bidirectional. Inbound, clients send one canonical `session.update`; outbound, clients receive one canonical event stream — the OpenAI Realtime server events (`session.created`, `response.output_audio.delta`, `conversation.item.input_audio_transcription.completed`, `response.done`, and so on). A client never sees a provider-native event shape, regardless of backend.

How each provider family honors this:

- Gemini / Vertex AI: full re-synthesis. Canonical events are constructed from scratch out of Gemini Live frames (`setupComplete` becomes `session.created`, `usageMetadata` becomes canonical usage, model path prefixes are stripped, modalities are lowercased). Unknown native frames are dropped, so nothing Gemini-shaped can leak.
- Bedrock Nova Sonic: full re-synthesis from Nova Sonic events; unknown events are dropped.
- OpenAI / Azure: the backend already speaks the canonical format; events pass through, with GA-to-beta event-name translation only for clients that connected with the beta header.
- xAI: passthrough with a targeted normalizer (rewrites `role: "tool"` to `"assistant"` on function-call items, injects missing indices via the shared correlation tracker described below, normalizes usage) plus a structural event-type allowlist: any event whose type is outside the canonical GA server-event vocabulary (derived from the openai SDK, ~46 types) is dropped, so unknown provider-native events cannot leak by construction. Payload-level deviations inside known event types still rely on the targeted fixes; treat any xAI-shaped surprise there as a bug in `XAIRealtimeNormalizer`.

### Correlation keys and `response.done.output` completeness

Every streaming event carries a `(response_id, item_id, output_index, content_index)` correlation key so a client can tell which phrase/item a delta or "done" signal belongs to — this matters most during barge-in, when an old response may not be fully closed yet while a new one starts. xAI, Gemini/Vertex, and Bedrock all build this key through the shared `litellm/litellm_core_utils/realtime_correlation/` module instead of each computing it independently (previously a source of provider-specific bugs — hardcoded `output_index=0`, and `response.done.output` silently empty or missing barge-in items). Guarantees that hold for all three:

- `output_index` / `content_index` are real, monotonically increasing values scoped to their response/item, never a hardcoded constant.
- `response.done.output` always contains every item the response ever opened, each with a `completed` or `incomplete` status — an item interrupted mid-phrase by barge-in still appears (as `incomplete`), it does not silently disappear.
- Depth of integration differs slightly per provider: Gemini's tool-call path builds its whole event sequence through the shared module's `tool_call_events()`; xAI and Bedrock's tool-call paths use only the module's index tracking, not full event construction. See `litellm/litellm_core_utils/realtime_correlation/README.md` for the exact per-provider breakdown and the full event-by-event contract.
- OpenAI / Azure need none of this — the backend already emits correct indices and a complete `response.done.output`, so there is no reconstruction layer for this family at all.

Capability discovery: these differences are machine-readable. Realtime models in the registry (`model_prices_and_context_window.json`) carry `supports_native_transcription`, `supports_turn_detection` and `supports_sampling_params`; clients can query them via `/model/info` before opening a session, and code can use `litellm.supports_native_transcription(model)` / `litellm.supports_turn_detection(model)`. Consult these instead of discovering capabilities by observing which events arrive.

Emission conditions differ per provider even for canonical events. The one that surprises people most: `conversation.item.input_audio_transcription.completed`. Gemini Live produces input transcription natively once enabled in setup. OpenAI runs it as a separate ASR process that is off by default; the event fires only when `audio.input.transcription` is configured with a valid transcription model and the audio buffer is committed (by VAD or explicitly), and it arrives asynchronously, possibly after `response.done`. Bedrock does not emit it at all. An identical client therefore hears this event on Gemini, only-with-config on OpenAI, and never on Bedrock; that is a capability difference, not an event-mapping bug.

Rules for contributors on the outbound side mirror the inbound ones: a new provider's event mapping must re-synthesize canonical events (prefer Gemini's re-synthesis approach over xAI's blacklist), drop what it cannot map, and never forward a provider-native event or field to the client.

### Backend reconnection and `litellm.session.*` events

The client-to-proxy WebSocket is the session; it stays up for the session's lifetime. The proxy-to-backend WebSocket is an implementation detail and may be re-established mid-session. Reconnects are triggered by a dropped backend socket (`connection_closed`) or a provider's advance warning (Gemini `goAway`), and recover in one of two modes: `native`, using the provider's own resumption token (Gemini `sessionResumptionUpdate` handle), or `fresh`, opening a new backend session and replaying the accumulated conversation transcript (user and assistant turns, with barge-in notes) to restore context.

Clients observe this through two proxy-emitted events that extend the canonical stream: `litellm.session.reconnecting` `{reason}` followed by `litellm.session.reconnected` `{resumed: "native" | "fresh" | "replayed"}`. Handle them (at minimum, ignore them without crashing on the unknown `litellm.`-prefixed type); do not treat them as provider events.

Current support:

- Gemini / Vertex AI: full support. `native` mode via the `sessionResumptionUpdate` handle when available, `fresh` mode with provider-format transcript replay otherwise; `goAway` triggers a proactive reconnect before the drop.
- OpenAI / Azure / xAI: `fresh` mode. There is no native resumption in the OpenAI GA protocol, so on a dropped backend socket the proxy opens a new one, re-sends the last client `session.update` (cached in its final GA form) to restore configuration, replays the accumulated transcript as canonical `conversation.item.create` items (user turns as `input_text`, assistant turns as `output_text`, prefixed with a context-restored note), and swallows the duplicate `session.created` so the client only sees the `litellm.session.*` pair.
- Bedrock Nova Sonic: not supported, by transport. AWS exposes realtime as `InvokeModelWithBidirectionalStream` (SigV4-signed HTTP/2 event stream, not a WebSocket), so Bedrock bypasses the shared WebSocket machinery entirely; a dropped backend stream ends the client session. A Bedrock-specific reconnect (re-issuing sessionStart/promptStart and replaying history inside its handler) is possible but is a separate implementation, not a configuration of the shared one.

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

Worked example, `context_window_compression`. OpenAI's native `truncation` (an enum-like strategy that drops old context) and Gemini's `contextWindowCompression` (sliding-window compression / summarization) address the same topic, keeping the context window from overflowing, with different behavior. The intended unification under the canonical `context_window_compression` is by intent, with these documented degradations:

- Gemini / Vertex: honored today. Enables `contextWindowCompression`, which compresses rather than hard-drops context, so a request for exact truncation becomes compression, not deletion.
- OpenAI: not yet mapped. There is no remap from canonical `context_window_compression` to native `truncation`, so on OpenAI-compatible backends the field is currently dropped by the GA allowlist. Mapping it to `truncation` is the intended next step; a client that needs OpenAI truncation today should send the native `truncation` field, which is GA-valid and passes through.
- Bedrock Nova Sonic: no equivalent; dropped.

Because both the drop and the compression-vs-truncation difference are silent at runtime, they are documented here so clients can reason about cross-provider behavior. This entry is also a live example of the policy: the canonical name is reserved by intent, but the matrix and this note record what actually happens per provider today, not the aspiration.

## Provider-specific behavior

Deviations from a naive reading of the contract, per provider. Keep this list in sync with the mappings.

- Bedrock Nova Sonic: ignores `turn_detection` and `input_audio_transcription`; hardcodes modalities to `["text","audio"]`; ignores `top_k`. The `sessionStart.inferenceConfiguration` block (`maxTokens`, `topP`, `temperature`) and `promptStart.audioOutputConfiguration.voiceId` are required-by-protocol: the AWS bidirectional-stream event schema presents them as part of the fixed event structure with no optional marker, so the implementation always sends them and falls back to the values from the AWS documentation examples (`maxTokens: 1024`, `topP: 0.9`, `temperature: 0.7`, `voiceId: "matthew"`) when the client omits them. This is the sanctioned required-by-protocol default, not a litellm opinion. Source: https://docs.aws.amazon.com/nova/latest/userguide/input-events.html
- OpenAI GA: has no session-level `temperature` / `top_p` / `top_k`; these union fields are dropped by the GA allowlist. `context_window_compression` is likewise dropped today (no remap to native `truncation` yet); send native `truncation` if you need it on OpenAI. Native GA fields (`instructions`, `tools`, `tool_choice`, `truncation`, `prompt`, `include`, `tracing`) pass through unchanged.
- Gemini / Vertex AI: audio formats are fixed by the Live model rather than taken from `input_audio_format` / `output_audio_format`. `context_window_compression` compresses context rather than hard-truncating it.

## Rules for contributors

- Client-facing surface is this contract only. Never require a client to send a provider-native shape.
- All mapping lives in the provider implementation (or the shared OpenAI-compatible remap). Adding a provider means adding its mapping there, not changing the client contract.
- A field the provider cannot honor is dropped silently; it must not be forwarded to the backend, and it must be recorded in the support matrix and, if its behavior is surprising, in "Provider-specific behavior".
- Do not hardcode a default for an optional field. If the client omits an optional field, do not send it and let the backend default apply. Write a default only for a field the provider's protocol makes mandatory, and use the provider's native default value.
- Before adding or unifying a field, enforce the one-name-one-meaning invariant: never reuse a canonical name for a different concept and never let a name's value semantics change per provider. If two things cannot share one meaning, give them two names.
- Before unifying a new pair of fields, apply the semantic aliasing rule above. When in doubt, keep them separate.
- Any new union field must land in the field reference (with `Origin: union`), the support matrix, and, if it degrades unevenly, the provider-specific section.
- The top-level allowlist does not protect nested structures. When a field carries a nested object (`tools[].parameters`, `turn_detection`, `voice`, `input_audio_transcription`, `context_window_compression`), the mapping must normalize that structure at every depth into the target provider's form, not just rename the top-level key. A single foreign nested key rejects the whole `session.update`. See "Nested structures have no safety net".
