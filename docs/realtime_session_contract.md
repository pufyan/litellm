# LiteLLM Realtime session contract

This is the single client-facing contract for the LiteLLM proxy `/v1/realtime` WebSocket. Clients send one provider-independent schema; the proxy maps it to whichever backend serves the requested model. You never send a provider-specific shape, and you never need to know which backend is behind the model you asked for.

## The rule that shapes this document

**The contract is provider-agnostic. Everything provider-specific lives in the provider implementation.**

That is not a stylistic preference; it is what makes the contract stable. A consequence you should read carefully, because it is stronger than it first appears:

> This document does not say which provider supports which field, and it never will. Not in a matrix, not in per-field notes, not in a "provider-specific behavior" appendix.

Adding a provider, or a provider adding a feature, must not require editing this file. If a change to a backend forces a change here, the layering is wrong.

What each side owns:

- **This contract** owns the field names, their types, their value ranges, and their *meaning*. One name, one meaning, everywhere.
- **The provider implementation** owns everything else: whether the backend understands a field at all, whether the answer differs between that backend's models, what the field is called natively, how nested shapes are rewritten, and which native default applies.

So a client sends the full canonical payload for the behavior it wants and does not branch on the backend. Each field is then honored, adapted, or dropped by the implementation. All three outcomes are normal and none of them break the session.

### What the implementation must do with each field

Every provider implementation resolves each canonical field into exactly one of these:

1. **Map it.** The backend has an equivalent. Translate the name, the value shape, the units, and any nesting into the native form.
2. **Adapt it to the model.** The backend supports the concept, but the details differ across its own models. The implementation branches on the model and sends the right native form. It must branch on the *model*, not only the provider — a field valid on one model of a backend and rejected by another must never be forwarded blindly.
3. **Drop it.** The backend has no equivalent. Remove the field entirely and log it. Never forward it, never approximate it with an unrelated native field.

Dropping is a first-class outcome, not a failure. It is what guarantees the central invariant below.

### Invariants

- **A canonical payload never breaks a session.** Any backend, any model, any combination of fields. Since unsupported fields are removed before the request leaves the proxy, no backend can reject a field it does not know.
- **Silent to the client, visible to the operator.** A dropped field produces no error event and no echo. The proxy logs one warning per `session.update` naming every dropped key and why: `unsupported_by_provider` (the backend has no equivalent), `unsupported_by_model` (the backend has it, this model does not), `mutually_exclusive` (another field expressing the same intent won), or `not_canonical` (the key is not part of this contract at all). Diagnosis is an operator concern, read from proxy logs.
- **Optional fields get no litellm defaults.** If the client omits an optional field, the implementation does not send it, and the backend's own default applies. The one exception is a field the backend's protocol makes *mandatory*: then the implementation must fill it with that provider's documented native default, never an arbitrary litellm choice. Hardcoding a default for an optional inference parameter silently overrides the vendor and drifts when the vendor changes.
- **One field, one address.** Every canonical field has exactly one client-facing name and location. Provider-native aliases are dropped, never merged.

Clients that need to know a backend's real capabilities before opening a session read the model registry (`/model/info`), not this document. See "Capability discovery".

## Connecting

```
wss://<proxy-host>/v1/realtime?model=<model_name>
Authorization: Bearer <litellm_virtual_key>
```

The model is selected by the `model` query parameter, not by a `session.model` field. A `session.model` value is optional and is resolved by the provider implementation.

Do not send the `OpenAI-Beta: realtime=v1` header. Without it the proxy runs in GA mode and applies the canonical mapping described here. Sending that header forces legacy beta passthrough and disables the mapping, so canonical fields would reach the backend unmapped and be rejected. Keep the header off unless you specifically need legacy beta event names and are only sending beta-valid fields.

## Canonical `session.update`

Configure the session by sending a `session.update` event with the flat schema below.

```json
{
  "type": "session.update",
  "session": {
    "modalities": ["audio"],
    "instructions": "You are a helpful voice assistant",
    "voice": "marin",
    "language": "ru-RU",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "output_audio_speed": 1.0,
    "input_audio_transcription": { "model": "whisper-1" },
    "output_audio_transcription": {},
    "input_audio_noise_reduction": { "type": "near_field" },
    "turn_detection": {
      "type": "server_vad",
      "threshold": 0.5,
      "prefix_padding_ms": 300,
      "silence_duration_ms": 500,
      "interrupt_response": true
    },
    "tools": [],
    "tool_choice": "auto",
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "stop_sequences": [],
    "max_response_output_tokens": 4096,
    "thinking_level": "low",
    "include_thoughts": false,
    "context_window_compression": { "sliding_window": {}, "trigger_tokens": 25600 },
    "session_resumption": { "enabled": true }
  }
}
```

Send everything you want; omit what you do not care about. Omission means "use the backend default", which is not the same as sending an explicit value.

### Field reference

The type and the meaning are normative. Where a range is given, it is the canonical range; an implementation clamps or rescales to its backend's range as part of mapping.

#### Core session

| Field | Type | Meaning |
|---|---|---|
| `modalities` | `["audio"]` \| `["text"]` \| `["audio","text"]` | Which output modalities the model may produce. Audio mode still delivers transcripts via events |
| `instructions` | string | System prompt for the session |
| `voice` | string | Voice identifier for synthesis. Values are backend vocabularies; the implementation validates |
| `language` | string (BCP-47) | Language hint for recognition and synthesis, e.g. `ru-RU` |
| `tools` | array | Tools available to the model. See "Tools" |
| `tool_choice` | `"none"` \| `"auto"` \| `"required"` \| object | Tool selection policy |
| `max_response_output_tokens` | int \| `"inf"` | Maximum output tokens per assistant response |

#### Audio

| Field | Type | Meaning |
|---|---|---|
| `input_audio_format` | string \| object | Format of audio the client sends, e.g. `pcm16`, `g711_ulaw` |
| `output_audio_format` | string \| object | Format of audio the client wants back |
| `output_audio_speed` | number | Playback speed multiplier for synthesized audio; `1.0` is natural rate |
| `input_audio_transcription` | object | Configuration for transcribing the *user's* audio. An empty object means "enable with backend defaults". See the default rule below |
| `output_audio_transcription` | object | Configuration for transcribing the *model's own* audio output. See the default rule below |
| `input_audio_noise_reduction` | object \| null | Noise filtering applied to input audio before recognition. `null` disables |
| `transcription_keyterms` | string[] | Domain terms to bias speech recognition toward — product names, jargon, proper nouns that a general recognizer mishears. A list of terms, not a sentence of instructions |

**Transcription defaults.** The outbound contract promises canonical transcript events, so transcription is not left to chance — but whether it is on by default depends on what it costs, which differs by direction and by backend:

- **Output transcription is always on.** The model knows what it said; producing that transcript costs nothing extra on any backend. Where a backend emits it automatically there is nothing to configure and the canonical field is dropped as having nothing to map to; where it must be requested, the implementation requests it.
- **Input transcription is on where it is native, off where it is a separate process.** A backend that transcribes the user natively has no separate model to choose and no extra bill, so it is enabled and there is nothing to turn off. A backend that runs input transcription as a distinct ASR — needing an explicit model, extra cost and extra latency — leaves it off until the client asks, because switching it on unasked spends the client's money.

An implementation therefore does not read these fields as plain optional inputs. It applies the rule for its backend and, where the answer is fixed, says so rather than pretending the field is a knob.

#### Turn detection

One object describing when a user turn ends and how the model reacts.

| Field | Type | Meaning |
|---|---|---|
| `type` | `"server_vad"` \| `"semantic_vad"` \| `null` | Detection strategy. `server_vad` is acoustic, `semantic_vad` uses a model to judge completion, `null` means the client controls turns manually |
| `threshold` | number 0..1 | Acoustic activation threshold; higher is less sensitive |
| `prefix_padding_ms` | int | Audio retained before detected speech start |
| `silence_duration_ms` | int | Silence required before the turn is considered over |
| `idle_timeout_ms` | int | Time after an assistant response before the backend re-engages |
| `start_sensitivity` | `"high"` \| `"medium"` \| `"low"` | How eagerly speech *start* is detected |
| `end_sensitivity` | `"high"` \| `"medium"` \| `"low"` | How eagerly speech *end* is detected. `high` reacts fast but may clip slow speakers |
| `eagerness` | `"low"` \| `"medium"` \| `"high"` \| `"auto"` | For `semantic_vad`: how quickly the model decides the user is done |
| `create_response` | bool | Whether end-of-turn automatically triggers a response |
| `interrupt_response` | bool | Whether user speech interrupts in-progress model audio (barge-in) |
| `turn_coverage` | `"activity_only"` \| `"all_input"` \| `"audio_activity_and_all_video"` | What a turn contains: only what the detector marked as activity, everything received, or a mix — audio limited to detected activity while video is kept whole. Backends pick their own default when this is omitted, and that default may differ between their models |

`start_sensitivity` / `end_sensitivity` and `threshold` overlap in intent but not in shape: the first pair is a coarse enum, the second a continuous acoustic knob. Send whichever your intent is expressed in; each implementation maps what its backend has and drops the other.

#### Sampling

| Field | Type | Meaning |
|---|---|---|
| `temperature` | number | Sampling temperature |
| `top_p` | number 0..1 | Nucleus sampling |
| `top_k` | int | Top-k sampling |
| `presence_penalty` | number | Penalty for tokens already present |
| `frequency_penalty` | number | Penalty scaled by token frequency |
| `stop_sequences` | string[] | Sequences that end generation |
| `candidate_count` | int | Number of response candidates to generate |

#### Reasoning

Two fields, deliberately **not** unified: one is a token budget, the other an effort level. They are different concepts, and no known model accepts both.

Sending both is **not** a client error. Resolving them is the proxy's job, like every other mapping decision: the implementation keeps whichever field the active model expresses reasoning in, drops the other with a `mutually_exclusive` log, and the session proceeds. A client that stores both in configuration and forwards both is behaving correctly — it is precisely the case the contract exists to absorb, since knowing which of the two a given model wants is backend knowledge the client is not supposed to have.

| Field | Type | Meaning |
|---|---|---|
| `thinking_budget` | int | Token budget for internal reasoning. `0` disables reasoning |
| `thinking_level` | `"minimal"` \| `"low"` \| `"medium"` \| `"high"` | Reasoning effort as a level rather than a token count |
| `include_thoughts` | bool | Whether reasoning summaries are emitted to the client |

An implementation whose backend expresses effort with a coarser scale maps the canonical levels onto it and documents the collapse in its own code, not here.

#### Session lifetime and context

| Field | Type | Meaning |
|---|---|---|
| `context_window_compression` | object | How to keep the context window from overflowing: `{ "sliding_window": {}, "trigger_tokens": int, "target_tokens": int }`. Expresses the intent "do not let the session die of context overflow"; a backend may satisfy it by compressing, summarizing, or truncating |
| `session_resumption` | object | `{ "enabled": bool }`. Asks the backend to make the session resumable across a dropped connection. The proxy's own reconnection machinery is always active regardless; what this field changes is *how much context survives* a reconnect — see `resumed: "native"` under "Backend reconnection". Any resumption token the backend issues is held and used by the proxy, never surfaced to the client |
| `media_resolution` | `"low"` \| `"medium"` \| `"high"` | Resolution at which visual input is processed |

#### Conversational style

| Field | Type | Meaning |
|---|---|---|
| `affective_dialog` | bool | Whether the model adapts tone and delivery to the emotional content of user speech |
| `proactive_audio` | bool | Whether the model may decide on its own not to respond to input not addressed to it |

### Tools

`tools` is one array holding both user-defined functions and backend built-in tools. Built-in tools are typed entries, never boolean session flags — otherwise there would be two ways to declare a tool.

| Entry | Meaning |
|---|---|
| `{ "type": "function", "name", "description", "parameters" }` | A client-implemented function. `parameters` is JSON Schema |
| `{ "type": "web_search" }` | Backend-side web search |
| `{ "type": "code_execution" }` | Backend-side code execution |
| `{ "type": "file_search", "vector_store_ids": [...] }` | Retrieval over an indexed corpus |
| `{ "type": "mcp", "server_label", "server_url", "authorization", "require_approval" }` | An MCP server the backend connects to |

A `tools[]` entry whose type the active backend cannot provide is dropped like any other unsupported field. Function tools are always available.

### Rejected inbound forms

These are dropped rather than interpreted, so that each canonical field keeps exactly one address:

- Any nested `audio` block (`audio.output.voice`, `audio.input.turn_detection`, `audio.*.format`). The proxy builds the backend's nested form itself from the flat fields. What is rejected here is the *address*, not nesting as such: `input_audio_format` may still take an object as its value, because that object is the value of a canonical field rather than a second place to put one.
- Renamed variants of canonical keys, e.g. `output_modalities` or `max_output_tokens` instead of `modalities` and `max_response_output_tokens`.
- Any backend-native key, in any backend's spelling.

A client written directly against a backend's own session shape must switch to the flat canonical form. This is deliberate: accepting both forms would mean two addresses for one field, and the merge order would become an undocumented behavior.

Outbound is separate: server events follow the canonical GA event vocabulary, so configuration *reads* come from the `session.created` / `session.updated` echo while configuration *writes* use only the flat canonical schema.

## Nested structures have no safety net

The single most important operational rule for implementers.

Top-level canonical keys are handled by a single shared step: an unsupported one is dropped before the request leaves the proxy. **Nested objects have no equivalent single choke point.** A key inside `tools[].parameters`, `turn_detection`, or `context_window_compression` reaches the backend verbatim unless the implementation rewrites it, and a backend that does not recognize it rejects the entire `session.update` — taking the system prompt and tool configuration down with it.

This is a statement about where the work has to happen, **not** a caveat on the invariant. The client's obligation is unchanged: send canonical shapes at every depth and nothing else. Meeting that obligation must be sufficient, so an implementation that lets a canonical nested payload reach its backend unrewritten and break a session has a bug — the contract is not being honored, and "the client sent a nested object" is never the diagnosis.

The asymmetry is in the mechanism, not in the guarantee: the top level is protected by one shared allowlist, while every nested structure must be normalized by each implementation, recursively, at every depth, including value case, key names, and container shape. That is more code and more places to forget, which is exactly why this section exists.

Structures that carry this risk, highest first:

1. **`tools[].parameters`** (JSON Schema). Recursive, so the risk repeats at every depth under `properties.*`, `items`, `additionalProperties`. Backends differ in type-name case, in which JSON Schema keywords they accept, in vendor-specific keys, and in the container shape of the tool entry itself (flat object vs a declarations wrapper vs a stringified schema).
2. **`turn_detection`**. Its fields span several backend concepts; each implementation must translate the subset its backend has and strip the rest. Canonical field names must never reach a backend verbatim.
3. **`input_audio_transcription`** / **`output_audio_transcription`**. Backends disagree on whether an empty object means "enable with defaults" or is invalid.
4. **`voice`**. Canonically a string; some backends want a structured object.
5. **`context_window_compression`**. Nested and naming-sensitive.

When a backend adds new nested surface, extend the shared normalization helpers rather than writing per-provider parsing inline.

## Capability discovery

This contract deliberately says nothing about which backend honors which field, so a natural question follows: how does anyone find out? There are two mechanisms, they answer different questions, and confusing them is the usual mistake.

**The model registry answers "what can this model do?" — before a session exists.**

Realtime models in `model_prices_and_context_window.json` carry capability flags, readable programmatically through `/model/info` or helpers such as `litellm.supports_native_transcription(model)` and `litellm.supports_turn_detection(model)`. This exists for clients that must *change their own behavior* based on the answer: a client that would otherwise run its own ASR needs to know whether the backend transcribes natively, because the decision is made before any audio flows.

The registry is the right home for such facts because it is per-model, machine-readable, and versioned alongside the implementation that changes. When an implementation gains or loses a capability, the registry entry moves with it and this document stays untouched — which is the whole point of keeping support out of the contract.

Its coverage is deliberately narrow. The registry describes capabilities that change *what a correct client does*, not every field in this contract. Most fields need no flag: sending `top_k` to a backend without top-k sampling is harmless, so a client has no decision to make and nothing to look up. Do not expect a registry flag per canonical field, and do not treat the absence of a flag as evidence a field is unsupported.

**Proxy logs answer "what actually happened to my configuration?" — after a session ran.**

Every dropped field is named in a warning log with its reason. This is diagnosis, not discovery: it tells an operator that a requested behavior never took effect, which is the failure mode this contract's silence would otherwise hide. It is read by humans investigating a session, not by client code deciding what to send.

**Neither is the event stream.** Clients must not infer capabilities by observing which events arrive. Emission conditions vary for reasons unrelated to support — a transcription event may be absent because transcription is off, because the buffer was never committed, or because it arrives later than expected — so absence of an event proves nothing about the backend.

## Server events: the outbound contract

The contract is bidirectional. Inbound, one canonical `session.update`; outbound, one canonical event stream — the OpenAI Realtime server events (`session.created`, `response.output_audio.delta`, `conversation.item.input_audio_transcription.completed`, `response.done`, and so on). **A client never sees a backend-native event shape.**

Implementations satisfy this by re-synthesizing canonical events from their backend's native stream and dropping native frames they cannot map. Passthrough is acceptable only where a backend already speaks the canonical vocabulary exactly, and even then an event-type allowlist must guarantee that nothing outside the canonical vocabulary can leak.

### Correlation keys

Every streaming event carries a `(response_id, item_id, output_index, content_index)` correlation key so a client can tell which item a delta or completion belongs to. This matters most during barge-in, when an old response may still be closing as a new one starts.

These are guarantees of the contract, owed to the client on every backend. Where a backend emits them correctly on its own they pass through; where it does not, the implementation is responsible for reconstructing them. A client seeing a hardcoded index or a truncated `response.done.output` is looking at a bug in that implementation, not at a backend limitation it should work around:

- `output_index` / `content_index` are real, monotonically increasing values scoped to their response and item — never a hardcoded constant.
- `response.done.output` contains every item the response ever opened, each with a `completed` or `incomplete` status. An item interrupted mid-phrase by barge-in still appears, as `incomplete`; it does not silently disappear.

Implementations build these through the shared `litellm/litellm_core_utils/realtime_correlation/` module rather than each computing indices independently, which was historically a source of per-backend bugs. See that module's README for the event-by-event contract.

### Response triggering

`conversation.item.create` does **not** start generation. An explicit `response.create` is required, for greetings and for tool results alike; without it the model stays silent until the next user turn. Some backends start generating on their own native "turn complete" signal, but that signal is not part of this contract — clients migrating from a direct backend integration must add the explicit `response.create`.

### Event emission is not uniform

Canonical events are canonical in *shape*, not in *when they fire*. `conversation.item.input_audio_transcription.completed` is the usual surprise: depending on the backend it may arrive natively, only when transcription is explicitly configured, asynchronously after `response.done`, or never. That is a capability difference, discoverable through the registry, not an event-mapping bug.

### Backend reconnection

The client-to-proxy WebSocket is the session and stays up for its lifetime. The proxy-to-backend connection is an implementation detail and may be re-established mid-session, triggered by a dropped socket or a backend's advance warning.

Clients observe this through two proxy-emitted events extending the canonical stream:

- `litellm.session.reconnecting` `{reason}`
- `litellm.session.reconnected` `{resumed: "native" | "fresh" | "replayed"}`

The three `resumed` values are not interchangeable; they tell a client how much context survived:

- `native` — the backend's own resumption mechanism restored the session, including server-side context. This is the outcome `session_resumption` asks for: requesting it is what makes `native` possible at all, and without it (or on a backend that has no such mechanism) a reconnect can only fall back to one of the two below.
- `replayed` — a new backend session was opened and the accumulated transcript was re-sent, so the conversation context is restored as text. Not identical to `native`: anything the backend held internally beyond the transcript is gone.
- `fresh` — a new backend session was opened and nothing was replayed, because there was no transcript to replay or the backend cannot accept one. The model starts with no memory of the conversation.

A client that behaves differently depending on whether context survived should branch on these values. Which modes are reachable at all depends on the backend and is an implementation matter.

Handle both events — at minimum, ignore unknown `litellm.`-prefixed types without crashing. Do not treat them as backend events. Not every transport can reconnect at all; where it cannot, a dropped backend connection ends the client session.

## Semantic aliasing policy

### Hard invariant: one name, one meaning

The canonical namespace must stay collision-free. A field name means exactly one thing across every backend. Two backends must never read the same canonical key as two different concepts, and a key's value semantics must never change depending on which backend is active.

This forbids:

- Reusing a canonical name for a new, unrelated concept because a new backend happens to spell it that way. Pick a new name.
- A field whose value shape or unit shifts per backend — one reading `temperature` as 0..1 and another as 0..2 under the same name. If the meaning cannot be made identical, use two names.

A field may vary in *how completely* a backend honors it. It must never vary in *what it means*. This is why `thinking_budget` and `thinking_level` stay separate: a token count and an effort level are different concepts, and one field whose type changed per model would break the invariant outright.

### Unifying fields with different names

Backends often express the same intent under different names. Unifying those is the point of the contract. The test:

> Unify two backend fields under one canonical field only if, for a **typical** value, the user's intent and the observable result match on every backend. Divergence at edge values is acceptable. If only the topic matches while the base behavior differs, keep them separate.

Unify the intent, not the JSON key. A canonical field is a contract about meaning; each implementation interprets that meaning in its backend's terms and degrades in a documented way for values it cannot honor — documented in the implementation, not here.

Worked example: `context_window_compression`. One backend may hard-truncate old context, another may compress or summarize it. Both serve the same intent — do not let the session die of context overflow — so they unify. The canonical field therefore promises the *intent*, and its docstring above deliberately says "compressing, summarizing, or truncating" rather than naming one mechanism.

Counter-example: `eagerness` and `end_sensitivity`. Both concern turn-end timing, but one describes the model's willingness to respond and the other the detector's sensitivity to silence, and they are not interchangeable at typical values. Same topic, different behavior, so they stay separate fields.

## Rules for contributors

**Contract changes**

- The client-facing surface is this contract only. Never require a client to send a backend-native shape.
- Never add a provider name, a model name, or a support matrix to this document. If you feel the need to write "on backend X this field means…", the field is wrong: either its meaning is not universal (split it) or the note belongs in the implementation.
- Before adding a field, enforce one-name-one-meaning. Before unifying two backend fields, apply the aliasing test. When in doubt, keep them separate.
- A new field lands in the field reference with a type and a *meaning*, expressed without reference to any backend.
- A field is worth adding as soon as one backend can honor it. The contract is the union of intents, not the intersection.

**Implementation changes**

- All mapping lives in the provider implementation. Adding a provider means adding its mapping there and touching nothing in this file.
- Resolve every canonical field into map / adapt-to-model / drop. Never forward a field the backend does not know, and never approximate it with an unrelated native field.
- Branch on the model, not only the provider. A field valid on one model of a backend and rejected by another must not be forwarded blindly.
- Log every drop with its reason. An unlogged drop is indistinguishable from a working configuration — which is exactly how a misconfigured production session goes unnoticed.
- Write a default only for a field the backend's protocol makes mandatory, using that backend's documented native default. Never default an optional field.
- Normalize nested structures at every depth into the backend's form. A single foreign nested key rejects the whole `session.update`.
- Re-synthesize outbound events into the canonical vocabulary and drop what you cannot map. Never forward a backend-native event or field to the client.
- Add a model registry capability flag only when a client must change its own behavior based on the answer, and keep any flag you add in sync with what your implementation actually does. Most fields need no flag; a flag per canonical field is not the goal.
- Verify a backend's behavior against its official documentation, and where documentation is inaccessible, against a live session. Do not encode assumptions.
