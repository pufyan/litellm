import { describe, expect, it } from "vitest";
import { filterModelsByMode, findModeMismatches } from "./model_mode_filter";

const catalog = {
  "tts-1": { mode: "audio_speech" },
  "tts-2": { mode: "audio_speech" },
  "chat-1": { mode: "chat" },
  "resp-1": { mode: "responses" },
  "embed-1": { mode: "embedding" },
  "modeless-1": { litellm_provider: "x" },
};
const models = Object.keys(catalog);

describe("filterModelsByMode", () => {
  it("keeps only models of the selected mode plus models without a catalog mode", () => {
    expect(filterModelsByMode(models, catalog, "audio_speech")).toEqual(["tts-1", "tts-2", "modeless-1"]);
  });

  it("lets chat accept responses models and drops other modes", () => {
    expect(filterModelsByMode(models, catalog, "chat")).toEqual(["chat-1", "resp-1", "modeless-1"]);
  });

  it.each(["realtime", "batch", "anthropic_messages"])("does not narrow the list for %s", (mode) => {
    expect(filterModelsByMode(models, catalog, mode)).toEqual(models);
  });

  it.each([undefined, null, ""])("does not narrow the list while the mode is %s", (mode) => {
    expect(filterModelsByMode(models, catalog, mode)).toEqual(models);
  });

  it("returns the whole list when nothing matches the mode", () => {
    expect(filterModelsByMode(["chat-1", "embed-1"], catalog, "audio_speech")).toEqual(["chat-1", "embed-1"]);
  });

  it("returns the whole list when the catalog is missing", () => {
    expect(filterModelsByMode(models, undefined, "audio_speech")).toEqual(models);
  });
});

describe("findModeMismatches", () => {
  it("reports a selected model whose catalog mode does not fit", () => {
    expect(findModeMismatches(["chat-1"], catalog, "audio_speech")).toEqual([
      { model: "chat-1", catalogMode: "chat", selectedMode: "audio_speech" },
    ]);
  });

  it("is empty when modes fit, including chat with responses", () => {
    expect(findModeMismatches(["tts-1"], catalog, "audio_speech")).toEqual([]);
    expect(findModeMismatches(["resp-1"], catalog, "chat")).toEqual([]);
  });

  it("ignores models that are not in the catalog and the special select values", () => {
    expect(
      findModeMismatches(["my-own-model", "custom", "all-wildcard", "modeless-1"], catalog, "audio_speech"),
    ).toEqual([]);
  });

  it("is empty for modes that do not filter and while no mode is chosen", () => {
    expect(findModeMismatches(["chat-1"], catalog, "realtime")).toEqual([]);
    expect(findModeMismatches(["chat-1"], catalog, undefined)).toEqual([]);
  });
});
