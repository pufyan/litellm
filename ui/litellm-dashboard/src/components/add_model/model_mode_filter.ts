export type ModelCatalog = Record<string, unknown> | null | undefined;

export interface ModeMismatch {
  model: string;
  catalogMode: string;
  selectedMode: string;
}

const MODES_WITHOUT_FILTERING = new Set(["realtime", "batch", "anthropic_messages"]);
const COMPATIBLE_CATALOG_MODES: Record<string, ReadonlySet<string>> = {
  chat: new Set(["chat", "responses"]),
};
const NON_CATALOG_MODEL_VALUES = new Set(["custom", "all-wildcard"]);

const catalogModeOf = (catalog: ModelCatalog, model: string): string | undefined => {
  const entry = catalog?.[model];
  if (entry === null || typeof entry !== "object") {
    return undefined;
  }
  const mode = (entry as { mode?: unknown }).mode;
  return typeof mode === "string" && mode !== "" ? mode : undefined;
};

const fitsMode = (catalogMode: string | undefined, selectedMode: string): boolean => {
  if (catalogMode === undefined) {
    return true;
  }
  const compatible = COMPATIBLE_CATALOG_MODES[selectedMode];
  return compatible ? compatible.has(catalogMode) : catalogMode === selectedMode;
};

const isFilteringMode = (mode: string | null | undefined): mode is string =>
  typeof mode === "string" && mode !== "" && !MODES_WITHOUT_FILTERING.has(mode);

export const filterModelsByMode = (
  models: string[],
  catalog: ModelCatalog,
  mode: string | null | undefined,
): string[] => {
  if (!isFilteringMode(mode)) {
    return models;
  }
  const matching = models.filter((model) => fitsMode(catalogModeOf(catalog, model), mode));
  return matching.length > 0 ? matching : models;
};

export const findModeMismatches = (
  selectedModels: string[],
  catalog: ModelCatalog,
  mode: string | null | undefined,
): ModeMismatch[] => {
  if (!isFilteringMode(mode)) {
    return [];
  }
  return selectedModels
    .filter((model) => !NON_CATALOG_MODEL_VALUES.has(model))
    .flatMap((model) => {
      const catalogMode = catalogModeOf(catalog, model);
      return catalogMode !== undefined && !fitsMode(catalogMode, mode)
        ? [{ model, catalogMode, selectedMode: mode }]
        : [];
    });
};
