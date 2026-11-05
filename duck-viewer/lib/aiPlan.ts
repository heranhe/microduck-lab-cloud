import type { Recipe } from "./plans";

export type WeightChange = { key: string; weight: number; was: number | null };

/** What the model changed relative to the recipe's own defaults: a term it added, or one whose weight it moved. */
export function changedWeights(recipe: Recipe | undefined, weights: Record<string, number>): WeightChange[] {
  const base = new Map((recipe?.terms ?? []).map((t) => [t.key, t.weight]));
  return Object.entries(weights)
    .filter(([key, weight]) => !base.has(key) || base.get(key) !== weight)
    .map(([key, weight]) => ({ key, weight, was: base.get(key) ?? null }));
}

/** The person-facing name of a reward term, falling back to its key. */
export function termLabel(recipe: Recipe | undefined, key: string): string {
  const all = [...(recipe?.terms ?? []), ...(recipe?.availableTerms ?? [])];
  return all.find((t) => t.key === key)?.friendly ?? key.replace(/_/g, " ");
}
