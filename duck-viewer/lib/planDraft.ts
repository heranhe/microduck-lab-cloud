import { planFile, type Plan } from "./plans";

type Snapshot = { draft: Plan | null; dirty: boolean };
type SaveRequest = { selection: number; draft: Plan };
export type SaveResult = "saved" | "edited" | "switched";
const fingerprint = (plan: Plan) => JSON.stringify(planFile(plan));

/** Own the latest editor state so async responses never rely on render closures. */
export class PlanDraftEditor {
  private snapshot: Snapshot = { draft: null, dirty: false };
  private baseline = "";
  private selection = 0;
  private listeners = new Set<() => void>();

  getSnapshot = () => this.snapshot;
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  };
  private publish(draft: Plan) {
    this.snapshot = { draft, dirty: fingerprint(draft) !== this.baseline };
    this.listeners.forEach((listener) => listener());
  }
  select(plan: Plan) {
    this.selection += 1;
    this.baseline = fingerprint(plan);
    this.publish(structuredClone(plan));
  }
  switchDraft(plan: Plan, confirmDiscard: () => boolean): boolean {
    if (this.snapshot.dirty && !confirmDiscard()) return false;
    this.select(plan);
    return true;
  }
  patch(fields: Partial<Plan>) {
    if (this.snapshot.draft) this.publish({ ...this.snapshot.draft, ...fields });
  }
  beginSave(): SaveRequest | null {
    return this.snapshot.draft
      ? { selection: this.selection, draft: structuredClone(this.snapshot.draft) }
      : null;
  }
  completeSave(request: SaveRequest, saved: Plan): SaveResult {
    // A response for a different selection must not touch the current editor.
    const current = this.snapshot.draft;
    if (!current || request.selection !== this.selection) return "switched";
    const edited = fingerprint(current) !== fingerprint(request.draft);
    this.baseline = fingerprint(saved);
    // Keep edits made while saving, but adopt the saved identity/revision so
    // the next save updates the right plan (including create and Save as).
    this.publish(edited ? {
      ...structuredClone(saved), ...planFile(current),
      id: saved.id, revision: saved.revision,
    } : structuredClone(saved));
    return edited ? "edited" : "saved";
  }
}
