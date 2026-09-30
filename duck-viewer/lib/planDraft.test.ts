import { describe, expect, it, vi } from "vitest";
import { PlanDraftEditor } from "./planDraft";
import type { Plan } from "./plans";

const plan = (title = "original"): Plan => ({
  version: 1, id: "plan-a", revision: 1, title, description: "goal", success: "hold",
  robot: "microduck", source: "manual", cloud: null,
  local: { behavior: "stand", steps: 100000, weights: { calm: 1 }, stageSteps: {}, stageWeights: {} },
});
const deferred = <T>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
};

describe("plan editor async responses", () => {
  it("keeps edits made while the save response is pending and advances the revision", async () => {
    const editor = new PlanDraftEditor(); editor.select(plan());
    const request = editor.beginSave()!;
    const response = deferred<Plan>();
    const saving = response.promise.then((saved) => editor.completeSave(request, saved));
    editor.patch({ title: "new unsaved title", local: { ...plan().local!, steps: 200000 } });
    response.resolve({ ...request.draft, revision: 2, path: "/plans/plan-a/v2.json" });
    expect(await saving).toBe("edited");
    expect(editor.getSnapshot()).toMatchObject({ dirty: true, draft: {
      title: "new unsaved title", revision: 2, path: "/plans/plan-a/v2.json", local: { steps: 200000 },
    } });
    const next = editor.beginSave()!;
    editor.completeSave(next, { ...next.draft, revision: 3 });
    expect(editor.getSnapshot().dirty).toBe(false);
  });
  it.each(["create", "save as"])("adopts the saved identity without losing concurrent edits on %s", (mode) => {
    const editor = new PlanDraftEditor();
    editor.select(mode === "create" ? { ...plan(), id: undefined, revision: undefined } : plan());
    const request = editor.beginSave()!;
    editor.patch({ description: "edited during request" });
    editor.completeSave(request, { ...plan(), id: "new-plan", revision: 1, path: "/new-plan/v1.json" });
    expect(editor.getSnapshot()).toMatchObject({ dirty: true, draft: {
      id: "new-plan", revision: 1, description: "edited during request", path: "/new-plan/v1.json",
    } });
  });
  it("does not replace a different selection with an old save response", () => {
    const editor = new PlanDraftEditor(); editor.select(plan());
    const request = editor.beginSave()!;
    editor.select({ ...plan("different"), id: "plan-b" });
    expect(editor.completeSave(request, { ...plan(), revision: 2 })).toBe("switched");
    expect(editor.getSnapshot().draft?.id).toBe("plan-b");
  });
  it("does not replace a reselected plan with an older request", () => {
    const editor = new PlanDraftEditor(); editor.select(plan());
    const request = editor.beginSave()!;
    editor.select(plan("reselected"));
    expect(editor.completeSave(request, { ...plan(), revision: 2 })).toBe("switched");
    expect(editor.getSnapshot().draft?.title).toBe("reselected");
  });
  it("checks current dirty state after an awaited draft load and respects cancel", async () => {
    const editor = new PlanDraftEditor(); editor.select(plan());
    const response = deferred<Plan>();
    const confirm = vi.fn(() => false);
    const loading = response.promise.then((loaded) => editor.switchDraft(loaded, confirm));
    editor.patch({ title: "edited while loading" });
    response.resolve(plan("loaded"));
    expect(await loading).toBe(false);
    expect(confirm).toHaveBeenCalledOnce();
    expect(editor.getSnapshot().draft?.title).toBe("edited while loading");
    expect(editor.getSnapshot().dirty).toBe(true);
  });
  it("allows explicitly confirmed discard and clears dirty state", () => {
    const editor = new PlanDraftEditor(); editor.select(plan()); editor.patch({ title: "dirty" });
    expect(editor.switchDraft(plan("replacement"), () => true)).toBe(true);
    expect(editor.getSnapshot()).toMatchObject({ dirty: false, draft: { title: "replacement" } });
  });
  it("accepts server-normalized fields when there are no newer edits", () => {
    const editor = new PlanDraftEditor(); editor.select(plan());
    const request = editor.beginSave()!;
    expect(editor.completeSave(request, { ...plan("normalized"), revision: 2 })).toBe("saved");
    expect(editor.getSnapshot()).toMatchObject({ dirty: false, draft: { title: "normalized", revision: 2 } });
  });
});
