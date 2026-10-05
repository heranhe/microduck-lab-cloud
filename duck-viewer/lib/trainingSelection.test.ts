import { describe, expect, it } from "vitest";
import { displayedTraining, trainingForDuck, type Frame, type TrainingPayload } from "./lab";

function job(runName: string, trainee: string, status: TrainingPayload["status"] = "training") {
  return { runName, trainee, status, restarting: false } as TrainingPayload;
}
function frame(jobs: TrainingPayload[]): Frame {
  return { cmd: [0, 0, 0], mode: "auto", ducks: [], trainings: jobs, training: jobs.at(-1) };
}

describe("training selection with concurrent save/resume", () => {
  it("keeps a saving trainee protected and locates it by its actual id", () => {
    const a = job("a", "trainee", "stopping");
    const b = job("b", "trainee2");
    expect(trainingForDuck(frame([a, b]), "trainee")).toBe(a);
    expect(trainingForDuck(frame([a, b]), "trainee2")).toBe(b);
    expect(trainingForDuck(frame([a, b]), "d0")).toBeNull();
  });
  it("both monitors follow the selected run and otherwise the newest live run", () => {
    const a = job("a", "trainee");
    const b = job("b", "trainee2", "done");
    expect(displayedTraining(frame([a, b]), "trainee2")).toBe(b);
    expect(displayedTraining(frame([a, b]), "d0")).toBe(a);
    expect(displayedTraining(frame([a, b]))).toBe(a);
  });
  it("retains single-job compatibility for an older server", () => {
    const a = job("a", "trainee", "stopping");
    const f = { ...frame([a]), trainings: undefined };
    expect(displayedTraining(f)).toBe(a);
    expect(trainingForDuck(f, "trainee")).toBe(a);
    expect(displayedTraining(null)).toBeNull();
  });
});
