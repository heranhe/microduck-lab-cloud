import { describe, expect, it } from "vitest";
import { displayedTraining, liveTraining, trainingForDuck, type Frame, type TrainingPayload } from "./lab";

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

describe("liveTraining: what the status bar follows", () => {
  const live = job("live", "trainee2");
  const old = job("old", "trainee", "done");
  it("is not hidden by selecting a finished run, which the panel card does follow", () => {
    expect(displayedTraining(frame([old, live]), "trainee")).toBe(old);
    expect(liveTraining(frame([old, live]), "trainee")).toBe(live);
  });
  it("follows the selected run while it is the one training, else the newest running one", () => {
    const a = job("a", "t1");
    const b = job("b", "t2");
    expect(liveTraining(frame([a, b]), "t1")).toBe(a);
    expect(liveTraining(frame([a, b]), "t2")).toBe(b);
    expect(liveTraining(frame([a, b]), "d0")).toBe(b);
    expect(liveTraining(frame([a, b]))).toBe(b);
  });
  it("counts a saving or restarting run as running, and nothing finished", () => {
    expect(liveTraining(frame([job("s", "t", "stopping")]))?.runName).toBe("s");
    const restarting = { ...job("r", "t", "stopped"), restarting: true } as TrainingPayload;
    expect(liveTraining(frame([restarting]))).toBe(restarting);
    expect(liveTraining(frame([old, job("x", "u", "failed")]), "trainee")).toBeNull();
    expect(liveTraining(null)).toBeNull();
  });
  it("keeps working against an older server that sends only one job", () => {
    const f = { ...frame([job("a", "trainee", "stopping")]), trainings: undefined };
    expect(liveTraining(f)?.runName).toBe("a");
  });
});
