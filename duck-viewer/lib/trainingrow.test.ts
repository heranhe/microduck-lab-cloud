import { describe, expect, it } from "vitest";

import { trainingForDuck } from "@/lib/lab";

/** The frame shape these rules read, trimmed to what matters. */
const job = (trainee: string | null | undefined, status = "training") =>
  ({ runName: "teach-moss_pick-2df5ad-s1", status, trainee,
     behavior: { title: "Scoop a can (MOSS)" },
     progress: { steps: 795_000, total: 2_500_000 } }) as never;

describe("which roster row is the robot that is actually training", () => {
  it("follows the trainee the SERVER names, not the first slot", () => {
    // The real case from the lab: a leftover called "trainee" from a finished
    // run is on stage, and the live job is on trainee4.
    const frame = { training: job("trainee4"), trainings: [job("trainee4")] };
    expect(trainingForDuck(frame as never, "trainee4")).toBeTruthy();
    // ...and the leftover must NOT be decorated. Hardcoding `d.id ===
    // "trainee"` put the progress label and the ＋ helper button here, on a
    // robot doing nothing, twice.
    expect(trainingForDuck(frame as never, "trainee")).toBeNull();
  });

  it("falls back to 'trainee' only when the payload names nobody", () => {
    // A lab older than the `trainee` field. Guessing is right here and
    // nowhere else.
    const frame = { training: job(undefined), trainings: [job(undefined)] };
    expect(trainingForDuck(frame as never, "trainee")).toBeTruthy();
    expect(trainingForDuck(frame as never, "trainee4")).toBeNull();
  });

  it("does not decorate a job that has stopped", () => {
    const frame = { training: job("trainee4", "done"),
                    trainings: [job("trainee4", "done")] };
    expect(trainingForDuck(frame as never, "trainee4")).toBeNull();
  });

  it("gives each concurrent job its OWN row", () => {
    const frame = { training: job("trainee5"),
                    trainings: [job("trainee4"), job("trainee5")] };
    expect(trainingForDuck(frame as never, "trainee4")).toBeTruthy();
    expect(trainingForDuck(frame as never, "trainee5")).toBeTruthy();
    expect(trainingForDuck(frame as never, "trainee2")).toBeNull();
  });

  it("is safe before the first frame arrives", () => {
    expect(trainingForDuck(null, "trainee")).toBeNull();
    expect(trainingForDuck({} as never, "trainee")).toBeNull();
  });
});
