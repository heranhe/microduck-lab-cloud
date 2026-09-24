// Run: node --experimental-transform-types --test tests/training-view.test.mjs
import assert from "node:assert/strict";
import test from "node:test";
import { trainingViewDucks } from "../lib/lab.ts";

test("training view hides other policies without mutating the server roster", () => {
  const trainee = { id: "trainee", name: "single leg" };
  const helper = { id: "helper1" };
  const roster = [{ id: "d0", name: "alpha_walking" }, trainee, helper];
  assert.deepEqual(trainingViewDucks(roster), [trainee, helper]);
  assert.equal(roster.length, 3);
  assert.equal(trainingViewDucks(roster)[0], trainee);
  const ordinary = [{ id: "d0" }];
  assert.equal(trainingViewDucks(ordinary), ordinary);
  assert.deepEqual(trainingViewDucks([]), []);
});
