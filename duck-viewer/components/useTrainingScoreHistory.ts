"use client";

import { useEffect, useState, type MutableRefObject } from "react";
import type { LabClient } from "@/lib/lab";

type Point = { x: number; y: number };

/** Track the full job across curriculum stages, including warm restarts. */
export function useTrainingScoreHistory(clientRef: MutableRefObject<LabClient | null>) {
  const [points, setPoints] = useState<Point[]>([]);
  useEffect(() => {
    let job = "";
    let lastSteps = -1;
    let history: Point[] = [];
    const sample = () => {
      const training = clientRef.current?.frame?.training;
      if (!training) {
        if (history.length) setPoints([]);
        history = [];
        job = "";
        lastSteps = -1;
        return;
      }
      const currentJob = (training.runName ?? "").replace(/-s\d+$/, "");
      if (currentJob !== job) {
        job = currentJob;
        history = [];
        lastSteps = -1;
        setPoints([]);
      }
      const steps = training.progress.overallSteps ?? training.progress.steps;
      const score = training.progress.ep_rew;
      if (steps == null || score == null || !Number.isFinite(steps) || !Number.isFinite(score)) return;
      if (steps < lastSteps) history = history.filter((point) => point.x < steps);
      if (steps === lastSteps && history.at(-1)?.y === score) return;
      lastSteps = steps;
      // Replace a revised score at the same step rather than draw a vertical edge.
      history = [...history.filter((point) => point.x !== steps), { x: steps, y: score }];
      setPoints(history);
    };
    const timer = window.setInterval(sample, 500);
    return () => window.clearInterval(timer);
  }, [clientRef]);
  return points;
}
