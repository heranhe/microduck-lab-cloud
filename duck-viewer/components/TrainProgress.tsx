"use client";

import styles from "./CloudPanel.module.css";
import { useI18n } from "@/lib/i18n";
import type { ProgressView } from "@/lib/trainProgress";

/** The same progress readout for a CPU run and a cloud GPU run. */
export function TrainProgress({ view }: { view: ProgressView }) {
  const { tr } = useI18n();
  return <>
    <div className={styles.progressLine}>
      <strong>{view.percent != null && view.where === "local" ? `${view.percent.toFixed(1)}%` : view.phase}</strong>
      {view.where === "local" && <span>{view.phase}</span>}
      {view.count && <span>{view.count}{view.percent != null && view.where === "cloud" ? ` · ${view.percent.toFixed(1)}%` : ""}</span>}
      {view.remaining && <span>{tr("Remaining ≈", "剩余约")}{view.remaining}</span>}
    </div>
    {view.percent != null && (
      <div className={styles.progressTrack} role="progressbar" aria-label={tr("Training progress", "训练进度")}
        aria-valuenow={view.percent} aria-valuemin={0} aria-valuemax={100}>
        <span style={{ width: `${view.percent}%` }}/>
      </div>
    )}
    {(view.score != null || view.extras.length > 0) && (
      <div className={styles.snapshotLine}>
        {view.score != null && <span>{tr("Score", "得分")} {view.score}</span>}
        {view.extras.map((text) => <span key={text}>{text}</span>)}
      </div>
    )}
  </>;
}
