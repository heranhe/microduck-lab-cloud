"use client";

import { useI18n } from "@/lib/i18n";
import { SERVOS, servoHelp, servoLabel, type Servo } from "@/lib/servo";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

/** The servo the simulator models while the Microduck trains. */
export function ServoChoice({ value, onChange }: { value: Servo; onChange: (servo: Servo) => void }) {
  const { tr, locale } = useI18n();
  return (
    <div style={{ margin: "0 0 6px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 4, flexWrap: "wrap" }}>
        <span style={{ color: "#8b93a3", fontSize: 10 }}>{tr("Servo", "舵机")}</span>
        {SERVOS.map((servo) => (
          <button key={servo} type="button" aria-pressed={value === servo} onClick={() => onChange(servo)}
            title={servoHelp(servo, locale)}
            style={{
              borderRadius: 12, padding: "1px 8px", font: `10px ${mono}`, cursor: "pointer",
              background: value === servo ? "#263c4d" : "#1c2230", color: value === servo ? "#e8f2fb" : "#9fb4d8",
              border: `1px solid ${value === servo ? "#70b8d8" : "rgba(255,255,255,0.08)"}`,
            }}>
            {servoLabel(servo, locale)}{servo === "feetech" ? tr(" (default)", "（默认）") : ""}
          </button>
        ))}
      </div>
      <div style={{ color: "#747e8d", fontSize: 10, marginTop: 3 }}>{servoHelp(value, locale)}</div>
    </div>
  );
}
