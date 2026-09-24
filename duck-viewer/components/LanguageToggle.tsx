"use client";

import { useI18n } from "@/lib/i18n";
import { usePolicyOpen } from "@/lib/ui";

const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";

export function LanguageToggle() {
  const { isZh, tr, toggleLocale } = useI18n();
  const policyOpen = usePolicyOpen();

  return (
    <button
      data-policy-ui
      type="button"
      onClick={toggleLocale}
      title={tr("switch to Chinese", "切换到英文")}
      aria-label={tr("switch to Chinese", "切换到英文")}
      style={{
        position: "absolute",
        zIndex: 21,
        top: 20,
        right: policyOpen ? 258 : 126,
        minWidth: 34,
        height: 28,
        background: "rgba(14,16,20,0.86)",
        border: "1px solid rgba(255,255,255,0.14)",
        borderRadius: 8,
        color: "#cfe4f5",
        cursor: "pointer",
        fontFamily: mono,
        fontSize: 11,
        padding: "0 8px",
        backdropFilter: "blur(6px)",
        transition: "right 140ms ease-out, border-color 90ms ease-out",
      }}
    >
      {isZh ? "EN" : "中"}
    </button>
  );
}
