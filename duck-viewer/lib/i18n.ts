"use client";

import { useEffect, useSyncExternalStore } from "react";
import { loadJSON, saveJSON } from "./persist";

export type Locale = "en" | "zh";

let locale = loadJSON<Locale>("locale", "en") === "zh" ? "zh" : "en";
const listeners = new Set<() => void>();

export function setLocale(next: Locale) {
  if (next === locale) return;
  locale = next;
  saveJSON("locale", next);
  listeners.forEach((listener) => listener());
}

export function useI18n() {
  const current = useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => locale,
    () => "en" as Locale
  );

  useEffect(() => {
    document.documentElement.lang = current === "zh" ? "zh-CN" : "en";
  }, [current]);

  return {
    locale: current,
    isZh: current === "zh",
    tr: (en: string, zh: string) => (current === "zh" ? zh : en),
    toggleLocale: () => setLocale(current === "zh" ? "en" : "zh"),
  };
}
