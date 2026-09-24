"use client";

import { useCallback, useEffect, useSyncExternalStore } from "react";

export type Locale = "en" | "zh";
const KEY = "microduck-locale";
let locale: Locale = "en";
const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function setLocale(next: Locale) {
  locale = next;
  if (typeof window !== "undefined") window.localStorage.setItem(KEY, next);
  listeners.forEach((listener) => listener());
}

export function useI18n() {
  const current = useSyncExternalStore(subscribe, () => locale, () => "en" as Locale);
  useEffect(() => {
    const saved = window.localStorage.getItem(KEY);
    if (saved === "zh" && locale !== "zh") setLocale("zh");
  }, []);
  useEffect(() => { document.documentElement.lang = current === "zh" ? "zh-CN" : "en"; }, [current]);
  const tr = useCallback((en: string, zh: string) => current === "zh" ? zh : en, [current]);
  const toggle = useCallback(() => setLocale(current === "en" ? "zh" : "en"), [current]);
  return { locale: current, tr, toggle };
}
