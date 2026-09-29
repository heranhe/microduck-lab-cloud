import { useCallback, useEffect, useRef, useState } from "react";

export type PanelPoint = { x: number; y: number };

export function usePanelDrag(key: string) {
  const [point, setPoint] = useState<PanelPoint | null>(() => {
    try {
      const raw = localStorage.getItem(`microduck.panel.${key}`);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      return Number.isFinite(parsed?.x) && Number.isFinite(parsed?.y) ? parsed : null;
    } catch { return null; }
  });
  const [size, setSize] = useState<{ width: number; height: number } | null>(() => {
    try {
      const value = JSON.parse(localStorage.getItem(`microduck.panel.size.${key}`) || "null");
      return value && Number.isFinite(value.width) && Number.isFinite(value.height) && value.width > 0 && value.height > 0 ? value : null;
    } catch { return null; }
  });
  useEffect(() => {
    try { if (size) localStorage.setItem(`microduck.panel.size.${key}`, JSON.stringify(size)); } catch {}
  }, [key, size]);
  const resize = useRef<{ id: number; left: boolean; right: boolean; top: boolean; bottom: boolean; rect: DOMRect } | null>(null);
  const resizeEdge = (e: React.PointerEvent<HTMLElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const left = e.clientX < r.left + 16, right = e.clientX > r.right - 16;
    const top = e.clientY < r.top + 16, bottom = e.clientY > r.bottom - 16;
    // Keep generous corner targets, but narrow edge targets leave title
    // buttons and the panel's scrollbars available.
    if ((left || right) && (top || bottom)) return { left, right, top, bottom, rect: r };
    const edges = {
      left: e.clientX < r.left + 6, right: e.clientX > r.right - 6,
      top: e.clientY < r.top + 6, bottom: e.clientY > r.bottom - 6,
    };
    return Object.values(edges).some(Boolean) ? { ...edges, rect: r } : null;
  };
  const panelProps = {
    onPointerDownCapture: (e: React.PointerEvent<HTMLElement>) => {
      const hit = resizeEdge(e);
      if (e.button !== 0 || !hit) return;
      resize.current = { ...hit, id: e.pointerId };
      e.currentTarget.setPointerCapture(e.pointerId);
      e.preventDefault();
      e.stopPropagation();
    },
    onPointerMoveCapture: (e: React.PointerEvent<HTMLElement>) => {
      const active = resize.current;
      const hit = active || resizeEdge(e);
      e.currentTarget.style.cursor = !hit ? "" :
        (hit.left || hit.right) && (hit.top || hit.bottom)
          ? (hit.left === hit.top ? "nwse-resize" : "nesw-resize")
          : (hit.left || hit.right) ? "ew-resize" : "ns-resize";
      if (!active) return;
      const r = active.rect;
      const minW = Math.min(key === "policy" ? 220 : 300, window.innerWidth - 16);
      const minH = Math.min(180, window.innerHeight - 16);
      const x = active.left ? Math.max(8, Math.min(r.right - minW, e.clientX)) : r.left;
      const y = active.top ? Math.max(8, Math.min(r.bottom - minH, e.clientY)) : r.top;
      const right = active.right ? Math.min(window.innerWidth - 8, Math.max(r.left + minW, e.clientX)) : r.right;
      const bottom = active.bottom ? Math.min(window.innerHeight - 8, Math.max(r.top + minH, e.clientY)) : r.bottom;
      setPoint({ x, y });
      setSize({ width: right - x, height: bottom - y });
      e.stopPropagation();
    },
    onPointerUpCapture: (e: React.PointerEvent<HTMLElement>) => {
      if (!resize.current) return;
      resize.current = null;
      e.currentTarget.releasePointerCapture(e.pointerId);
      e.stopPropagation();
    },
    onPointerCancelCapture: () => { resize.current = null; },
  };
  const resizeStyle: React.CSSProperties = size ? {
    width: size.width, height: size.height, maxWidth: "calc(100vw - 16px)",
    maxHeight: "calc(100vh - 16px)", boxSizing: "border-box",
  } : {};
  const drag = useRef<{ id: number; dx: number; dy: number } | null>(null);
  useEffect(() => {
    try { if (point) localStorage.setItem(`microduck.panel.${key}`, JSON.stringify(point)); } catch {}
  }, [key, point]);
  const onPointerDown = useCallback((e: React.PointerEvent<HTMLElement>) => {
    if (e.button !== 0) return;
    const target = e.target as HTMLElement;
    if (target.closest("button, input, select, textarea, a")) return;
    const panel = e.currentTarget.parentElement;
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    drag.current = { id: e.pointerId, dx: e.clientX - rect.left, dy: e.clientY - rect.top };
    e.currentTarget.setPointerCapture(e.pointerId);
    e.preventDefault();
  }, []);
  const onPointerMove = useCallback((e: React.PointerEvent<HTMLElement>) => {
    const active = drag.current;
    if (!active || active.id !== e.pointerId) return;
    const panel = e.currentTarget.parentElement;
    if (!panel) return;
    const rect = panel.getBoundingClientRect();
    const x = Math.max(8, Math.min(window.innerWidth - rect.width - 8, e.clientX - active.dx));
    const y = Math.max(8, Math.min(window.innerHeight - rect.height - 8, e.clientY - active.dy));
    setPoint({ x, y });
  }, []);
  const onPointerUp = useCallback((e: React.PointerEvent<HTMLElement>) => {
    if (drag.current?.id === e.pointerId) drag.current = null;
  }, []);
  const reset = useCallback(() => setPoint(null), []);
  return { point, size, panelProps, resizeStyle, dragHandle: { onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp, style: { touchAction: "none" } }, reset };
}
