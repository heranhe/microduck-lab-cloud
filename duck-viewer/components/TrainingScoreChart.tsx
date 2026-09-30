export function TrainingScoreChart({ points, label }: { points: { x: number; y: number }[]; label?: string }) {
  if (!points.length) return null;
  const w = 250, h = 36;
  const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const path = points
    .map((p, i) => {
      const px = ((p.x - x0) / Math.max(x1 - x0, 1)) * w;
      const py = h - ((p.y - y0) / Math.max(y1 - y0, 1e-6)) * (h - 4) - 2;
      return `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg viewBox={`-2 -2 ${w + 4} ${h + 4}`} width="100%" height={h} preserveAspectRatio="none" style={{ display: "block" }} role="img" aria-label={label}>
      {points.length === 1 && <circle cx={0} cy={h - 2} r={2} fill="#7db8d8" />}
      <path d={path} fill="none" stroke="#7db8d8" strokeWidth={1.5} />
    </svg>
  );
}
