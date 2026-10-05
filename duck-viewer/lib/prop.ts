/** What shape a trainee's prop payload describes, and how big it is.
 *
 * `DuckFrame.ball` and `DuckFrame.ballGhost` carry one prop in one field,
 * and the server tells the shapes apart by LENGTH alone
 * (`robots/moss_env.MossPickEnv.marker_payload`):
 *
 * | len | payload                                   | shape          |
 * |-----|-------------------------------------------|----------------|
 * |   4 | `[x, y, z, r]`                            | sphere         |
 * |   5 | `[x, y, z, r, halfH]`                     | cylinder, upright |
 * |   9 | `[x, y, z, r, halfH, qw, qx, qy, qz]`     | cylinder, posed   |
 * |  10 | `[x, y, z, rx, ry, rz, qw, qx, qy, qz]`   | box (HALF extents)|
 *
 * Pure and in one place because the meshes that draw these are three
 * separate `visible` flags inside a render loop. Written as three ad-hoc
 * length tests, one of them asked only "is it a cylinder?" — so every BOX
 * prop was drawn with the orange ball still switched on underneath it, at
 * the box's x half-extent read as a sphere radius.
 */

export type PropShape = "none" | "sphere" | "cylinder" | "box";

export function propShape(
  payload: number[] | null | undefined,
): PropShape {
  if (!payload || payload.length < 4) return "none";
  if (payload.length >= 10) return "box";
  if (payload.length > 4) return "cylinder";
  return "sphere";
}

/** The footprint radius to draw this prop — and its ghost — at.
 *
 * A box has no radius: `payload[3]` is its x half-extent and nothing more,
 * so the wider of the two ground half-extents is used, which is the number
 * `GraspProp.radius` gives MuJoCo for the same prop. `fallback` covers a
 * payload with no prop in it and a zero slot.
 */
export function propRadius(
  payload: number[] | null | undefined,
  fallback = 0.035,
): number {
  const p = payload;
  switch (propShape(p)) {
    case "box":
      return Math.max(p![3], p![4]) || fallback;
    case "sphere":
    case "cylinder":
      return p![3] || fallback;
    default:
      return fallback;
  }
}
