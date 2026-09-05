import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev server is started on localhost and the smoke path (and every
  // curl in .claude/skills/sim-smoke) reaches it on 127.0.0.1, which Next
  // treats as a different host and blocks from /_next/* — so the page loads,
  // bails out to client rendering, and paints black with three 403s and no
  // explanation on screen. Both spellings of this machine are the same
  // machine; say so.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
};

export default nextConfig;
