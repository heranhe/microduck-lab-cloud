import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dev server is started on localhost and the smoke path (and every
  // curl in .claude/skills/sim-smoke) reaches it on 127.0.0.1.
  // 同时允许本地与局域网 IP 访问 Next.js 开发静态资源
  allowedDevOrigins: ["127.0.0.1", "localhost", "192.168.1.22", "*.local"],
};

export default nextConfig;
