import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // 允许本地与局域网 IP 访问 Next.js 开发静态资源
  allowedDevOrigins: ["127.0.0.1", "localhost", "192.168.1.22", "*.local"],
};

export default nextConfig;
