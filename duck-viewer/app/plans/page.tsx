"use client";
import dynamic from "next/dynamic";
const PlansPanel = dynamic(() => import("@/components/PlansPanel"), {ssr:false});
export default function PlansPage() { return <PlansPanel />; }
