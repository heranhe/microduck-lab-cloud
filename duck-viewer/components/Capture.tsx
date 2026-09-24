"use client";

// In-Canvas helpers for 🎥 record and 📷 shot, shared by the lab page
// (Viewer.tsx) and the /sim page (SimViewer.tsx). They own nothing but the
// gl handle: the phase machine is lib/record.ts, the take is useTake.ts.

import { useEffect } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { pumpCaptureFrame, setCaptureCanvas, setSnapshotFn } from "@/lib/record";
import { pushToast } from "./Toasts";

/** Registers the WebGL canvas for MediaRecorder and pushes one captured
 *  video frame per RENDERED frame while a take is rolling (lib/record.ts
 *  explains why automatic capture is not enough). */
export function CaptureCanvas() {
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    setCaptureCanvas(gl.domElement);
    return () => setCaptureCanvas(null);
  }, [gl]);
  useFrame(() => pumpCaptureFrame());
  return null;
}

/** 📷 snapshot: registers the synchronous take-a-PNG implementation (see
 *  lib/record.ts for why it must be synchronous — the download has to stay
 *  inside the button's user gesture or Chrome drops it as "automatic").
 *  The WebGL buffer (preserveDrawingBuffer:false) is only readable in the
 *  same task as a render, so this re-renders, reads with toDataURL (sync),
 *  and restores. Objects tagged `userData.hideInCapture` (selection rings)
 *  are hidden for the capture render only. */
export function Snapshotter() {
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const scene3 = useThree((s) => s.scene);
  useEffect(() => {
    setSnapshotFn((name) => {
      const hidden: THREE.Object3D[] = [];
      scene3.traverse((o) => {
        if (o.visible && o.userData.hideInCapture) {
          o.visible = false;
          hidden.push(o);
        }
      });
      gl.render(scene3, camera);
      const dataUrl = gl.domElement.toDataURL("image/png");
      hidden.forEach((o) => (o.visible = true));
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = `${name}.png`;
      a.click();
      pushToast(`📷 ${name}.png → downloads`);
    });
    return () => setSnapshotFn(null);
  }, [camera, gl, scene3]);
  return null;
}
