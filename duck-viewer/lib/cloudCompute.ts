export type CloudProvider = "colab" | "hf";
export type ComputeSource = "local" | CloudProvider;

export type CloudAccount = {
  installed?: boolean;
  connected: boolean;
  balance?: number | null;
  rate?: number | null;
  username?: string;
  message?: string;
};

export type CloudHardware = {
  id: string;
  label: string;
  gpu: string;
  quantity: string;
  vram: string;
  cost: number;
  unit: string;
};

export type HfCloudAccount = CloudAccount & { hardware: CloudHardware[] };

/**
 * Local teaching recipes and the official microduck_rl tasks are different
 * training stacks. Keep this mapping deliberately small: a cloud launch is
 * offered only when the selected local action has the same meaning upstream.
 */
export function cloudTaskFor(robotId: string, behaviorId?: string): string | null {
  if (robotId !== "microduck") return null;
  if (behaviorId === "stand") return "Mjlab-VelStand-Flat-MicroDuck";
  return null;
}

