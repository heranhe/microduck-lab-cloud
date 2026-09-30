export type CloudProvider = "colab" | "hf";
export type ComputeSource = "local" | CloudProvider;

export type CloudAccount = {
  installed?: boolean;
  connected: boolean;
  balance?: number | null;
  rate?: number | null;
  username?: string;
  email?: string | null;
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
 * offered only when the explicitly selected official task has a registered upstream implementation.
 */
export function cloudTaskFor(robotId: string, behaviorId?: string): string | null {
  if (robotId !== "microduck") return null;
  if (behaviorId === "official_velstand") return "Mjlab-VelStand-Flat-MicroDuck";
  if (behaviorId === "official_velocity") return "Mjlab-Velocity-Flat-MicroDuck";
  return null;
}
