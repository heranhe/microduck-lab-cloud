// Which servo the simulator models while the Microduck trains. The two are
// different motors with different physics, so a policy trained on one is a
// different artifact from one trained on the other.

export type Servo = "feetech" | "official";
type Locale = "en" | "zh";

/** What a training request gets when the person has not chosen. */
export const DEFAULT_SERVO: Servo = "feetech";
export const SERVOS: Servo[] = ["feetech", "official"];

export function isServo(value: unknown): value is Servo {
  return value === "feetech" || value === "official";
}

export function servoLabel(servo: Servo, locale: Locale): string {
  if (servo === "feetech") return locale === "zh" ? "飞特 STS3215" : "Feetech STS3215";
  return locale === "zh" ? "官方 XL330" : "Official XL330";
}

export function servoHelp(servo: Servo, locale: Locale): string {
  if (servo === "feetech") {
    return locale === "zh"
      ? "飞特 STS3215（7.4V）：按开源 BAM 辨识的参数模拟，力矩大、没有固件电流限制。参数来自上游，尚未在你的机器人上实测。"
      : "Feetech STS3215 (7.4 V), simulated with the open-source BAM identification: strong, no firmware current limit. Not measured on your robot.";
  }
  return locale === "zh"
    ? "官方 XL330：官方 microduck_rl 训练所用的舵机模型，有 1.75A 固件电流限制。"
    : "Official XL330: the servo model the upstream microduck_rl trains with, with a 1.75 A firmware current limit.";
}

/** Cloud tasks run upstream's own stack, which only models the XL330. */
export function cloudServoNote(locale: Locale): string {
  return locale === "zh"
    ? "云端官方任务固定使用官方 XL330 舵机，不能改用飞特。"
    : "Cloud official tasks always use the official XL330 servo; Feetech is not available there.";
}
