import { describe, expect, it } from "vitest";
import { DEFAULT_SERVO, SERVOS, cloudServoNote, isServo, servoHelp, servoLabel } from "./servo";

describe("servo choice", () => {
  it("defaults to the Feetech servo and offers exactly the two the lab models", () => {
    expect(DEFAULT_SERVO).toBe("feetech");
    expect(SERVOS).toEqual(["feetech", "official"]);
  });
  it("accepts only a servo the lab knows, including what comes back from storage", () => {
    expect(isServo("feetech") && isServo("official")).toBe(true);
    for (const bad of ["dynamixel", "", null, undefined, 3, "FEETECH"]) expect(isServo(bad)).toBe(false);
  });
  it("names each servo in both languages and says what differs", () => {
    expect(servoLabel("feetech", "zh")).toContain("STS3215");
    expect(servoLabel("official", "en")).toContain("XL330");
    expect(servoHelp("feetech", "zh")).toContain("电流限制");
    expect(servoHelp("official", "zh")).toContain("1.75A");
  });
  it("says plainly that the cloud cannot use Feetech", () => {
    expect(cloudServoNote("zh")).toContain("不能改用飞特");
  });
});
