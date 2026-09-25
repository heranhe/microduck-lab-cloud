import { redirect } from "next/navigation";

/** Keep old bookmarks working. Cloud compute now lives inside the lab so the
 * runtime timer and emergency disconnect stay visible beside the robots. */
export default function LegacyCloudPage() {
  redirect("/?cloud=1");
}
