import type { Locale } from "./i18n";
import type { BehaviorCard, TermCard } from "./lab";

// UI copy is keyed by the recipe's stable identifiers. Training still sends
// the original term keys and weights to duck-lab, regardless of language.
const oneLegTerms: Record<string, string> = {
  one_leg_hold: "仅左脚着地并保持直立，可获得大量分数",
  foot_in_air: "右脚保持离地约 5 厘米，可获得分数",
  flat_stance_foot: "左脚平贴地面，可获得分数",
  head_up: "头部保持自然抬起姿态，可获得分数",
  head_up_pull: "将头部抬向自然姿态，可获得分数",
  planted_foot: "左脚持续着地，可获得分数",
  stay_upright: "身体保持直立，可获得分数",
  calm_body: "身体晃动或乱甩会扣分",
  stay_home: "偏离起始位置会扣分",
  face_home: "偏离起始朝向会扣分",
  stay_put: "从原来的位置漂移会扣分",
  smooth_moves: "动作突然或抽搐会扣分",
  gentle_joints: "关节快速乱甩会扣分",
  save_energy: "电机过度用力会扣分",
};

const oneLegMetricLabels: Record<string, string> = {
  one_leg_hold: "单脚站稳",
  foot_in_air: "右脚抬起",
  flat_stance_foot: "左脚贴地",
  head_up: "头部抬起",
  head_up_pull: "抬头动作",
  planted_foot: "左脚着地",
  stay_upright: "身体直立",
  calm_body: "身体平稳",
  stay_home: "保持原位",
  face_home: "保持朝向",
  stay_put: "避免漂移",
  smooth_moves: "动作平顺",
  gentle_joints: "关节轻柔",
  save_energy: "节省能量",
};

export function rewardMetricLabel(key: string, behaviorId: string, locale: Locale): string {
  const normalized = key.replace(/_penalty$/, "");
  if (locale === "zh" && behaviorId === "one_leg") {
    return oneLegMetricLabels[normalized] ?? normalized.replace(/_/g, " ");
  }
  return normalized.replace(/_/g, " ");
}

export function localizeTerm(term: TermCard, behaviorId: string, locale: Locale): TermCard {
  const friendly = locale === "zh" && behaviorId === "one_leg"
    ? oneLegTerms[term.key]
    : undefined;
  return friendly ? { ...term, friendly } : term;
}

export function localizeBehavior(card: BehaviorCard, locale: Locale): BehaviorCard {
  if (locale !== "zh" || card.id !== "one_leg") return card;
  return {
    ...card,
    title: "单脚站立",
    description: "右脚抬起、重心放在左腿，平稳地保持单脚站立。",
    howItLearns: "小鸭子起初会随机尝试动作，也常常摔倒。仿真每 20 毫秒按下方规则评分：单脚站稳得分，摔倒不得分。PPO 会通过大量尝试逐渐学会得分更高的平衡动作。第二版配方还奖励身体平稳、支撑脚贴地，并对乱甩身体扣分，避免只为站稳而动作失控。",
    terms: card.terms.map((term) => localizeTerm(term, card.id, locale)),
    availableTerms: card.availableTerms?.map((term) => localizeTerm(term, card.id, locale)),
  };
}
