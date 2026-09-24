#!/usr/bin/env bash
# One-command workspace setup — a Mac (Apple Silicon or Intel) or Linux.
#
#   git clone <this repo> microduck-workspace && cd microduck-workspace && ./scripts/setup.sh
#
# Clones the two upstream Pollen repos INSIDE this checkout (beside
# microduck_local/ and duck-viewer/ — where CI, the docs and contract.py look
# for them) at the shas CI pins (the contract, golden-bit and symmetry tests
# are measured against those exact models and policies), fetches the shipped
# policy set from the Hub at a pinned revision, syncs the Python env with uv,
# installs the viewer's npm packages, and runs the quick contract tests.
# Re-running it is safe: it only moves the upstream checkouts to the pinned
# shas and downloads whichever policy files are missing.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
ws="$here"   # the workspace IS this checkout; the upstream clones are .gitignored inside it
RL_SHA="$(sed -n 's/.*git -C microduck_rl checkout \([0-9a-f]\{40\}\).*/\1/p' "$here/AGENTS.md" | head -1)"
MD_SHA="$(sed -n 's/.*git -C microduck checkout \([0-9a-f]\{40\}\).*/\1/p' "$here/AGENTS.md" | head -1)"
[ -n "$RL_SHA" ] && [ -n "$MD_SHA" ] || { echo "could not read the pinned upstream shas from AGENTS.md"; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1 — $2"; exit 1; }; }
need git "https://git-scm.com"
need uv "https://docs.astral.sh/uv/  (brew install uv)"
need npm "https://nodejs.org  (brew install node)"

clone_at() {  # repo dir sha
  if [ ! -d "$ws/$2/.git" ]; then
    echo "→ cloning pollen-robotics/$1 into $ws/$2"
    git clone -q "https://github.com/pollen-robotics/$1" "$ws/$2"
  fi
  if [ "$(git -C "$ws/$2" rev-parse HEAD)" != "$3" ]; then
    git -C "$ws/$2" fetch -q origin
    git -C "$ws/$2" checkout -q "$3"
  fi
  echo "✓ $2 at $(git -C "$ws/$2" rev-parse --short HEAD) (pinned)"
}
clone_at microduck_rl microduck_rl "$RL_SHA"
clone_at microduck microduck "$MD_SHA"

echo "→ uv sync (microduck_local)"
(cd "$here/microduck_local" && uv sync -q)

# The shipped policies. Upstream stopped vendoring them on 2026-09-03
# (pollen-robotics/microduck ef4becf: "policies/ leaves the repository") — a
# `microduck` checkout past the pinned sha has no policies/ at all, and the
# by-hand clone in README.md lands on main. Their home is now the Hub repo
# pollen-robotics/microduck-policies. Fetch the set at a PINNED revision: the
# nine files at this commit are byte-identical to the ones the pinned
# microduck sha vendored (git blob hashes compared 2026-09-06), which is what
# the golden-bit and symmetry tests were measured against. Bump the revision
# the way the shas are bumped — on purpose, with the tests re-measured.
POLICY_REPO="pollen-robotics/microduck-policies"
POLICY_REV="088524a64e2557dc453256b6071dbb9d23888802"   # the files of tag v4
POLICIES=(alpha_walking alpha_stand alpha_sitstand alpha_ground_pick
          ball_kick_left ball_kick_right roller roller_crouch roulade)
policy_dir="$ws/microduck/policies"
missing=()
for p in "${POLICIES[@]}"; do [ -s "$policy_dir/$p.onnx" ] || missing+=("$p.onnx"); done
if [ "${#missing[@]}" -gt 0 ]; then
  echo "→ downloading ${#missing[@]} shipped policies from huggingface.co/$POLICY_REPO @ ${POLICY_REV:0:7}"
  mkdir -p "$policy_dir"
  (cd "$here/microduck_local" && uv run hf download "$POLICY_REPO" "${missing[@]}" \
     --revision "$POLICY_REV" --local-dir "$policy_dir" --quiet >/dev/null)
  for p in "${missing[@]}"; do
    [ -s "$policy_dir/$p" ] || { echo "download failed: $policy_dir/$p"; exit 1; }
  done
fi
echo "✓ shipped policies (${#POLICIES[@]}) in $policy_dir"

echo "→ npm install (duck-viewer)"
(cd "$here/duck-viewer" && npm install --silent --no-audit --no-fund)

for b in follow-v1 follow-v2; do
  [ -f "$here/microduck_local/brains/$b/brain.onnx" ] && echo "✓ shipped brain $b" || echo "! brains/$b/brain.onnx missing (git lfs? a partial clone?)"
done

echo "→ contract smoke tests"
(cd "$here/microduck_local" && uv run --with pytest pytest tests/test_env_contract.py tests/test_world.py tests/test_brain.py -q -p no:cacheprovider)

cat <<MSG

Ready. From microduck_local/:
  uv run --with pytest pytest tests/          # the whole suite (~4 min); the golden-bit
                                             # files SKIP until you record this Mac's:
  MICRODUCK_RECORD_GOLDENS=1 uv run --with pytest pytest tests/test_step_perf_parity.py tests/test_bam_perf_parity.py
  uv run duck-lab --world pitch               # then, from duck-viewer/: npm run dev → /sim
  uv run eval-pitch --seeds 4 --seconds 300 --jobs 4
  uv run eval-tidy --seeds 16 --seconds 300 --jobs 4
MSG
