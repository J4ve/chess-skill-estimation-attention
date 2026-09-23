"""Aggregate the per-(arm,checkpoint) score_test_split.py JSONs + the paired
bootstrap JSON into a single analysis/heldout-test-eval-results.json and print a
Markdown summary table.

Usage:
    python analysis/scripts/aggregate_heldout_results.py \
        --eval_dir analysis/heldout_test_eval \
        --bootstrap analysis/heldout_test_eval/bootstrap.json \
        --out_json analysis/heldout-test-eval-results.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# display order + human labels
ARM_ORDER = [
    ("baseline", "reproduced baseline (arm1), lr 1e-4, no attention"),
    ("baseline_lr3e4", "baseline + lr 3e-4 (arm6), no attention  [confound control]"),
    ("attn_untuned", "baseline + attention (arm2), lr 1e-4"),
    ("attn_tuned", "tuned attention (dir preflight_check_2m), lr 3e-4  [FROZEN ARCH]"),
    ("deepcnn", "deeper-CNN (arm5), lr 3e-4 + attention  [not promoted]"),
    ("lowdropout", "low-dropout diagnostic, dropout 0.3  [internal only]"),
]
VAL_MAE = {
    "baseline": 175.56,
    "baseline_lr3e4": None,
    "attn_untuned": 173.59,
    "attn_tuned": 172.38,
    "deepcnn": 172.37,
    "lowdropout": 174.48,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="analysis/heldout_test_eval")
    ap.add_argument("--bootstrap", default="analysis/heldout_test_eval/bootstrap.json")
    ap.add_argument("--out_json", default="analysis/heldout-test-eval-results.json")
    args = ap.parse_args()

    per = {}
    for path in sorted(glob.glob(os.path.join(args.eval_dir, "*.json"))):
        base = os.path.basename(path)
        if base in ("bootstrap.json",) or base.endswith("results.json"):
            continue
        if "__" not in base:
            continue
        arm, tag = base[:-5].split("__", 1)
        with open(path) as f:
            per.setdefault(arm, {})[tag] = json.load(f)

    bootstrap = None
    if os.path.exists(args.bootstrap):
        with open(args.bootstrap) as f:
            bootstrap = json.load(f)

    out = {
        "split_manifest_sha256": "dee7d11779df12bfc1242445079f9903b1c48b0787103db63f7b410bd9ed2b3b",
        "n_test_games": 255000,
        "ratings_std": 366,
        "metric": "MAE in rating points, last-ply prediction, mean over both sides x all test games",
        "arms": {},
        "paired_bootstrap": bootstrap,
    }
    rows = []
    for arm, desc in ARM_ORDER:
        d = per.get(arm, {})
        best = d.get("best", {})
        ep60 = d.get("ep60", {})
        entry = {
            "description": desc,
            "val_mae": VAL_MAE.get(arm),
            "best_val_checkpoint": {
                "test_mae": best.get("overall_mae_rating_points"),
                "checkpoint": best.get("checkpoint"),
                "best_epoch": best.get("checkpoint_best_epoch"),
                "mae_by_time_control": best.get("mae_by_time_control"),
            }
            if best
            else None,
            "epoch60_checkpoint": {
                "test_mae": ep60.get("overall_mae_rating_points"),
                "checkpoint": ep60.get("checkpoint"),
                "mae_by_time_control": ep60.get("mae_by_time_control"),
            }
            if ep60
            else None,
        }
        out["arms"][arm] = entry
        rows.append(
            (
                arm,
                desc,
                VAL_MAE.get(arm),
                best.get("overall_mae_rating_points"),
                ep60.get("overall_mae_rating_points"),
            )
        )

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    def fmt(x):
        return f"{x:.2f}" if isinstance(x, (int, float)) else "-"

    print(f"\n{'arm':<16} {'val MAE':>8} {'test MAE (best-val)':>20} {'test MAE (ep60)':>16}")
    print("-" * 64)
    for arm, desc, v, b, e in rows:
        print(f"{arm:<16} {fmt(v):>8} {fmt(b):>20} {fmt(e):>16}")
    if bootstrap:
        print("\npaired bootstrap (per-game abs-err delta, 95% CI):")
        for name, r in bootstrap["pairs"].items():
            print(
                f"  {name}: {r['point_delta_mae']:+.2f}  "
                f"[{r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}]  "
                f"crosses_zero={r['ci_crosses_zero']}"
            )
    print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
