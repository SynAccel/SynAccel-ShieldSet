"""Command-line interface for SynAccel-ShieldSet."""
import argparse
import os
import sys

import numpy as np

from .auditor import ShieldSetAuditor


def build_parser():
    p = argparse.ArgumentParser(
        prog="synaccel-shieldset",
        description="Audit a training dataset for label-flipping poisoning.")
    p.add_argument("--data-x", required=True, help="features (.npy)")
    p.add_argument("--data-y", required=True, help="labels (.npy)")
    p.add_argument("--k", type=int, default=7, help="neighbors (default 7)")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="suspicion threshold in [0,1] (default 0.7)")
    p.add_argument("--output-dir", default="shieldset_output",
                   help="directory for results")
    p.add_argument("--strategy", choices=["relabel", "remove"],
                   default="relabel", help="repair strategy")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        X = np.load(args.data_x, allow_pickle=False)
        y = np.load(args.data_y, allow_pickle=False)
        auditor = ShieldSetAuditor(k=args.k, suspicion_threshold=args.threshold)
        print("=== SynAccel-ShieldSet Audit ===")
        print(f"[1/3] Loaded X{X.shape} y{y.shape}")
        print(f"[2/3] Scanning with k={args.k}, threshold={args.threshold} ...")
        result = auditor.audit(X, y)
        print(f"      Flagged samples: {len(result.flagged_indices)}")
        print(f"[3/3] Repairing (strategy={args.strategy}) ...")
        Xc, yc = auditor.repair(X, y, strategy=args.strategy)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "flagged_indices.npy"),
            result.flagged_indices)
    np.save(os.path.join(args.output_dir, "suspicion_scores.npy"),
            result.suspicion_scores)
    np.save(os.path.join(args.output_dir, "X_clean.npy"), Xc)
    np.save(os.path.join(args.output_dir, "y_clean.npy"), yc)

    print("\n--- Summary ---")
    for key, val in result.metrics.items():
        print(f"{key:>18}: {val:.4f}" if isinstance(val, float)
              else f"{key:>18}: {val}")
    print(f"{'clean_samples':>18}: {len(yc)}")
    print(f"Results written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
