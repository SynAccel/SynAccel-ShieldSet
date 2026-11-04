# SynAccel-ShieldSet
Automated dataset vulnerability auditor and neighborhood consensus analyzer for detecting label-flipping and data poisoning attacks.

# SynAccel-ShieldSet

A research contribution to dataset integrity and adversarial ML defense.
SynAccel-ShieldSet audits training datasets for label-flipping and
neighborhood-based poisoning by measuring label consensus among each sample's
k nearest neighbors. A sample's suspicion score is the fraction of neighbors
disagreeing with its label; samples at or above the threshold are flagged.

## Install

    pip install -e .

## Python usage

    from synaccel_shieldset import ShieldSetAuditor
    auditor = ShieldSetAuditor(k=7, suspicion_threshold=0.7)
    result = auditor.audit(X, y)   # flagged_indices, suspicion_scores, metrics
    X_clean, y_clean = auditor.repair(X, y, strategy="relabel")  # or "remove"

## CLI

    synaccel-shieldset --data-x X.npy --data-y y.npy --k 7 --threshold 0.7 --output-dir out

Outputs `flagged_indices.npy`, `suspicion_scores.npy`, `X_clean.npy`, `y_clean.npy`.
