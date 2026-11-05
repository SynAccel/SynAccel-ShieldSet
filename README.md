# SynAccel ShieldSet

A research tool for auditing dataset label consistency with neighborhood consensus. It flags samples for review; a flagged sample is not, by itself, proof of data poisoning or an incorrect label.

SynAccel-ShieldSet measures label agreement among each sample's `k` nearest neighbors. A sample's suspicion score is the fraction of neighbors whose labels differ from its own; samples at or above the configured threshold are flagged. The method is a heuristic and depends on the selected features, distance metric, `k`, and threshold.

## Install

```bash
pip install -e .
```

## Python usage

```python
from synaccel_shieldset import ShieldSetAuditor

auditor = ShieldSetAuditor(k=7, suspicion_threshold=0.7)
result = auditor.audit(X, y)  # flagged_indices, suspicion_scores, metrics
X_clean, y_clean = auditor.repair(X, y, strategy="relabel")  # or "remove"
```

Review flagged samples and validate the repair strategy for the dataset before using repaired data in downstream training.

## CLI

```bash
synaccel-shieldset --data-x X.npy --data-y y.npy --k 7 --threshold 0.7 --output-dir out
```

Outputs `flagged_indices.npy`, `suspicion_scores.npy`, `X_clean.npy`, and `y_clean.npy`.