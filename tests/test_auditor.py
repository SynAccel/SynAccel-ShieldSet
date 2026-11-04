import numpy as np
from synaccel_shieldset import ShieldSetAuditor


def data():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(50, 2), rng.randn(50, 2) + 10])
    y = np.array([0] * 50 + [1] * 50)
    y[3] = 1
    return X, y


def test_audit_flags_flipped():
    X, y = data()
    r = ShieldSetAuditor().audit(X, y)
    assert 3 in r.flagged_indices


def test_repair():
    X, y = data()
    a = ShieldSetAuditor()
    _, yc = a.repair(X, y)
    assert yc[3] == 0
    Xc, yc = a.repair(X, y, strategy="remove")
    assert len(Xc) == 99
