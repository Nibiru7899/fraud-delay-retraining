import numpy as np

from fraud.cost import best_threshold, net_value


def test_net_value_hand():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.8, 0.3, 0.1])
    amt = np.array([100.0, 50.0, 20.0, 10.0])
    # threshold 0.5 flags rows 0,1 -> caught 100, cost 2*10
    assert net_value(y, p, amt, 0.5, 10) == 80.0


def test_best_threshold_matches_brute_force():
    r = np.random.default_rng(1)
    y = r.integers(0, 2, 300)
    p = np.round(r.uniform(size=300), 2)  # ties on purpose
    amt = r.uniform(1, 200, 300)
    thr, val = best_threshold(y, p, amt, 30.0)
    brute = max(net_value(y, p, amt, t, 30.0) for t in np.unique(p))
    assert np.isclose(val, max(brute, 0.0))
    if np.isfinite(thr):
        assert np.isclose(net_value(y, p, amt, thr, 30.0), val)
