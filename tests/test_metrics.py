import numpy as np
from jevbench.metrics import accuracy, brier_score, ece, selective_coverage

def test_metrics_sane():
    y = np.array([0, 1, 1, 0])
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]])
    assert accuracy(y, p) == 1.0
    assert 0 <= brier_score(y, p) <= 2
    assert 0 <= ece(y, p) <= 1
    cov, thr = selective_coverage(y, p, 0.05)
    assert 0 <= cov <= 1

def test_wire_format():
    from jevbench.client import systemone_local_predict
    q = {"c": {"type": "choice", "instructions": "q", "criteria": {"a": "a", "b": "b"}},
         "n": {"type": "noul", "instructions": "s?"}}
    ans = systemone_local_predict("t", "state", q, lambda qq: np.array([0.7, 0.3]))
    assert ans["answers"]["c"]["choice"] == "a"
    assert ans["usage"]["input_tokens"] > 0
