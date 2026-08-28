import numpy as np
from histrestore.evidence import extract_pair_evidence
from histrestore.region import alpha_map

def test_identical_pair_with_edges_has_zero_vccrp():
    x = np.full((64, 64, 3), 220, dtype=np.uint8)
    x[:, :32] = 40
    e = extract_pair_evidence(x, x)
    assert abs(e.content_risk) < 1e-8

def test_alpha_range():
    x = np.full((64, 64, 3), 180, dtype=np.uint8)
    a = alpha_map(x)
    assert a.min() >= 0.68 - 1e-5
    assert a.max() <= 0.98 + 1e-5
