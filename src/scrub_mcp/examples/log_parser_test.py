import pytest
from log_parser_messy import find_anomalies


def test_find_anomalies_empty():
    assert find_anomalies([]) == []


def test_find_anomalies_no_outliers():
    entries = [{"pattern": "a"}, {"pattern": "b"}, {"pattern": "c"}]
    assert find_anomalies(entries) == []


@pytest.mark.parametrize("threshold", [1, 2, 3])
def test_find_anomalies_with_outlier(threshold):
    entries = [{"pattern": "a"}] * 100 + [{"pattern": "b"}]
    result = find_anomalies(entries, threshold=threshold)
    assert len(result) >= 1
    assert result[0]["pattern"] == "a"
    assert result[0]["z_score"] > 0


def test_find_anomalies_returns_sorted_by_z_score():
    entries = [{"pattern": "a"}] * 50 + [{"pattern": "b"}] * 20 + [{"pattern": "c"}]
    result = find_anomalies(entries, threshold=1)
    z_scores = [r["z_score"] for r in result]
    assert z_scores == sorted(z_scores, reverse=True)


def test_find_anomalies_zero_std_no_crash():
    entries = [{"pattern": "a"}, {"pattern": "a"}, {"pattern": "a"}]
    result = find_anomalies(entries)
    assert isinstance(result, list)
