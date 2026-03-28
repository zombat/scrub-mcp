import pytest
from retry_backoff_messy import compute_delay, is_retryable, retry


def test_retry_succeeds_on_first_attempt():
    calls = []

    def f():
        calls.append(1)
        return 42

    assert retry(f) == 42
    assert len(calls) == 1


def test_retry_retries_on_failure_then_succeeds():
    calls = []

    def f():
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "ok"

    assert retry(f, max_attempts=3, base_delay=0) == "ok"
    assert len(calls) == 3


def test_retry_raises_after_max_attempts():
    def f():
        raise RuntimeError("always fails")

    with pytest.raises(RuntimeError, match="always fails"):
        retry(f, max_attempts=3, base_delay=0)


def test_retry_only_catches_specified_exceptions():
    def f():
        raise TypeError("wrong type")

    with pytest.raises(TypeError):
        retry(f, max_attempts=3, base_delay=0, exceptions=(ValueError,))


@pytest.mark.parametrize("attempt,base,backoff,expected", [
    (0, 1.0, 2.0, 1.0),
    (1, 1.0, 2.0, 2.0),
    (2, 1.0, 2.0, 4.0),
    (3, 0.5, 3.0, 13.5),
])
def test_compute_delay(attempt, base, backoff, expected):
    assert compute_delay(attempt, base, backoff) == pytest.approx(expected)


def test_compute_delay_respects_cap():
    assert compute_delay(10, 1.0, 2.0, cap=30.0) == 30.0


def test_is_retryable_match():
    assert is_retryable(ValueError("x"), [ValueError, IOError]) is True


def test_is_retryable_no_match():
    assert is_retryable(KeyError("x"), [ValueError, IOError]) is False


def test_is_retryable_empty_list():
    assert is_retryable(Exception(), []) is False
