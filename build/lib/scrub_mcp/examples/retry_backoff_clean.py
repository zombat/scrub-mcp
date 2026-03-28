"""Retry and backoff utilities for resilient function execution."""

import random
import time


def retry(
    func: callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> object:
    """Call *func* repeatedly until it succeeds or attempts are exhausted.

    Uses exponential backoff with optional full jitter between attempts.

    Args:
        func: Zero-argument callable to invoke.
        max_attempts: Maximum number of calls before re-raising. Defaults to 3.
        base_delay: Initial sleep duration in seconds. Defaults to 1.0.
        backoff: Multiplier applied to the delay after each failure. Defaults to 2.0.
        jitter: If True, randomize the sleep duration within [0.5x, 1.5x] of
            the computed delay. Defaults to True.
        exceptions: Tuple of exception types that trigger a retry. Defaults to
            (Exception,).

    Returns:
        The return value of *func* on success.

    Raises:
        Exception: Re-raises the last caught exception when all attempts fail.
    """
    attempt = 0
    delay = base_delay
    last_exc: Exception | None = None

    while attempt < max_attempts:
        try:
            return func()
        except exceptions as e:
            last_exc = e
            attempt += 1
            if attempt >= max_attempts:
                break
            sleep_time = delay * (0.5 + random.random()) if jitter else delay
            time.sleep(sleep_time)
            delay *= backoff

    raise last_exc


def with_timeout(func: callable, timeout: float) -> object:
    """Execute *func* in a background thread, raising if it exceeds *timeout*.

    Args:
        func: Zero-argument callable to run.
        timeout: Maximum allowed execution time in seconds.

    Returns:
        The return value of *func*.

    Raises:
        TimeoutError: If *func* does not complete within *timeout* seconds.
        Exception: Any exception raised by *func* is re-raised.
    """
    import threading

    result: list[object] = [None]
    error: list[Exception | None] = [None]

    def run() -> None:
        try:
            result[0] = func()
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=run)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError(f"Function did not complete within {timeout}s")
    if error[0] is not None:
        raise error[0]
    return result[0]


def compute_delay(attempt: int, base: float, backoff: float, cap: float | None = None) -> float:
    """Compute the exponential backoff delay for a given attempt number.

    Args:
        attempt: Zero-indexed attempt number.
        base: Base delay in seconds.
        backoff: Exponential multiplier applied per attempt.
        cap: Maximum delay in seconds. No cap applied if None.

    Returns:
        Delay in seconds for the given attempt.
    """
    d = base * (backoff ** attempt)
    if cap is not None:
        d = min(d, cap)
    return d


def is_retryable(exc: Exception, retryable_types: list[type[Exception]]) -> bool:
    """Check whether an exception matches any of the retryable types.

    Args:
        exc: The exception to check.
        retryable_types: List of exception classes that should trigger a retry.

    Returns:
        True if *exc* is an instance of any type in *retryable_types*, False otherwise.
    """
    return any(isinstance(exc, t) for t in retryable_types)
