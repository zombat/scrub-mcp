import time
import random


def retry(func, max_attempts=3, base_delay=1.0, backoff=2.0, jitter=True, exceptions=(Exception,)):
    attempt = 0
    delay = base_delay
    last_exc = None

    while attempt < max_attempts:
        try:
            return func()
        except exceptions as e:
            last_exc = e
            attempt += 1
            if attempt >= max_attempts:
                break
            sleep_time = delay
            if jitter:
                sleep_time = delay * (0.5 + random.random())
            time.sleep(sleep_time)
            delay *= backoff

    raise last_exc


def with_timeout(func, timeout):
    import threading
    result = [None]
    error = [None]

    def run():
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


def compute_delay(attempt, base, backoff, cap=None):
    d = base * (backoff ** attempt)
    if cap is not None:
        d = min(d, cap)
    return d


def is_retryable(exc, retryable_types):
    for t in retryable_types:
        if isinstance(exc, t):
            return True
    return False
