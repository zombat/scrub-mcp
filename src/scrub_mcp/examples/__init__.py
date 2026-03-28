from pathlib import Path


def bundled_examples_dir() -> Path:
    """Return the path to the bundled training examples directory."""
    return Path(__file__).parent
