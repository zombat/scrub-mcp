#!/usr/bin/env python3
"""CLI wrapper around scrub_mcp.optimizers.examples_gen.

After pip install, prefer:
  python -m scrub_mcp.optimizers.tune --build-examples ./examples
"""

import argparse
import sys
from pathlib import Path

from scrub_mcp.optimizers.examples_gen import TOPICS, generate_examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "src" / "scrub_mcp" / "examples")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--topics", type=str, default=None,
                        help=f"Comma-separated subset of: {', '.join(TOPICS)}")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    topics = [t.strip() for t in args.topics.split(",")] if args.topics else None
    ok = generate_examples(args.output_dir, model=args.model, count=args.count,
                           topics=topics, overwrite=args.overwrite)
    sys.exit(0 if ok > 0 else 1)


if __name__ == "__main__":
    main()
