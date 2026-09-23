"""CLI entry point for the nightly batch scanner.

Usage:
    python scripts/run_batch.py [--batch-run-id UUID] [--dry-run] [--max-customers N]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nightly fraud-risk batch scan.")
    parser.add_argument(
        "--batch-run-id",
        default=None,
        help="UUID for idempotency. Omit to generate a new one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Find candidates but do not run assessments.",
    )
    parser.add_argument(
        "--max-customers",
        type=int,
        default=None,
        help="Cap on the number of customers to process.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max parallel assessments (default: 5).",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    from fraud_risk_agent.batch.nightly import run_batch

    result = await run_batch(
        batch_run_id=args.batch_run_id,
        max_customers=args.max_customers,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
    )

    print(f"\n{'='*50}")
    print(f"Batch run: {result.batch_run_id}")
    print(f"Candidates found: {result.total_candidates}")
    print(f"Processed: {result.processed}")
    print(f"Skipped (already done): {result.skipped}")
    print(f"Errors: {result.errors}")
    print(f"Duration: {result.duration_seconds:.1f}s")
    print(f"{'='*50}\n")

    if result.error_details:
        print("Error details:")
        for detail in result.error_details:
            print(f"  - {detail}")

    return 1 if result.errors > 0 else 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
