"""Seed the policies vector collection with sample policy documents.

Usage: python scripts/seed_policies.py [--policies-dir config/policies]
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest policy documents into pgvector.")
    parser.add_argument(
        "--policies-dir",
        default="config/policies",
        help="Directory containing .md policy files (default: config/policies).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = _parse_args()

    try:
        from langchain_community.embeddings import FakeEmbeddings
    except ImportError:
        logging.error(
            "langchain-community not installed. "
            "For production, configure a real embedding model."
        )
        sys.exit(1)

    from fraud_risk_agent.knowledge.ingest import ingest_policies_from_dir

    embeddings = FakeEmbeddings(size=1536)
    count = ingest_policies_from_dir(args.policies_dir, embeddings)

    if count == 0:
        print(f"No .md files found in {args.policies_dir}")
        print("Create policy markdown files there first.")
        sys.exit(1)

    print(f"Ingested {count} policy chunks into pgvector.")


if __name__ == "__main__":
    main()
