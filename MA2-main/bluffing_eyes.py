"""Minimal starter to launch the tabletop Kivy application."""

from __future__ import annotations

import argparse
from typing import Sequence

from tabletop.app import main as app_main


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the experiment launcher."""

    parser = argparse.ArgumentParser(description="Start the Bluffing Eyes tabletop app")
    parser.add_argument(
        "--session",
        type=int,
        required=False,
        default=None,
        help="Optional: Session-ID. Wenn nicht gesetzt, fragt die UI.",
    )
    parser.add_argument(
        "--block",
        type=int,
        required=False,
        default=None,
        help="Optional: einzelner Block. Wenn nicht gesetzt, steuert der Code die Blöcke.",
    )
    parser.add_argument(
        "--player",
        type=str,
        default="VP1",
        choices=("VP1", "VP2"),
        required=False,
        help="Optional: Player-Label; Default VP1.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point that wires CLI arguments into the Kivy application."""

    args = parse_args(argv)
    app_main(session=args.session, block=args.block, player=args.player)


if __name__ == "__main__":  # pragma: no cover - convenience wrapper
    main()
