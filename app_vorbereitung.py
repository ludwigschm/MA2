"""Legacy entry point forwarding to :mod:`tabletop.app`.

The original implementation bundled the Kivy :class:`TabletopApp` and the
``main`` convenience function directly inside ``app_vorbereitung.py``.
Both implementations now live inside :mod:`tabletop.app`; the names are
re-exported here so that older launch scripts importing from
``app_vorbereitung`` continue to function without modification.
"""

from tabletop.app import TabletopApp, main

__all__ = ["TabletopApp", "main"]


if __name__ == "__main__":  # pragma: no cover - transitional entry point
    main()
