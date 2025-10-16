"""Legacy entry point forwarding to :mod:`tabletop.app`.

This module remains for backwards compatibility with earlier tooling that
imported or executed ``app_vorbereitung.py`` directly. Prefer running the
application via ``python -m tabletop.app`` or ``python main.py``.
"""

from tabletop.app import main


if __name__ == "__main__":  # pragma: no cover - transitional entry point
    main()
