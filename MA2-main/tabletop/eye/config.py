"""Configuration helpers for Neon eye tracker integration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from tabletop.data.config import ROOT


def _parse_key_value_lines(lines: Iterable[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


@dataclass(frozen=True)
class NeonConfig:
    """Resolved Neon headset configuration for a single player."""

    player: str
    ip: str = ""
    port: int = 8080
    device_id: Optional[str] = None
    enabled: bool = False

    @property
    def base_url(self) -> str:
        if not self.enabled or not self.ip:
            return ""
        port_segment = f":{self.port}" if self.port else ""
        return f"http://{self.ip}{port_segment}".rstrip(":")

    @classmethod
    def load_all(cls, path: Optional[Path] = None) -> Dict[str, "NeonConfig"]:
        if path is None:
            path = ROOT / "neon_devices.txt"

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return {}
        except OSError:
            return {}

        parsed = _parse_key_value_lines(lines)
        configs: Dict[str, NeonConfig] = {}

        for player_label in ("VP1", "VP2"):
            ip = parsed.get(f"{player_label}_IP", "").strip()
            device_id = parsed.get(f"{player_label}_ID", "").strip() or None
            port_value = parsed.get(f"{player_label}_PORT", "").strip()
            if ip:
                try:
                    port = int(port_value) if port_value else 8080
                except ValueError:
                    port = 8080
                configs[player_label] = cls(
                    player=player_label,
                    ip=ip,
                    port=port,
                    device_id=device_id,
                    enabled=True,
                )
            else:
                configs[player_label] = cls(player=player_label)

        return configs

    @classmethod
    def for_player(
        cls, player: str, path: Optional[Path] = None
    ) -> "NeonConfig":
        player_key = (player or "").strip().upper() or "VP1"
        configs = cls.load_all(path=path)
        return configs.get(player_key, cls(player=player_key))
