"""Integration helpers for external hardware bridges."""

from .neon import NeonEyeTrackerBridge, load_neon_configuration

__all__ = ["NeonEyeTrackerBridge", "load_neon_configuration"]
