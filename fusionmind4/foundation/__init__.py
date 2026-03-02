"""UPFM — Universal Plasma Foundation Model (Patent Family PF3).

Dimensionless tokenization for cross-device transfer learning.
Core insight: Plasmas with identical βN, ν*, ρ*, q95, H98
behave identically regardless of device.
"""
from .core import (
    DimensionlessTokenizer,
    CrossDeviceValidator,
    PlasmaFoundationModel,
    DIMENSIONLESS_TOKENS,
    DEVICES,
)

__all__ = [
    "DimensionlessTokenizer",
    "CrossDeviceValidator",
    "PlasmaFoundationModel",
    "DIMENSIONLESS_TOKENS",
    "DEVICES",
]
