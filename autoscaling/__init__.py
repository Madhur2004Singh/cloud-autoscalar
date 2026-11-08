"""Autoscaling package for simulated cloud infrastructure.

This package provides base abstractions and concrete autoscaling strategies
for dynamic cloud resource allocation during live event broadcasts.
"""

from autoscaling.base_scaler import BaseScaler, ScalerConfig, ScalingDecision
from autoscaling.reactive_scaler import ReactiveScaler

__all__ = [
    "BaseScaler",
    "ScalerConfig",
    "ScalingDecision",
    "ReactiveScaler",
]
