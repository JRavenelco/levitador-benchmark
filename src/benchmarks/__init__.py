"""
Benchmarks Module
=================

This module contains benchmarking utilities for:
1. Physical parameter identification (Phase 1)
2. KAN-PINN hyperparameter optimization (Phase 2)
"""

from .parameter_benchmark import ParameterBenchmark

__all__ = ['ParameterBenchmark']
