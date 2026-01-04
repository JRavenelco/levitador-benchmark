"""
JAX Backend for Vectorized GPU Computation
==========================================

This module provides JAX-accelerated implementations for:
- Vectorized physics simulation (vmap over population)
- Vectorized genetic algorithm operations
"""

from .physics_jax import LevitadorPhysicsJAX, vectorized_fitness
from .genetic_jax import GeneticAlgorithmJAX

__all__ = ['LevitadorPhysicsJAX', 'vectorized_fitness', 'GeneticAlgorithmJAX']
