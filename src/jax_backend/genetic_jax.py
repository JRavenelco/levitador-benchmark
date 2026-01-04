"""
JAX-Vectorized Genetic Algorithm
=================================

Fully vectorized GA that runs entirely on GPU.
All operations (selection, crossover, mutation) are batched.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from functools import partial
from typing import Tuple, Callable


class GeneticAlgorithmJAX:
    """
    GPU-accelerated Genetic Algorithm using JAX.
    
    All operations are JIT-compiled and vectorized:
    - Selection: Tournament selection (vectorized)
    - Crossover: BLX-α crossover (vectorized)
    - Mutation: Gaussian mutation (vectorized)
    """
    
    def __init__(self,
                 fitness_fn: Callable,
                 bounds: jnp.ndarray,
                 pop_size: int = 100,
                 elite_size: int = 5,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 mutation_scale: float = 0.1,
                 tournament_size: int = 3,
                 seed: int = 42):
        """
        Parameters
        ----------
        fitness_fn : Callable
            Vectorized fitness function: population [N, D] -> fitness [N]
        bounds : jnp.ndarray
            Parameter bounds [D, 2] where [:,0] = lower, [:,1] = upper
        pop_size : int
            Population size
        elite_size : int
            Number of elite individuals to preserve
        crossover_rate : float
            Probability of crossover
        mutation_rate : float
            Probability of mutation per gene
        mutation_scale : float
            Scale of Gaussian mutation (fraction of range)
        tournament_size : int
            Size of tournament for selection
        seed : int
            Random seed
        """
        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.dim = bounds.shape[0]
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.tournament_size = tournament_size
        
        self.key = random.PRNGKey(seed)
        
        # Precompute ranges for mutation
        self.ranges = bounds[:, 1] - bounds[:, 0]
        
    def initialize_population(self) -> jnp.ndarray:
        """Initialize random population within bounds."""
        self.key, subkey = random.split(self.key)
        
        # Uniform random in [0, 1], then scale to bounds
        pop = random.uniform(subkey, shape=(self.pop_size, self.dim))
        pop = pop * self.ranges + self.bounds[:, 0]
        
        return pop
    
    @partial(jit, static_argnums=(0,))
    def _tournament_select(self, 
                           key: jax.Array,
                           population: jnp.ndarray, 
                           fitness: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized tournament selection.
        Select pop_size parents using tournament selection.
        """
        # Generate tournament indices for all selections at once
        # Shape: [pop_size, tournament_size]
        tournament_indices = random.randint(
            key, 
            shape=(self.pop_size, self.tournament_size),
            minval=0, 
            maxval=self.pop_size
        )
        
        # Get fitness of tournament participants
        # Shape: [pop_size, tournament_size]
        tournament_fitness = fitness[tournament_indices]
        
        # Find winner (minimum fitness) in each tournament
        # Shape: [pop_size]
        winner_idx_in_tournament = jnp.argmin(tournament_fitness, axis=1)
        
        # Get the actual population index of each winner
        winner_indices = tournament_indices[jnp.arange(self.pop_size), winner_idx_in_tournament]
        
        return population[winner_indices]
    
    @partial(jit, static_argnums=(0,))
    def _crossover(self,
                   key: jax.Array,
                   parents: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized BLX-α crossover.
        Pairs consecutive parents and creates offspring.
        """
        key1, key2 = random.split(key)
        
        # Pair parents: even indices with odd indices
        p1 = parents[::2]   # [pop_size/2, dim]
        p2 = parents[1::2]  # [pop_size/2, dim]
        
        n_pairs = p1.shape[0]
        
        # BLX-alpha: offspring = p1 + alpha * (p2 - p1) where alpha in [-0.5, 1.5]
        alpha = random.uniform(key1, shape=(n_pairs, self.dim), minval=-0.5, maxval=1.5)
        
        offspring1 = p1 + alpha * (p2 - p1)
        offspring2 = p2 + alpha * (p1 - p2)
        
        # Interleave offspring
        offspring = jnp.empty((self.pop_size, self.dim))
        offspring = offspring.at[::2].set(offspring1)
        offspring = offspring.at[1::2].set(offspring2)
        
        # Apply crossover mask (some individuals don't crossover)
        crossover_mask = random.uniform(key2, shape=(self.pop_size, 1)) < self.crossover_rate
        offspring = jnp.where(crossover_mask, offspring, parents)
        
        return offspring
    
    @partial(jit, static_argnums=(0,))
    def _mutate(self,
                key: jax.Array,
                population: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized Gaussian mutation.
        """
        key1, key2 = random.split(key)
        
        # Generate mutation mask
        mutation_mask = random.uniform(key1, shape=population.shape) < self.mutation_rate
        
        # Generate Gaussian noise
        noise = random.normal(key2, shape=population.shape) * self.ranges * self.mutation_scale
        
        # Apply mutation
        mutated = population + mutation_mask * noise
        
        # Clip to bounds
        mutated = jnp.clip(mutated, self.bounds[:, 0], self.bounds[:, 1])
        
        return mutated
    
    @partial(jit, static_argnums=(0,))
    def _generation_step(self,
                         key: jax.Array,
                         population: jnp.ndarray,
                         fitness: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Single generation step (fully JIT-compiled).
        """
        key1, key2, key3 = random.split(key, 3)
        
        # Elitism: keep best individuals
        elite_indices = jnp.argsort(fitness)[:self.elite_size]
        elite = population[elite_indices]
        
        # Selection
        parents = self._tournament_select(key1, population, fitness)
        
        # Crossover
        offspring = self._crossover(key2, parents)
        
        # Mutation
        offspring = self._mutate(key3, offspring)
        
        # Replace worst with elite
        new_population = offspring.at[:self.elite_size].set(elite)
        
        # Evaluate new population
        new_fitness = self.fitness_fn(new_population)
        
        return new_population, new_fitness
    
    def run(self, n_generations: int = 100, verbose: bool = True) -> dict:
        """
        Run the genetic algorithm for n_generations.
        
        Returns
        -------
        result : dict
            'best_params': Best parameters found
            'best_fitness': Best fitness value
            'history': Fitness history per generation
            'population': Final population
        """
        # Initialize
        population = self.initialize_population()
        fitness = self.fitness_fn(population)
        
        history = []
        best_fitness = jnp.min(fitness)
        best_idx = jnp.argmin(fitness)
        best_params = population[best_idx]
        
        if verbose:
            print(f"Gen 0: Best Fitness = {best_fitness:.6e}")
        
        for gen in range(n_generations):
            self.key, subkey = random.split(self.key)
            
            population, fitness = self._generation_step(subkey, population, fitness)
            
            # Track best
            gen_best = jnp.min(fitness)
            gen_best_idx = jnp.argmin(fitness)
            
            if gen_best < best_fitness:
                best_fitness = gen_best
                best_params = population[gen_best_idx]
            
            history.append(float(gen_best))
            
            if verbose and (gen + 1) % 50 == 0:
                print(f"Gen {gen+1}: Best Fitness = {gen_best:.6e}")
        
        return {
            'best_params': best_params,
            'best_fitness': float(best_fitness),
            'history': history,
            'population': population,
            'final_fitness': fitness
        }


class DifferentialEvolutionJAX:
    """
    GPU-accelerated Differential Evolution using JAX.
    DE/rand/1/bin strategy, fully vectorized.
    """
    
    def __init__(self,
                 fitness_fn: Callable,
                 bounds: jnp.ndarray,
                 pop_size: int = 100,
                 F: float = 0.8,
                 CR: float = 0.9,
                 seed: int = 42):
        """
        Parameters
        ----------
        fitness_fn : Callable
            Vectorized fitness function
        bounds : jnp.ndarray
            Parameter bounds [D, 2]
        pop_size : int
            Population size
        F : float
            Differential weight (mutation factor)
        CR : float
            Crossover rate
        """
        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.dim = bounds.shape[0]
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        
        self.key = random.PRNGKey(seed)
        self.ranges = bounds[:, 1] - bounds[:, 0]
    
    def initialize_population(self) -> jnp.ndarray:
        self.key, subkey = random.split(self.key)
        pop = random.uniform(subkey, shape=(self.pop_size, self.dim))
        return pop * self.ranges + self.bounds[:, 0]
    
    @partial(jit, static_argnums=(0,))
    def _mutation_crossover(self,
                            key: jax.Array,
                            population: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorized DE/rand/1/bin mutation and crossover.
        """
        key1, key2, key3 = random.split(key, 3)
        
        # Generate 3 distinct random indices for each individual
        # This is a simplification - ideally indices should be distinct
        r1 = random.randint(key1, shape=(self.pop_size,), minval=0, maxval=self.pop_size)
        r2 = random.randint(key2, shape=(self.pop_size,), minval=0, maxval=self.pop_size)
        r3 = random.randint(key3, shape=(self.pop_size,), minval=0, maxval=self.pop_size)
        
        # Mutant vector: v = x_r1 + F * (x_r2 - x_r3)
        mutant = population[r1] + self.F * (population[r2] - population[r3])
        
        # Binomial crossover
        key4, key5 = random.split(key)
        crossover_mask = random.uniform(key4, shape=(self.pop_size, self.dim)) < self.CR
        
        # Ensure at least one gene is from mutant
        j_rand = random.randint(key5, shape=(self.pop_size,), minval=0, maxval=self.dim)
        j_indices = jnp.arange(self.dim)
        force_mask = (j_indices == j_rand[:, None])
        crossover_mask = crossover_mask | force_mask
        
        # Trial vector
        trial = jnp.where(crossover_mask, mutant, population)
        
        # Clip to bounds
        trial = jnp.clip(trial, self.bounds[:, 0], self.bounds[:, 1])
        
        return trial
    
    @partial(jit, static_argnums=(0,))
    def _selection(self,
                   population: jnp.ndarray,
                   trial: jnp.ndarray,
                   fitness: jnp.ndarray,
                   trial_fitness: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Vectorized selection: keep better of parent or trial.
        """
        better_mask = (trial_fitness < fitness)[:, None]
        new_pop = jnp.where(better_mask, trial, population)
        new_fitness = jnp.where(better_mask[:, 0], trial_fitness, fitness)
        return new_pop, new_fitness
    
    def run(self, n_generations: int = 100, verbose: bool = True) -> dict:
        """Run DE optimization."""
        population = self.initialize_population()
        fitness = self.fitness_fn(population)
        
        history = []
        best_fitness = jnp.min(fitness)
        best_idx = jnp.argmin(fitness)
        best_params = population[best_idx]
        
        if verbose:
            print(f"Gen 0: Best Fitness = {best_fitness:.6e}")
        
        for gen in range(n_generations):
            self.key, subkey = random.split(self.key)
            
            # Mutation + Crossover
            trial = self._mutation_crossover(subkey, population)
            
            # Evaluate trial population (VECTORIZED!)
            trial_fitness = self.fitness_fn(trial)
            
            # Selection
            population, fitness = self._selection(population, trial, fitness, trial_fitness)
            
            # Track best
            gen_best = jnp.min(fitness)
            gen_best_idx = jnp.argmin(fitness)
            
            if gen_best < best_fitness:
                best_fitness = gen_best
                best_params = population[gen_best_idx]
            
            history.append(float(gen_best))
            
            if verbose and (gen + 1) % 50 == 0:
                print(f"Gen {gen+1}: Best Fitness = {gen_best:.6e}")
        
        return {
            'best_params': best_params,
            'best_fitness': float(best_fitness),
            'history': history,
            'population': population
        }
