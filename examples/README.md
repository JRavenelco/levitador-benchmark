# Example Scripts - Levitador Benchmark

This directory contains standalone example scripts demonstrating how to use different optimization algorithms with the Levitador Magnético benchmark.

## 📁 Available Examples

### 1. Particle Swarm Optimization (PSO)
**File:** `example_pso.py`

PSO is inspired by the social behavior of bird flocking or fish schooling. Each particle represents a potential solution that moves through the search space influenced by its own best known position and the swarm's best known position.

**Run:**
```bash
python examples/example_pso.py
```

**Key Parameters:**
- `n_particles`: Number of particles in the swarm (default: 30)
- `max_iter`: Maximum number of iterations (default: 100)
- `w`: Inertia weight, controls exploration vs exploitation (default: 0.7)
- `c1`: Cognitive coefficient, personal best influence (default: 1.5)
- `c2`: Social coefficient, global best influence (default: 1.5)

**Algorithm Overview:**
1. Initialize particles with random positions and velocities
2. Evaluate fitness of each particle
3. Update personal best (pbest) and global best (gbest)
4. Update velocities: `v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)`
5. Update positions: `x = x + v`
6. Repeat until convergence

---

### 2. Genetic Algorithm (GA)
**File:** `example_ga.py`

GA is inspired by the process of natural selection and evolution. It maintains a population of candidate solutions that evolve over generations through selection, crossover (recombination), and mutation.

**Run:**
```bash
python examples/example_ga.py
```

**Key Parameters:**
- `pop_size`: Population size (default: 50)
- `generations`: Number of generations to evolve (default: 100)
- `crossover_prob`: Probability of crossover (default: 0.8)
- `mutation_prob`: Probability of mutation (default: 0.2)
- `alpha`: BLX-α crossover parameter (default: 0.5)
- `tournament_size`: Tournament selection size (default: 3)

**Algorithm Overview:**
1. Initialize random population
2. Evaluate fitness of all individuals
3. Selection: Tournament selection to create parent pool
4. Crossover: BLX-alpha (Blend Crossover) to create offspring
5. Mutation: Gaussian mutation for diversity
6. Elitism: Preserve best individual
7. Repeat until convergence

---

### 3. Differential Evolution (DE)
**File:** `example_de.py`

DE is a population-based optimization algorithm that uses difference vectors between population members to guide the search. It's particularly effective for continuous optimization problems.

**Run:**
```bash
python examples/example_de.py
```

**Key Parameters:**
- `pop_size`: Population size (default: 30)
- `max_iter`: Maximum number of generations (default: 100)
- `F`: Mutation factor/scaling factor (default: 0.8)
- `CR`: Crossover rate (default: 0.9)

**Strategy:** DE/rand/1/bin

**Algorithm Overview:**
1. Initialize population randomly
2. For each target vector:
   - Select 3 random distinct vectors (a, b, c)
   - Create mutant: `v = a + F * (b - c)`
   - Crossover: Binomial (uniform) between target and mutant
   - Selection: Greedy (keep better fitness)
3. Repeat until convergence

---

## 🔧 Requirements

All examples require the same dependencies as the main benchmark:

```bash
pip install numpy scipy pandas matplotlib
```

Or install from the repository root:

```bash
pip install -r requirements.txt
```

---

## 📊 Expected Output

Each example script will:

1. **Load the benchmark** with synthetic data (500 data points)
2. **Configure the algorithm** with specified parameters
3. **Run optimization** showing progress every 10 iterations
4. **Display results** including:
   - Best solution found (k0, k, a parameters)
   - Best fitness (MSE)
   - Total function evaluations
   - Comparison with reference solution
5. **Generate visualization** (optional, saved as PNG file)

### Sample Output:

```
======================================================================
  PARTICLE SWARM OPTIMIZATION (PSO) - LEVITADOR BENCHMARK
======================================================================

[1/3] Creating benchmark problem...
  ✓ Benchmark loaded with 500 data points
  ✓ Search space: [(0.0001, 0.1), (0.0001, 0.1), (0.0001, 0.05)]

[2/3] Configuring PSO...
  ✓ PSO configured:
    - Particles: 30
    - Max iterations: 100
    - Inertia (w): 0.7
    - Cognitive (c1): 1.5
    - Social (c2): 1.5

[3/3] Running PSO optimization...
  Iteration  10: Best fitness = 5.507001e-04
  Iteration  20: Best fitness = 2.041205e-04
  ...

======================================================================
  RESULTS
======================================================================

🏆 Best solution found:
  k0 = 0.066951 H  (Inductancia base)
  k  = 0.051942 H  (Coeficiente de inductancia)
  a  = 0.000344 m  (Parámetro geométrico)

📊 Performance:
  Best fitness (MSE): 7.105632e-05
  Total evaluations: 3030
```

---

## 🎯 Customization

You can modify the examples to:

1. **Use real experimental data:**
   ```python
   problema = LevitadorBenchmark("data/datos_levitador.txt")
   ```

2. **Adjust algorithm parameters:**
   ```python
   # PSO with more particles and iterations
   pso = ParticleSwarmOptimizer(
       problema=problema,
       n_particles=50,
       max_iter=200,
       w=0.9,
       c1=2.0,
       c2=2.0
   )
   ```

3. **Change random seed for different runs:**
   ```python
   problema = LevitadorBenchmark(random_seed=123)
   algo = ParticleSwarmOptimizer(problema, random_seed=123)
   ```

4. **Disable visualization:**
   ```python
   # Comment out or remove the visualization section in main()
   ```

---

## 📖 Algorithm Comparison

| Algorithm | Evaluations* | Convergence | Complexity | Best For |
|-----------|-------------|-------------|------------|----------|
| **PSO**   | ~3,000      | Fast        | Low        | Continuous problems, fast exploration |
| **GA**    | ~5,000      | Medium      | Medium     | Mixed problems, good diversity |
| **DE**    | ~3,000      | Fast        | Low        | Continuous problems, reliable |

*Approximate values for the default configurations shown above.

### When to Use Each:

- **PSO**: When you need fast convergence and the problem is continuous. Good starting point.
- **GA**: When you need good exploration and the problem may have many local optima.
- **DE**: When you need reliable convergence on continuous problems. Often best performance.

---

## 🔬 Understanding the Problem

The Levitador Magnético benchmark optimizes parameters of a magnetic levitation system:

- **k0**: Base inductance [H]
- **k**: Inductance coefficient [H]  
- **a**: Geometric parameter [m]

These parameters define the inductance function:
```
L(y) = k0 + k / (1 + y/a)
```

The fitness function minimizes the Mean Squared Error (MSE) between:
- Simulated trajectory from the dynamic model
- Real experimental data

**Lower MSE = Better fit = Better solution**

---

## 📚 References

1. **PSO**: Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization. ICNN'95.
2. **GA**: Holland, J. H. (1992). Adaptation in natural and artificial systems. MIT press.
3. **DE**: Storn, R., & Price, K. (1997). Differential evolution. Journal of global optimization.

---

## 💡 Tips

1. **Start simple**: Run the examples as-is first to understand the output
2. **Experiment**: Try different parameter values to see their effect
3. **Compare**: Run multiple algorithms and compare their performance
4. **Visualize**: Look at the generated plots to validate solutions
5. **Iterate**: Use the best solution from one run as initialization for refinement

---

## 🤝 Contributing

Found a bug or want to add a new algorithm example? Pull requests are welcome!

Please ensure your example:
- Follows the same structure as existing examples
- Includes clear documentation in the docstring
- Uses the `LevitadorBenchmark` class properly
- Provides helpful output for users

---

## 📧 Support

For questions or issues:
- Open an issue on GitHub
- Check the main repository README
- Contact: jesus.santana@uaq.mx
