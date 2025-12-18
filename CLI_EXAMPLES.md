# 🖥️ CLI Examples - Levitador Benchmark

This document provides comprehensive examples of using the command-line interface for the Levitador Magnético Benchmark.

## 📋 Table of Contents

1. [Basic Usage](#basic-usage)
2. [Running Algorithms](#running-algorithms)
3. [Advanced Options](#advanced-options)
4. [Output and Visualization](#output-and-visualization)
5. [Reproducibility](#reproducibility)
6. [Algorithm Comparison](#algorithm-comparison)

---

## Basic Usage

### List Available Algorithms

```bash
python benchmark_cli.py list-algorithms
```

Output shows all 8 available algorithms with their parameters:
- Random Search (baseline)
- Differential Evolution (DE)
- Genetic Algorithm (GA)
- Grey Wolf Optimizer (GWO)
- Artificial Bee Colony (ABC)
- Honey Badger Algorithm (HBA)
- Shrimp Optimizer
- Tianji Optimizer

### Quick Test

Run a quick test to verify the benchmark is working:

```bash
python benchmark_cli.py test
```

With custom seed:
```bash
python benchmark_cli.py test --seed 42
```

---

## Running Algorithms

### Random Search (Baseline)

Simple random search as a baseline:

```bash
python benchmark_cli.py run --algorithm random --n-iterations 1000
```

### Differential Evolution

Classic DE algorithm:

```bash
python benchmark_cli.py run --algorithm DE --pop-size 30 --max-iter 100
```

With custom parameters:
```bash
python benchmark_cli.py run --algorithm DE \
  --pop-size 50 \
  --max-iter 200 \
  --F 0.7 \
  --CR 0.85
```

### Genetic Algorithm

```bash
python benchmark_cli.py run --algorithm GA --generations 100
```

With full parameters:
```bash
python benchmark_cli.py run --algorithm GA \
  --pop-size 50 \
  --generations 100 \
  --crossover-prob 0.9 \
  --mutation-prob 0.1 \
  --alpha 0.3
```

### Grey Wolf Optimizer

```bash
python benchmark_cli.py run --algorithm GWO --pop-size 30 --max-iter 100
```

### Artificial Bee Colony

```bash
python benchmark_cli.py run --algorithm ABC --pop-size 40 --max-iter 100
```

With custom limit:
```bash
python benchmark_cli.py run --algorithm ABC \
  --pop-size 40 \
  --max-iter 100 \
  --limit 50
```

### Honey Badger Algorithm

```bash
python benchmark_cli.py run --algorithm HBA --pop-size 30 --max-iter 100
```

With custom beta:
```bash
python benchmark_cli.py run --algorithm HBA \
  --pop-size 30 \
  --max-iter 100 \
  --beta 8.0
```

### Shrimp Optimizer

```bash
python benchmark_cli.py run --algorithm Shrimp --pop-size 30 --max-iter 100
```

### Tianji Optimizer

```bash
python benchmark_cli.py run --algorithm Tianji --pop-size 30 --max-iter 100
```

---

## Advanced Options

### Using Real Experimental Data

If you have experimental data:

```bash
python benchmark_cli.py run \
  --algorithm DE \
  --data data/datos_levitador.txt \
  --pop-size 30 \
  --max-iter 100
```

### Custom Noise Level (Synthetic Data)

Control noise in synthetic data:

```bash
python benchmark_cli.py run \
  --algorithm GA \
  --noise 1e-4 \
  --generations 50
```

### Quiet Mode

For automation or scripts, use quiet mode:

```bash
python benchmark_cli.py run --algorithm random --n-iterations 500 --quiet
```

---

## Output and Visualization

### Save Results to JSON

```bash
python benchmark_cli.py run \
  --algorithm DE \
  --pop-size 30 \
  --max-iter 100 \
  --output results/de_run_001.json
```

JSON output includes:
- Algorithm name and ID
- All parameters used
- Best solution found (k0, k, a)
- Error (MSE)
- Number of evaluations
- Full convergence history
- Reference solution for comparison

### Generate Visualization

Create a comparison plot of simulation vs. data:

```bash
python benchmark_cli.py run \
  --algorithm GA \
  --generations 50 \
  --output results/ga_run.json \
  --visualize
```

This creates two files:
- `results/ga_run.json` - Results data
- `results/ga_run.png` - Visualization plot

### Combined: Output + Visualization

```bash
python benchmark_cli.py run \
  --algorithm GWO \
  --pop-size 30 \
  --max-iter 100 \
  --output results/gwo_experiment.json \
  --visualize \
  --quiet
```

---

## Reproducibility

### Using Seeds

For reproducible results, always specify a seed:

```bash
python benchmark_cli.py run --algorithm DE --seed 42
```

Run the same experiment multiple times:

```bash
# Run 1
python benchmark_cli.py run --algorithm GA --seed 42 --output run1.json

# Run 2 (identical results)
python benchmark_cli.py run --algorithm GA --seed 42 --output run2.json
```

### Different Seeds for Comparison

```bash
# Seed 1
python benchmark_cli.py run --algorithm DE --seed 1 --output de_seed1.json --quiet

# Seed 2
python benchmark_cli.py run --algorithm DE --seed 2 --output de_seed2.json --quiet

# Seed 3
python benchmark_cli.py run --algorithm DE --seed 3 --output de_seed3.json --quiet
```

---

## Algorithm Comparison

### Quick Comparison Script

Create a simple comparison by running multiple algorithms:

```bash
#!/bin/bash
# Run all algorithms with same configuration
ALGORITHMS=("random" "DE" "GA" "GWO" "ABC" "HBA" "Shrimp" "Tianji")

for algo in "${ALGORITHMS[@]}"; do
    echo "Running $algo..."
    if [ "$algo" == "random" ]; then
        python benchmark_cli.py run \
            --algorithm $algo \
            --n-iterations 1000 \
            --seed 42 \
            --output results/${algo}_results.json \
            --quiet
    else
        python benchmark_cli.py run \
            --algorithm $algo \
            --pop-size 30 \
            --max-iter 100 \
            --seed 42 \
            --output results/${algo}_results.json \
            --quiet
    fi
done

echo "All algorithms completed!"
```

### Parameter Sweep

Test different population sizes:

```bash
for pop in 10 20 30 40 50; do
    python benchmark_cli.py run \
        --algorithm DE \
        --pop-size $pop \
        --max-iter 50 \
        --seed 42 \
        --output results/de_pop${pop}.json \
        --quiet
done
```

---

## 💡 Pro Tips

1. **Start Small**: Use small populations and few iterations for testing, then scale up.

   ```bash
   # Quick test
   python benchmark_cli.py run --algorithm DE --pop-size 10 --max-iter 10 --quiet
   
   # Full run
   python benchmark_cli.py run --algorithm DE --pop-size 50 --max-iter 200
   ```

2. **Use Quiet Mode for Automation**: When running multiple experiments in scripts.

3. **Always Set Seeds**: For reproducible research.

4. **Save Results**: Use `--output` to keep track of all experiments.

5. **Visualize Important Runs**: Use `--visualize` to create plots for papers/presentations.

6. **Compare Multiple Seeds**: Run with different seeds to assess algorithm robustness.

---

## 🔍 Troubleshooting

### Algorithm Not Found

```bash
# ❌ Wrong
python benchmark_cli.py run --algorithm de

# ✅ Correct (case-sensitive)
python benchmark_cli.py run --algorithm DE
```

### Invalid Parameters

Check available parameters with:
```bash
python benchmark_cli.py list-algorithms
```

### Memory Issues

Reduce population size or iterations:
```bash
python benchmark_cli.py run --algorithm GA --pop-size 15 --generations 30
```

---

## 📚 Additional Resources

- Main README: [README.md](README.md)
- Python API: [levitador_benchmark.py](levitador_benchmark.py)
- Algorithm Implementations: [example_optimization.py](example_optimization.py)
- Tests: [tests/test_cli.py](tests/test_cli.py)

---

## 🤝 Contributing

Found a bug or have a suggestion? Please open an issue or submit a pull request!

**Repository**: https://github.com/JRavenelco/levitador-benchmark
