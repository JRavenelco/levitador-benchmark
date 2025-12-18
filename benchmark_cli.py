#!/usr/bin/env python3
"""
Levitador Benchmark CLI
=======================
Command-line interface to run optimization benchmarks.

Usage:
    python benchmark_cli.py run --algorithm DE --iterations 100
    python benchmark_cli.py list-algorithms
    python benchmark_cli.py test
    python benchmark_cli.py run --algorithm GA --pop-size 50 --visualize

Author: Jesús (Doctorado UAQ)
License: MIT
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from levitador_benchmark import LevitadorBenchmark
from example_optimization import (
    RandomSearch, DifferentialEvolution, GeneticAlgorithm,
    GreyWolfOptimizer, ArtificialBeeColony, HoneyBadgerAlgorithm,
    ShrimpOptimizer, TianjiOptimizer
)


# =============================================================================
# ALGORITHM REGISTRY
# =============================================================================

ALGORITHMS = {
    'random': {
        'name': 'Random Search',
        'class': RandomSearch,
        'description': 'Búsqueda aleatoria (baseline)',
        'params': {
            'n_iterations': {'type': int, 'default': 1000, 'help': 'Número de iteraciones'}
        }
    },
    'DE': {
        'name': 'Differential Evolution',
        'class': DifferentialEvolution,
        'description': 'Evolución Diferencial (DE/rand/1/bin)',
        'params': {
            'pop_size': {'type': int, 'default': 30, 'help': 'Tamaño de población'},
            'max_iter': {'type': int, 'default': 100, 'help': 'Número de generaciones'},
            'F': {'type': float, 'default': 0.8, 'help': 'Factor de mutación'},
            'CR': {'type': float, 'default': 0.9, 'help': 'Probabilidad de cruce'}
        }
    },
    'GA': {
        'name': 'Genetic Algorithm',
        'class': GeneticAlgorithm,
        'description': 'Algoritmo Genético con cruce BLX-alpha',
        'params': {
            'pop_size': {'type': int, 'default': 30, 'help': 'Tamaño de población'},
            'generations': {'type': int, 'default': 50, 'help': 'Número de generaciones'},
            'crossover_prob': {'type': float, 'default': 0.8, 'help': 'Probabilidad de cruce'},
            'mutation_prob': {'type': float, 'default': 0.2, 'help': 'Probabilidad de mutación'},
            'alpha': {'type': float, 'default': 0.5, 'help': 'Parámetro BLX-alpha'}
        }
    },
    'GWO': {
        'name': 'Grey Wolf Optimizer',
        'class': GreyWolfOptimizer,
        'description': 'Optimizador de Lobos Grises',
        'params': {
            'pop_size': {'type': int, 'default': 30, 'help': 'Tamaño de población'},
            'max_iter': {'type': int, 'default': 100, 'help': 'Número de iteraciones'}
        }
    },
    'ABC': {
        'name': 'Artificial Bee Colony',
        'class': ArtificialBeeColony,
        'description': 'Colonia Artificial de Abejas',
        'params': {
            'pop_size': {'type': int, 'default': 30, 'help': 'Número de fuentes de alimento'},
            'max_iter': {'type': int, 'default': 100, 'help': 'Número de iteraciones'},
            'limit': {'type': int, 'default': None, 'help': 'Límite de estancamiento (None = auto)'}
        }
    },
    'HBA': {
        'name': 'Honey Badger Algorithm',
        'class': HoneyBadgerAlgorithm,
        'description': 'Algoritmo del Tejón de Miel',
        'params': {
            'pop_size': {'type': int, 'default': 30, 'help': 'Tamaño de población'},
            'max_iter': {'type': int, 'default': 100, 'help': 'Número de iteraciones'},
            'beta': {'type': float, 'default': 6.0, 'help': 'Factor de intensidad'}
        }
    },
    'Shrimp': {
        'name': 'Shrimp Optimizer',
        'class': ShrimpOptimizer,
        'description': 'Optimizador del Camarón',
        'params': {
            'pop_size': {'type': int, 'default': 30, 'help': 'Tamaño de población'},
            'max_iter': {'type': int, 'default': 100, 'help': 'Número de iteraciones'}
        }
    },
    'Tianji': {
        'name': 'Tianji Optimizer',
        'class': TianjiOptimizer,
        'description': 'Estrategia de Carreras de Caballos Tianji',
        'params': {
            'pop_size': {'type': int, 'default': 30, 'help': 'Tamaño de población'},
            'max_iter': {'type': int, 'default': 100, 'help': 'Número de iteraciones'}
        }
    }
}


# =============================================================================
# CLI COMMANDS
# =============================================================================

def cmd_list_algorithms(args):
    """Lista todos los algoritmos disponibles."""
    print("\n" + "="*70)
    print("🧬 ALGORITMOS DISPONIBLES")
    print("="*70)
    
    for algo_id, info in ALGORITHMS.items():
        print(f"\n[{algo_id}] {info['name']}")
        print(f"  {info['description']}")
        print(f"  Parámetros:")
        for param, pinfo in info['params'].items():
            default = pinfo['default']
            print(f"    --{param.replace('_', '-')}: {pinfo['help']} (default: {default})")
    
    print("\n" + "="*70)
    print("💡 Ejemplo:")
    print("  python benchmark_cli.py run --algorithm DE --pop-size 50 --max-iter 100")
    print("="*70 + "\n")


def cmd_test(args):
    """Ejecuta un test rápido del benchmark."""
    print("\n" + "="*70)
    print("🧪 TEST RÁPIDO DEL BENCHMARK")
    print("="*70)
    
    # Crear benchmark
    print("\n[1/3] Inicializando benchmark...")
    benchmark = LevitadorBenchmark(
        datos_reales_path=args.data,
        random_seed=args.seed,
        verbose=True
    )
    
    print(f"\n{benchmark}")
    print(f"Espacio de búsqueda:")
    for name, (lb, ub) in zip(benchmark.variable_names, benchmark.bounds):
        print(f"  {name}: [{lb}, {ub}]")
    
    # Evaluar solución de referencia
    print("\n[2/3] Evaluando solución de referencia...")
    ref = benchmark.reference_solution
    error_ref = benchmark.fitness_function(ref)
    print(f"Solución de referencia: k0={ref[0]:.4f}, k={ref[1]:.4f}, a={ref[2]:.4f}")
    print(f"Error (MSE): {error_ref:.6e}")
    
    # Evaluar una solución aleatoria
    print("\n[3/3] Evaluando solución aleatoria...")
    np.random.seed(args.seed if args.seed else 42)
    random_sol = [np.random.uniform(lb, ub) for lb, ub in benchmark.bounds]
    error_rand = benchmark.fitness_function(random_sol)
    print(f"Solución aleatoria: k0={random_sol[0]:.4f}, k={random_sol[1]:.4f}, a={random_sol[2]:.4f}")
    print(f"Error (MSE): {error_rand:.6e}")
    
    print("\n" + "="*70)
    print("✅ Test completado exitosamente")
    print("="*70 + "\n")


def cmd_run(args):
    """Ejecuta un algoritmo de optimización."""
    algorithm_id = args.algorithm.upper() if args.algorithm.upper() in ALGORITHMS else args.algorithm
    
    if algorithm_id not in ALGORITHMS:
        print(f"❌ Algoritmo '{args.algorithm}' no encontrado")
        print("   Usa 'list-algorithms' para ver algoritmos disponibles")
        sys.exit(1)
    
    algo_info = ALGORITHMS[algorithm_id]
    
    print("\n" + "="*70)
    print(f"🚀 EJECUTANDO: {algo_info['name']}")
    print("="*70)
    
    # Crear benchmark
    print(f"\n[1/4] Inicializando benchmark...")
    if args.data:
        print(f"   Datos: {args.data}")
    else:
        print(f"   Datos: sintéticos")
    
    benchmark = LevitadorBenchmark(
        datos_reales_path=args.data,
        random_seed=args.seed,
        noise_level=args.noise,
        verbose=args.verbose
    )
    
    print(f"   Dimensión: {benchmark.dim}")
    print(f"   Muestras: {len(benchmark.t_real)}")
    
    # Construir parámetros del algoritmo
    print(f"\n[2/4] Configurando algoritmo...")
    algo_params = {'random_seed': args.seed, 'verbose': args.verbose}
    
    # Usar valores de args o defaults
    for param, pinfo in algo_info['params'].items():
        cli_param = param.replace('_', '-')
        value = getattr(args, param, None)
        if value is None:
            value = pinfo['default']
        algo_params[param] = value
        print(f"   {param}: {value}")
    
    # Crear instancia del algoritmo
    print(f"\n[3/4] Ejecutando optimización...")
    print(f"   Algoritmo: {algo_info['name']}")
    print(f"   Semilla: {args.seed}")
    
    algorithm = algo_info['class'](benchmark, **algo_params)
    best_solution, best_error = algorithm.optimize()
    
    # Mostrar resultados
    print(f"\n[4/4] Resultados:")
    print("="*70)
    print(f"🏆 MEJOR SOLUCIÓN ENCONTRADA")
    print("="*70)
    print(f"  k0 = {best_solution[0]:.6f} H")
    print(f"  k  = {best_solution[1]:.6f} H")
    print(f"  a  = {best_solution[2]:.6f} m")
    print(f"  Error (MSE): {best_error:.6e}")
    print(f"  Evaluaciones: {algorithm.evaluations}")
    print("="*70)
    
    # Comparar con referencia
    ref = benchmark.reference_solution
    ref_error = benchmark.fitness_function(ref)
    print(f"\n📊 Comparación con referencia:")
    print(f"  Referencia MSE: {ref_error:.6e}")
    print(f"  Mejor MSE:      {best_error:.6e}")
    if best_error < ref_error:
        mejora = ((ref_error - best_error) / ref_error) * 100
        print(f"  ✅ Mejora: {mejora:.2f}%")
    else:
        diff = ((best_error - ref_error) / ref_error) * 100
        print(f"  ⚠️  Diferencia: +{diff:.2f}%")
    
    # Guardar resultados si se especifica
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results = {
            'algorithm': algo_info['name'],
            'algorithm_id': algorithm_id,
            'parameters': algo_params,
            'solution': {
                'k0': float(best_solution[0]),
                'k': float(best_solution[1]),
                'a': float(best_solution[2])
            },
            'error_mse': float(best_error),
            'evaluations': int(algorithm.evaluations),
            'history': [float(x) for x in algorithm.history],
            'reference_solution': {
                'k0': float(ref[0]),
                'k': float(ref[1]),
                'a': float(ref[2])
            },
            'reference_error': float(ref_error),
            'seed': args.seed
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Resultados guardados: {output_path}")
    
    # Visualizar si se solicita
    if args.visualize:
        print(f"\n📊 Generando visualización...")
        try:
            viz_path = args.output.replace('.json', '.png') if args.output else 'benchmark_result.png'
            benchmark.visualize_solution(best_solution, save_path=viz_path)
            print(f"   Guardado: {viz_path}")
        except Exception as e:
            print(f"   ⚠️  Error en visualización: {e}")
    
    print("\n✅ Optimización completada\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Punto de entrada principal del CLI."""
    parser = argparse.ArgumentParser(
        description='Levitador Magnético Benchmark - CLI para optimización',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Listar algoritmos disponibles
  %(prog)s list-algorithms
  
  # Ejecutar test rápido
  %(prog)s test
  
  # Ejecutar Evolución Diferencial
  %(prog)s run --algorithm DE --pop-size 30 --max-iter 100
  
  # Ejecutar Algoritmo Genético y guardar resultados
  %(prog)s run --algorithm GA --generations 50 --output results.json
  
  # Ejecutar con datos reales y visualización
  %(prog)s run --algorithm GWO --data data/datos_levitador.txt --visualize
  
  # Ejecutar con semilla para reproducibilidad
  %(prog)s run --algorithm ABC --seed 42 --output results.json

Más información: https://github.com/JRavenelco/levitador-benchmark
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # =========================================================================
    # Comando: list-algorithms
    # =========================================================================
    parser_list = subparsers.add_parser(
        'list-algorithms',
        aliases=['list', 'ls'],
        help='Lista algoritmos disponibles'
    )
    
    # =========================================================================
    # Comando: test
    # =========================================================================
    parser_test = subparsers.add_parser(
        'test',
        help='Ejecuta un test rápido del benchmark'
    )
    parser_test.add_argument(
        '--data',
        type=str,
        help='Ruta a archivo de datos experimentales'
    )
    parser_test.add_argument(
        '--seed',
        type=int,
        help='Semilla para reproducibilidad'
    )
    
    # =========================================================================
    # Comando: run
    # =========================================================================
    parser_run = subparsers.add_parser(
        'run',
        help='Ejecuta un algoritmo de optimización'
    )
    parser_run.add_argument(
        '--algorithm', '-a',
        type=str,
        required=True,
        help='Algoritmo a ejecutar (ver list-algorithms)'
    )
    parser_run.add_argument(
        '--data',
        type=str,
        help='Ruta a archivo de datos experimentales'
    )
    parser_run.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad (default: 42)'
    )
    parser_run.add_argument(
        '--noise',
        type=float,
        default=1e-5,
        help='Nivel de ruido para datos sintéticos (default: 1e-5)'
    )
    parser_run.add_argument(
        '--output', '-o',
        type=str,
        help='Ruta para guardar resultados (JSON)'
    )
    parser_run.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generar visualización de la solución'
    )
    parser_run.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Mostrar información detallada (default: True)'
    )
    parser_run.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Modo silencioso (deshabilita verbose)'
    )
    
    # Añadir parámetros específicos de cada algoritmo
    for param_name in ['pop_size', 'max_iter', 'n_iterations', 'generations',
                       'F', 'CR', 'crossover_prob', 'mutation_prob', 'alpha',
                       'beta', 'limit']:
        parser_run.add_argument(
            f'--{param_name.replace("_", "-")}',
            type=float if param_name in ['F', 'CR', 'crossover_prob', 'mutation_prob', 'alpha', 'beta'] else int,
            help=argparse.SUPPRESS  # Ocultar en help general
        )
    
    # Parsear argumentos
    args = parser.parse_args()
    
    # Ajustar verbose si se usa --quiet
    if hasattr(args, 'quiet') and args.quiet:
        args.verbose = False
    
    # Ejecutar comando
    if args.command in ['list-algorithms', 'list', 'ls']:
        cmd_list_algorithms(args)
    elif args.command == 'test':
        cmd_test(args)
    elif args.command == 'run':
        cmd_run(args)
    else:
        parser.print_help()
        print("\n💡 Usa 'list-algorithms' para ver algoritmos disponibles")
        print("   o 'test' para ejecutar un test rápido del benchmark\n")


if __name__ == '__main__':
    main()
