#!/usr/bin/env python3
"""
Pipeline Orchestrator
=====================

End-to-end pipeline for magnetic levitator parameter identification and KAN-PINN training.

This script orchestrates the complete two-phase pipeline:
- Phase 1: Physical parameter identification [K0, A, R0, α]
- Phase 2: KAN-PINN training (flux observer + position predictor)

Usage:
    python scripts/pipeline_identificacion_kanpinn.py --config config/pipeline_config.yaml
    python scripts/pipeline_identificacion_kanpinn.py --phase1-only
    python scripts/pipeline_identificacion_kanpinn.py --phase2-only --use-params results/parametros_optimos.json
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_phase1(config_path: str, output_dir: str):
    """
    Run Phase 1: Parameter identification.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    output_dir : str
        Output directory for results
    """
    print("\n" + "="*70)
    print("  PHASE 1: Physical Parameter Identification")
    print("="*70 + "\n")
    
    cmd = [
        sys.executable,
        "scripts/optimize_parameters.py",
        "--config", config_path,
        "--output", output_dir
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print("\nERROR: Phase 1 failed!")
        return False
    
    print("\n" + "="*70)
    print("  PHASE 1: Complete")
    print("="*70 + "\n")
    
    return True


def run_phase2(config_path: str, params_file: str, output_dir: str):
    """
    Run Phase 2: KAN-PINN training.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
    params_file : str
        Path to optimal parameters from Phase 1
    output_dir : str
        Output directory for trained models
    """
    print("\n" + "="*70)
    print("  PHASE 2: KAN-PINN Training")
    print("="*70 + "\n")
    
    # Check if parameters exist
    if not Path(params_file).exists():
        print(f"ERROR: Parameters file not found: {params_file}")
        print("Please run Phase 1 first or provide a valid parameters file.")
        return False
    
    cmd = [
        sys.executable,
        "scripts/train_kanpinn.py",
        "--config", config_path,
        "--use-params", params_file,
        "--output", output_dir
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print("\nERROR: Phase 2 failed!")
        return False
    
    print("\n" + "="*70)
    print("  PHASE 2: Complete")
    print("="*70 + "\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Two-Phase Pipeline: Parameter Identification + KAN-PINN Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python scripts/pipeline_identificacion_kanpinn.py --config config/pipeline_config.yaml
  
  # Run only Phase 1
  python scripts/pipeline_identificacion_kanpinn.py --phase1-only
  
  # Run only Phase 2 with existing parameters
  python scripts/pipeline_identificacion_kanpinn.py --phase2-only --use-params results/parametros_optimos.json
        """
    )
    
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                       help='Path to pipeline configuration file')
    parser.add_argument('--phase1-only', action='store_true',
                       help='Run only Phase 1 (parameter identification)')
    parser.add_argument('--phase2-only', action='store_true',
                       help='Run only Phase 2 (KAN-PINN training)')
    parser.add_argument('--use-params', type=str,
                       default='results/parameter_identification/parametros_optimos.json',
                       help='Path to parameters file (for Phase 2)')
    parser.add_argument('--output-phase1', type=str,
                       default='results/parameter_identification',
                       help='Output directory for Phase 1')
    parser.add_argument('--output-phase2', type=str,
                       default='results/kanpinn_training',
                       help='Output directory for Phase 2')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  MAGNETIC LEVITATOR - TWO-PHASE PIPELINE")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Phase 1 output: {args.output_phase1}")
    print(f"Phase 2 output: {args.output_phase2}")
    
    if args.phase1_only and args.phase2_only:
        print("\nERROR: Cannot specify both --phase1-only and --phase2-only")
        return 1
    
    # Determine which phases to run
    run_p1 = not args.phase2_only
    run_p2 = not args.phase1_only
    
    print(f"Running phases: {'1' if run_p1 else ''}{', 2' if run_p2 else '2'}")
    print("="*70 + "\n")
    
    # Phase 1: Parameter Identification
    if run_p1:
        success = run_phase1(args.config, args.output_phase1)
        if not success:
            print("\nPipeline aborted due to Phase 1 failure.")
            return 1
    
    # Phase 2: KAN-PINN Training
    if run_p2:
        params_file = args.use_params
        if run_p1:
            # Use parameters from Phase 1
            params_file = str(Path(args.output_phase1) / "parametros_optimos.json")
        
        success = run_phase2(args.config, params_file, args.output_phase2)
        if not success:
            print("\nPipeline aborted due to Phase 2 failure.")
            return 1
    
    # Success
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    if run_p1:
        print(f"  - Phase 1 parameters: {args.output_phase1}/parametros_optimos.json")
        print(f"  - Phase 1 results: {args.output_phase1}/")
    if run_p2:
        print(f"  - Phase 2 models: {args.output_phase2}/")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
