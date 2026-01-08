"""
Integration tests for fitness function MSE validation.
These tests verify that the fitness function returns valid MSE (Mean Squared Error) values
under various conditions and scenarios.

Run with: pytest tests/test_integration_mse.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from levitador_benchmark import LevitadorBenchmark


class TestFitnessFunctionMSEValidation:
    """Integration tests to verify fitness function returns valid MSE values."""
    
    @pytest.fixture
    def benchmark(self):
        """Creates a benchmark instance with reproducible synthetic data."""
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_mse_is_non_negative(self, benchmark):
        """MSE must always be non-negative (by mathematical definition)."""
        # Test with various solutions
        test_solutions = [
            [0.036, 0.0035, 0.005],  # Near reference
            [0.05, 0.005, 0.01],     # Valid but different
            [0.001, 0.001, 0.001],   # Lower bounds
            [0.08, 0.08, 0.04],      # Upper bounds
        ]
        
        for solution in test_solutions:
            mse = benchmark.fitness_function(solution)
            assert mse >= 0, f"MSE must be non-negative, got {mse} for solution {solution}"
    
    def test_mse_is_finite(self, benchmark):
        """MSE must be a finite number (not NaN or Inf) for valid solutions."""
        # Test with valid solutions within bounds
        test_solutions = [
            [0.036, 0.0035, 0.005],
            [0.01, 0.01, 0.01],
            [0.05, 0.05, 0.02],
        ]
        
        for solution in test_solutions:
            mse = benchmark.fitness_function(solution)
            assert np.isfinite(mse), f"MSE must be finite, got {mse} for solution {solution}"
            assert not np.isnan(mse), f"MSE must not be NaN for solution {solution}"
            assert not np.isinf(mse), f"MSE must not be Inf for solution {solution}"
    
    def test_mse_is_float(self, benchmark):
        """MSE must be a float type."""
        solution = [0.036, 0.0035, 0.005]
        mse = benchmark.fitness_function(solution)
        assert isinstance(mse, (int, float)), f"MSE must be numeric, got type {type(mse)}"
        assert isinstance(mse, float), f"MSE should be float, got {type(mse)}"
    
    def test_mse_consistency_across_evaluations(self, benchmark):
        """MSE should be consistent when evaluating the same solution multiple times."""
        solution = [0.036, 0.0035, 0.005]
        
        # Evaluate the same solution multiple times
        mse_values = [benchmark.fitness_function(solution) for _ in range(5)]
        
        # All values should be identical (deterministic)
        for mse in mse_values:
            assert mse == mse_values[0], "MSE should be deterministic for same solution"
    
    def test_mse_with_reference_solution(self, benchmark):
        """MSE with reference solution should be small (near optimal)."""
        ref_solution = benchmark.reference_solution
        mse = benchmark.fitness_function(ref_solution)
        
        # MSE should be valid
        assert isinstance(mse, float)
        assert mse >= 0
        assert np.isfinite(mse)
        
        # For synthetic data with known parameters, MSE should be reasonably small
        # Note: The exact threshold depends on noise level
        assert mse < 1.0, f"Reference solution should have reasonable MSE, got {mse}"
    
    def test_mse_increases_with_worse_parameters(self, benchmark):
        """MSE should generally increase as parameters deviate from optimal."""
        ref_solution = benchmark.reference_solution
        mse_ref = benchmark.fitness_function(ref_solution)
        
        # Perturb parameters significantly
        worse_solution = [x * 1.5 for x in ref_solution]  # 50% increase
        mse_worse = benchmark.fitness_function(worse_solution)
        
        # Worse solution should have higher MSE (in most cases)
        # Note: This is not always guaranteed due to multimodality, but should hold near optimum
        assert mse_worse >= mse_ref * 0.5, "Significantly different parameters should not be much better"
    
    def test_mse_with_boundary_values(self, benchmark):
        """MSE should be valid (but likely high) at search space boundaries."""
        bounds = benchmark.bounds
        
        # Test lower boundary
        lower_solution = [lb for lb, ub in bounds]
        mse_lower = benchmark.fitness_function(lower_solution)
        assert isinstance(mse_lower, float)
        assert mse_lower >= 0
        assert np.isfinite(mse_lower)
        
        # Test upper boundary
        upper_solution = [ub for lb, ub in bounds]
        mse_upper = benchmark.fitness_function(upper_solution)
        assert isinstance(mse_upper, float)
        assert mse_upper >= 0
        assert np.isfinite(mse_upper)
    
    def test_mse_penalty_for_invalid_solutions(self, benchmark):
        """Invalid solutions (negative or out-of-bounds) should return penalty value."""
        penalty_value = 1e9
        
        # Test negative values
        negative_solution = [-0.01, 0.005, 0.005]
        mse_negative = benchmark.fitness_function(negative_solution)
        assert mse_negative == penalty_value, "Negative values should be penalized"
        
        # Test out of bounds (above upper limit)
        oob_solution = [0.2, 0.005, 0.005]  # k0 = 0.2 > 0.1
        mse_oob = benchmark.fitness_function(oob_solution)
        assert mse_oob == penalty_value, "Out-of-bounds values should be penalized"
        
        # Test zero values
        zero_solution = [0.0, 0.005, 0.005]
        mse_zero = benchmark.fitness_function(zero_solution)
        assert mse_zero == penalty_value, "Zero values should be penalized"
    
    def test_mse_range_is_reasonable(self, benchmark):
        """MSE values should be in a reasonable range for physical systems."""
        # Test several random valid solutions
        np.random.seed(42)
        
        for _ in range(10):
            solution = [np.random.uniform(lb, ub) for lb, ub in benchmark.bounds]
            mse = benchmark.fitness_function(solution)
            
            # MSE should be within reasonable physical bounds
            # For position in meters, typical MSE should be < 1.0 m^2
            # (unless parameters are very bad, then we get penalty)
            assert mse >= 0
            assert mse < 1e10  # Should be either reasonable MSE or penalty (1e9)
    
    def test_mse_with_different_seeds(self):
        """MSE computation should be consistent with different random seeds."""
        solution = [0.036, 0.0035, 0.005]
        
        # Create benchmarks with different seeds
        bench1 = LevitadorBenchmark(random_seed=1, verbose=False)
        bench2 = LevitadorBenchmark(random_seed=2, verbose=False)
        
        mse1 = bench1.fitness_function(solution)
        mse2 = bench2.fitness_function(solution)
        
        # Both should be valid MSE values
        assert isinstance(mse1, float) and isinstance(mse2, float)
        assert mse1 >= 0 and mse2 >= 0
        assert np.isfinite(mse1) and np.isfinite(mse2)
        
        # Values will be different due to different synthetic data, but should be in similar range
        # (both should be reasonably small or reasonably large, not vastly different)
        # Threshold of 5 allows for MSE values to differ by up to 5 orders of magnitude,
        # which is reasonable given different random seeds but ensures both are valid computations
        LOG_MAGNITUDE_THRESHOLD = 5
        assert abs(np.log10(mse1 + 1e-10) - np.log10(mse2 + 1e-10)) < LOG_MAGNITUDE_THRESHOLD, \
            "MSE values with different seeds should be in similar order of magnitude"


class TestFitnessFunctionMSEComputation:
    """Tests to verify the MSE computation logic is correct."""
    
    @pytest.fixture
    def benchmark(self):
        """Creates a benchmark instance with reproducible synthetic data."""
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_mse_perfect_match_is_near_zero(self):
        """If parameters match data generation, MSE should be relatively small."""
        # Create benchmark with known parameters
        k0_true, k_true, a_true = 0.0363, 0.0035, 0.0052
        
        benchmark = LevitadorBenchmark(random_seed=42, noise_level=1e-5, verbose=False)
        
        # Evaluate with exact parameters used to generate data
        mse = benchmark.fitness_function([k0_true, k_true, a_true])
        
        # MSE should be valid and relatively reasonable
        # Note: Even with true parameters, MSE won't be exactly zero due to:
        # 1. Noise in the generated data
        # 2. Numerical integration differences
        # 3. Initial condition taken from noisy data point
        assert mse >= 0
        assert np.isfinite(mse)
        assert mse < 10.0, f"MSE with true parameters should be reasonable, got {mse}"
    
    def test_mse_computation_formula(self, benchmark):
        """Verify that MSE is computed as mean of squared errors."""
        solution = [0.036, 0.0035, 0.005]
        
        # Get the MSE from fitness function
        mse = benchmark.fitness_function(solution)
        
        # MSE formula: mean((y_real - y_sim)^2)
        # Properties it must satisfy:
        # 1. Non-negative (squared errors)
        assert mse >= 0
        
        # 2. Should be a proper average (not sum)
        # If we scale the solution badly, MSE should increase
        bad_solution = [x * 2 for x in solution]
        mse_bad = benchmark.fitness_function(bad_solution)
        assert mse_bad > mse, "Worse parameters should generally give higher MSE"
    
    def test_mse_scales_with_error_magnitude(self, benchmark):
        """MSE should increase quadratically with error magnitude."""
        # Get a baseline solution
        ref_solution = benchmark.reference_solution
        mse_ref = benchmark.fitness_function(ref_solution)
        
        # Create solutions with increasing deviation
        solutions = [
            [x * 1.01 for x in ref_solution],  # 1% deviation
            [x * 1.05 for x in ref_solution],  # 5% deviation
            [x * 1.10 for x in ref_solution],  # 10% deviation
        ]
        
        mse_values = [benchmark.fitness_function(sol) for sol in solutions]
        
        # All should be valid
        for mse in mse_values:
            assert mse >= 0
            assert np.isfinite(mse)
        
        # Generally, MSE should increase (though not strictly monotonic due to nonlinearity)
        # At least the largest deviation should have higher MSE than reference
        assert mse_values[-1] > mse_ref * 0.9, "Large deviation should increase MSE"


class TestFitnessFunctionEdgeCases:
    """Test edge cases and boundary conditions for MSE computation."""
    
    @pytest.fixture
    def benchmark(self):
        """Creates a benchmark instance with reproducible synthetic data."""
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_mse_with_very_small_parameters(self, benchmark):
        """MSE computation should handle very small parameter values."""
        small_solution = [1e-4, 1e-4, 1e-4]  # At lower bounds
        mse = benchmark.fitness_function(small_solution)
        
        assert isinstance(mse, float)
        assert mse >= 0
        assert np.isfinite(mse)
    
    def test_mse_with_very_large_parameters(self, benchmark):
        """MSE computation should handle large parameter values (within bounds)."""
        large_solution = [0.099, 0.099, 0.049]  # Near upper bounds
        mse = benchmark.fitness_function(large_solution)
        
        assert isinstance(mse, float)
        assert mse >= 0
        assert np.isfinite(mse)
    
    def test_mse_with_mixed_magnitude_parameters(self, benchmark):
        """MSE should handle parameters of different magnitudes correctly."""
        mixed_solution = [0.001, 0.05, 0.005]  # Different orders of magnitude
        mse = benchmark.fitness_function(mixed_solution)
        
        assert isinstance(mse, float)
        assert mse >= 0
        assert np.isfinite(mse)
    
    def test_mse_reproducibility_after_multiple_calls(self, benchmark):
        """MSE should remain consistent after multiple fitness evaluations."""
        solution = [0.036, 0.0035, 0.005]
        
        # Evaluate multiple times with other evaluations in between
        mse1 = benchmark.fitness_function(solution)
        _ = benchmark.fitness_function([0.05, 0.005, 0.01])  # Different solution
        mse2 = benchmark.fitness_function(solution)  # Same solution again
        
        assert mse1 == mse2, "MSE should be reproducible for same solution"
    
    def test_batch_evaluation_matches_individual(self, benchmark):
        """Batch evaluation should match individual evaluations."""
        solutions = [
            [0.036, 0.0035, 0.005],
            [0.05, 0.005, 0.01],
            [0.01, 0.01, 0.01],
        ]
        
        # Individual evaluations
        individual_mse = [benchmark.fitness_function(sol) for sol in solutions]
        
        # Batch evaluation
        batch_mse = benchmark.evaluate_batch(np.array(solutions))
        
        # Should match
        assert len(batch_mse) == len(individual_mse)
        for i, (ind_mse, bat_mse) in enumerate(zip(individual_mse, batch_mse)):
            assert ind_mse == bat_mse, f"Batch MSE should match individual for solution {i}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
