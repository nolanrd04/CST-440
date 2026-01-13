"""
Keras Model Accuracy Evaluator
Author: CST-440 Team
Date: January 13, 2026

Evaluates the accuracy of trig_model_all.keras by comparing predictions
against true trigonometric function values.

Usage:
    python evaluate_keras_accuracy.py

    # With custom number of test samples
    python evaluate_keras_accuracy.py --samples 1000

    # With different tolerance threshold
    python evaluate_keras_accuracy.py --tolerance 0.01
"""

import tensorflow as tf
import numpy as np
import argparse
import os
import sys


class KerasModelEvaluator:
    """Evaluates accuracy of Keras models."""

    def __init__(self, model_path):
        """
        Initialize evaluator with Keras model.

        Args:
            model_path: Path to .keras model file
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the Keras model."""
        print(f"{'='*70}")
        print("Loading Keras Model")
        print(f"{'='*70}")

        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found: {self.model_path}")
            sys.exit(1)

        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)

            print(f"✓ Model loaded: {self.model_path}")
            print(f"\nModel Details:")

            # Display model architecture
            self.model.summary()

            # Get model size
            size_bytes = os.path.getsize(self.model_path)
            total_params = self.model.count_params()

            print(f"\nModel Statistics:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  File size: {size_bytes:,} bytes ({size_bytes/1024:.2f} KB)")

        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def predict(self, x_input):
        """
        Run inference on input data.

        Args:
            x_input: NumPy array of shape (n, 4) with format [x_norm, is_sin, is_cos, is_tan]

        Returns:
            NumPy array of predictions
        """
        predictions = self.model.predict(x_input, verbose=0)
        return predictions.flatten()

    def generate_test_data(self, num_samples=1000):
        """
        Generate test data for trigonometric functions.

        Args:
            num_samples: Number of x values to test (will be ~3x this for all functions)

        Returns:
            Tuple of (X_test, y_test, function_labels)
        """
        print(f"\n{'='*70}")
        print(f"Generating Test Data")
        print(f"{'='*70}")

        x_base = np.linspace(-3.14, 3.14, num_samples)
        X_samples = []
        y_samples = []
        func_labels = []

        for x_val in x_base:
            # Normalize x to [0, 1] (same as training)
            x_norm = (x_val + 3.14) / (2 * 3.14)

            # Test sin
            X_samples.append([x_norm, 1, 0, 0])
            y_samples.append(np.sin(x_val))
            func_labels.append('sin')

            # Test cos
            X_samples.append([x_norm, 0, 1, 0])
            y_samples.append(np.cos(x_val))
            func_labels.append('cos')

            # Test tan (only in safe regions where |tan(x)| < 3)
            tan_val = np.tan(x_val)
            if np.abs(tan_val) < 3:
                X_samples.append([x_norm, 0, 0, 1])
                y_samples.append(tan_val)
                func_labels.append('tan')

        X_test = np.array(X_samples, dtype=np.float32)
        y_test = np.array(y_samples, dtype=np.float32)

        print(f"✓ Generated {len(X_test)} test samples:")
        print(f"  sin samples: {func_labels.count('sin')}")
        print(f"  cos samples: {func_labels.count('cos')}")
        print(f"  tan samples: {func_labels.count('tan')}")

        return X_test, y_test, func_labels

    def evaluate_accuracy(self, X_test, y_test, func_labels, tolerance=0.05, verbose=True):
        """
        Evaluate model accuracy on test data using RELATIVE error.

        Args:
            X_test: Test inputs (n, 4)
            y_test: True outputs (n,)
            func_labels: Function type for each sample ('sin', 'cos', 'tan')
            tolerance: Relative error threshold (e.g., 0.05 = 5% error)
            verbose: Show detailed sample-by-sample results

        Returns:
            Dictionary with accuracy metrics
        """
        print(f"\n{'='*70}")
        print(f"Evaluating Model Accuracy (Relative Error)")
        print(f"{'='*70}")
        print(f"Tolerance: {tolerance*100:.1f}% relative error")
        print(f"Running inference on {len(X_test)} samples...")

        # Get predictions
        y_pred = self.predict(X_test)

        # Calculate absolute errors
        abs_errors = np.abs(y_test - y_pred)

        # Calculate relative errors with safeguard for near-zero values
        # Use relative error for |true| > 0.01, otherwise use absolute error
        threshold = 0.01
        relative_errors = np.where(
            np.abs(y_test) > threshold,
            abs_errors / np.abs(y_test),  # Relative error: |pred - true| / |true|
            abs_errors                     # Absolute error for values near zero
        )

        # Overall metrics
        overall_accuracy = np.mean(relative_errors < tolerance) * 100
        overall_mae = np.mean(abs_errors)
        overall_max_error = np.max(abs_errors)
        overall_rmse = np.sqrt(np.mean(abs_errors ** 2))
        overall_mean_rel_error = np.mean(relative_errors) * 100  # as percentage

        results = {
            'overall': {
                'accuracy': overall_accuracy,
                'mae': overall_mae,
                'max_error': overall_max_error,
                'rmse': overall_rmse,
                'mean_rel_error': overall_mean_rel_error,
                'samples': len(X_test)
            }
        }

        # Calculate per-function metrics
        for func_name in ['sin', 'cos', 'tan']:
            mask = np.array([label == func_name for label in func_labels])

            if np.sum(mask) > 0:
                func_y_test = y_test[mask]
                func_y_pred = y_pred[mask]
                func_abs_errors = abs_errors[mask]
                func_rel_errors = relative_errors[mask]

                func_accuracy = np.mean(func_rel_errors < tolerance) * 100
                func_mae = np.mean(func_abs_errors)
                func_max_error = np.max(func_abs_errors)
                func_rmse = np.sqrt(np.mean(func_abs_errors ** 2))
                func_mean_rel_error = np.mean(func_rel_errors) * 100

                results[func_name] = {
                    'accuracy': func_accuracy,
                    'mae': func_mae,
                    'max_error': func_max_error,
                    'rmse': func_rmse,
                    'mean_rel_error': func_mean_rel_error,
                    'samples': np.sum(mask)
                }

        # Print results
        print(f"\n{'='*70}")
        print("ACCURACY RESULTS")
        print(f"{'='*70}\n")

        # Per-function results
        for func_name in ['sin', 'cos', 'tan']:
            if func_name in results:
                r = results[func_name]
                print(f"{func_name}(x):")
                print(f"  Accuracy: {r['accuracy']:.2f}% (within {tolerance*100:.0f}% relative error)")
                print(f"  Mean Relative Error: {r['mean_rel_error']:.2f}%")
                print(f"  MAE: {r['mae']:.6f}")
                print(f"  RMSE: {r['rmse']:.6f}")
                print(f"  Max Error: {r['max_error']:.6f}")
                print(f"  Samples: {r['samples']}")
                print()

        # Overall results
        r = results['overall']
        print(f"Overall Performance:")
        print(f"  Accuracy: {r['accuracy']:.2f}% (within {tolerance*100:.0f}% relative error)")
        print(f"  Mean Relative Error: {r['mean_rel_error']:.2f}%")
        print(f"  MAE: {r['mae']:.6f}")
        print(f"  RMSE: {r['rmse']:.6f}")
        print(f"  Max Error: {r['max_error']:.6f}")
        print(f"  Total Samples: {r['samples']}")

        print(f"\n{'='*70}")

        # Show some example predictions if verbose
        if verbose:
            print("\nSample Predictions (first 10):")
            print(f"{'='*70}")
            print(f"{'Func':<6} {'x':<8} {'True':<10} {'Predicted':<10} {'Abs Err':<10} {'Rel Err %':<10}")
            print(f"{'-'*70}")

            for i in range(min(10, len(X_test))):
                x_denorm = X_test[i][0] * (2 * 3.14) - 3.14
                func = func_labels[i]
                true_val = y_test[i]
                pred_val = y_pred[i]
                abs_err = abs_errors[i]
                rel_err = relative_errors[i] * 100

                print(f"{func:<6} {x_denorm:>7.3f} {true_val:>9.5f} {pred_val:>9.5f} {abs_err:>9.5f} {rel_err:>9.2f}")

            print(f"{'='*70}\n")

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate Keras model accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', '-m', default='trig_model_all.keras',
                       help='Path to Keras model (default: trig_model_all.keras)')
    parser.add_argument('--samples', '-s', type=int, default=1000,
                       help='Number of x values to test per function (default: 1000)')
    parser.add_argument('--tolerance', '-t', type=float, default=0.05,
                       help='Relative error tolerance for accuracy calculation (default: 0.05 = 5%%)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress sample-by-sample output')

    args = parser.parse_args()

    # Create evaluator
    evaluator = KerasModelEvaluator(args.model)

    # Generate test data
    X_test, y_test, func_labels = evaluator.generate_test_data(num_samples=args.samples)

    # Evaluate accuracy
    results = evaluator.evaluate_accuracy(
        X_test, y_test, func_labels,
        tolerance=args.tolerance,
        verbose=not args.quiet
    )

    # Return success
    return 0


if __name__ == "__main__":
    sys.exit(main())