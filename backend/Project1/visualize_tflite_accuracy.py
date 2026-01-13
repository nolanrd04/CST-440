"""
TFLite Model Accuracy Visualizer
Author: CST-440 Team
Date: January 13, 2026

Creates visualizations showing the accuracy of trig_model_all.tflite
compared to true trigonometric function values.

Usage:
    python visualize_tflite_accuracy.py

    # With custom number of samples
    python visualize_tflite_accuracy.py --samples 500

    # Save to custom output file
    python visualize_tflite_accuracy.py --output my_results.png
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys


class TFLiteModelVisualizer:
    """Visualizes accuracy of TensorFlow Lite models."""

    def __init__(self, model_path):
        """
        Initialize visualizer with TFLite model.

        Args:
            model_path: Path to .tflite model file
        """
        self.model_path = model_path
        self.interpreter = None
        self.is_quantized = False
        self.input_scale = None
        self.input_zero_point = None
        self.output_scale = None
        self.output_zero_point = None
        self._load_model()

    def _load_model(self):
        """Load the TFLite model."""
        print(f"{'='*70}")
        print("Loading TFLite Model")
        print(f"{'='*70}")

        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found: {self.model_path}")
            sys.exit(1)

        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()

            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()

            print(f"✓ Model loaded: {self.model_path}")

            # Check if model is quantized
            input_dtype = input_details[0]['dtype']
            output_dtype = output_details[0]['dtype']

            if input_dtype == np.int8 or output_dtype == np.int8:
                self.is_quantized = True

                # Get quantization parameters
                input_quant = input_details[0]['quantization_parameters']
                output_quant = output_details[0]['quantization_parameters']

                self.input_scale = input_quant['scales'][0]
                self.input_zero_point = input_quant['zero_points'][0]
                self.output_scale = output_quant['scales'][0]
                self.output_zero_point = output_quant['zero_points'][0]

                print(f"  ⚠️  QUANTIZED MODEL (int8)")
                print(f"  Using automatic quantization-aware inference")
            else:
                print(f"  ✓ Float32 model (not quantized)")

            size_bytes = os.path.getsize(self.model_path)
            print(f"  Model size: {size_bytes:,} bytes ({size_bytes/1024:.2f} KB)")

        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def predict(self, x_input):
        """Run inference on input data with automatic quantization handling."""
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        predictions = []
        for i in range(len(x_input)):
            input_data = x_input[i:i+1]

            # Quantize input if model expects int8
            if self.is_quantized and input_details[0]['dtype'] == np.int8:
                input_data = (input_data / self.input_scale + self.input_zero_point).astype(np.int8)
            else:
                input_data = input_data.astype(input_details[0]['dtype'])

            # Run inference
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(output_details[0]['index'])[0][0]

            # Dequantize output if model returns int8
            if self.is_quantized and output_details[0]['dtype'] == np.int8:
                output = (output.astype(np.float32) - self.output_zero_point) * self.output_scale

            predictions.append(output)

        return np.array(predictions)

    def generate_test_data(self, num_samples=500):
        """Generate test data for all three trig functions."""
        print(f"\n{'='*70}")
        print(f"Generating Test Data ({num_samples} x-values per function)")
        print(f"{'='*70}")

        x_base = np.linspace(-3.14, 3.14, num_samples)
        X_samples = []
        y_samples = []
        x_raw_values = []
        func_labels = []

        for x_val in x_base:
            x_norm = (x_val + 3.14) / (2 * 3.14)

            # sin
            X_samples.append([x_norm, 1, 0, 0])
            y_samples.append(np.sin(x_val))
            x_raw_values.append(x_val)
            func_labels.append('sin')

            # cos
            X_samples.append([x_norm, 0, 1, 0])
            y_samples.append(np.cos(x_val))
            x_raw_values.append(x_val)
            func_labels.append('cos')

            # tan (safe regions only)
            tan_val = np.tan(x_val)
            if np.abs(tan_val) < 3:
                X_samples.append([x_norm, 0, 0, 1])
                y_samples.append(tan_val)
                x_raw_values.append(x_val)
                func_labels.append('tan')

        X_test = np.array(X_samples, dtype=np.float32)
        y_test = np.array(y_samples, dtype=np.float32)
        x_raw = np.array(x_raw_values, dtype=np.float32)

        print(f"✓ Generated {len(X_test)} test samples")
        return X_test, y_test, x_raw, func_labels

    def visualize_accuracy(self, X_test, y_test, x_raw, func_labels, output_file='tflite_accuracy_visualization.png'):
        """Create comprehensive accuracy visualizations."""
        print(f"\n{'='*70}")
        print("Running Model Inference")
        print(f"{'='*70}")

        # Get predictions
        y_pred = self.predict(X_test)

        # Calculate errors
        abs_errors = np.abs(y_test - y_pred)

        # Calculate relative errors (as percentage, for display only)
        relative_errors = np.where(
            np.abs(y_test) > 0.01,
            (abs_errors / np.abs(y_test)) * 100,  # Relative error as percentage
            abs_errors * 100  # For near-zero values
        )

        print(f"✓ Inference complete")
        print(f"\nCreating visualizations...")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # Define colors for each function
        colors = {'sin': '#1f77b4', 'cos': '#ff7f0e', 'tan': '#2ca02c'}

        # 1. Predictions vs True Values (for each function)
        for idx, func_name in enumerate(['sin', 'cos', 'tan']):
            ax = plt.subplot(3, 4, idx + 1)

            mask = np.array([label == func_name for label in func_labels])
            func_x = x_raw[mask]
            func_y_test = y_test[mask]
            func_y_pred = y_pred[mask]

            # Sort by x for line plots
            sort_idx = np.argsort(func_x)

            ax.plot(func_x[sort_idx], func_y_test[sort_idx],
                   label='True', color='black', linewidth=2, alpha=0.7)
            ax.plot(func_x[sort_idx], func_y_pred[sort_idx],
                   label='Predicted', color=colors[func_name],
                   linewidth=1.5, linestyle='--', alpha=0.8)

            ax.set_xlabel('x')
            ax.set_ylabel(f'{func_name}(x)')
            ax.set_title(f'{func_name}(x): Prediction vs True')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Absolute Error vs x (for each function)
        for idx, func_name in enumerate(['sin', 'cos', 'tan']):
            ax = plt.subplot(3, 4, idx + 5)

            mask = np.array([label == func_name for label in func_labels])
            func_x = x_raw[mask]
            func_abs_errors = abs_errors[mask]

            ax.scatter(func_x, func_abs_errors, alpha=0.5,
                      color=colors[func_name], s=10)
            ax.axhline(y=0.05, color='red', linestyle='--',
                      linewidth=1, label='0.05 threshold')

            ax.set_xlabel('x')
            ax.set_ylabel('Absolute Error')
            ax.set_title(f'{func_name}(x): Absolute Error')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. Error Distribution Histograms (for each function)
        for idx, func_name in enumerate(['sin', 'cos', 'tan']):
            ax = plt.subplot(3, 4, idx + 9)

            mask = np.array([label == func_name for label in func_labels])
            func_abs_errors = abs_errors[mask]

            ax.hist(func_abs_errors, bins=30, color=colors[func_name],
                   alpha=0.7, edgecolor='black')
            ax.axvline(x=np.mean(func_abs_errors), color='red',
                      linestyle='--', linewidth=2, label=f'Mean: {np.mean(func_abs_errors):.4f}')

            ax.set_xlabel('Absolute Error')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{func_name}(x): Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Overall Comparison - Accuracy Summary
        ax = plt.subplot(3, 4, 4)

        accuracies = []
        mae_values = []
        function_names = []

        for func_name in ['sin', 'cos', 'tan']:
            mask = np.array([label == func_name for label in func_labels])
            func_abs_errors = abs_errors[mask]

            # Use absolute error for accuracy: % within 0.05 error
            accuracy = np.mean(func_abs_errors < 0.05) * 100
            mae = np.mean(func_abs_errors)

            accuracies.append(accuracy)
            mae_values.append(mae)
            function_names.append(func_name)

        x_pos = np.arange(len(function_names))
        bars = ax.bar(x_pos, accuracies, color=[colors[f] for f in function_names], alpha=0.7, edgecolor='black')

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Comparison\n(within 0.05 absolute error)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(function_names)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 5. Overall Comparison - MAE
        ax = plt.subplot(3, 4, 8)

        bars = ax.bar(x_pos, mae_values, color=[colors[f] for f in function_names], alpha=0.7, edgecolor='black')

        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('MAE Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(function_names)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (bar, mae) in enumerate(zip(bars, mae_values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mae:.4f}', ha='center', va='bottom', fontsize=9)

        # 6. Relative Error vs x (combined plot)
        ax = plt.subplot(3, 4, 12)

        for func_name in ['sin', 'cos', 'tan']:
            mask = np.array([label == func_name for label in func_labels])
            func_x = x_raw[mask]
            func_rel_errors = relative_errors[mask]  # already as percentage

            ax.scatter(func_x, func_rel_errors, alpha=0.4,
                      color=colors[func_name], s=10, label=func_name)

        ax.axhline(y=5, color='red', linestyle='--', linewidth=1, label='5% threshold')
        ax.set_xlabel('x')
        ax.set_ylabel('Relative Error (%)')
        ax.set_title('Relative Error: All Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 20])

        plt.suptitle('TFLite Model Accuracy Analysis', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_file}")

        # Calculate and print summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")

        for func_name in ['sin', 'cos', 'tan']:
            mask = np.array([label == func_name for label in func_labels])
            func_abs_errors = abs_errors[mask]
            func_rel_errors = relative_errors[mask]  # already as percentage

            # Use absolute error for accuracy
            accuracy = np.mean(func_abs_errors < 0.05) * 100
            mae = np.mean(func_abs_errors)
            max_error = np.max(func_abs_errors)
            mean_rel_error = np.mean(func_rel_errors)

            print(f"\n{func_name}(x):")
            print(f"  Accuracy: {accuracy:.2f}% (within 0.05 absolute error)")
            print(f"  Mean Relative Error: {mean_rel_error:.2f}%")
            print(f"  MAE: {mae:.6f}")
            print(f"  Max Error: {max_error:.6f}")

        print(f"\n{'='*70}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize TFLite model accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', '-m', default='trig_model_all.tflite',
                       help='Path to TFLite model (default: trig_model_all.tflite)')
    parser.add_argument('--samples', '-s', type=int, default=500,
                       help='Number of x values to test per function (default: 500)')
    parser.add_argument('--output', '-o', default='tflite_accuracy_visualization.png',
                       help='Output file for visualization (default: tflite_accuracy_visualization.png)')

    args = parser.parse_args()

    # Create visualizer
    visualizer = TFLiteModelVisualizer(args.model)

    # Generate test data
    X_test, y_test, x_raw, func_labels = visualizer.generate_test_data(num_samples=args.samples)

    # Create visualizations
    visualizer.visualize_accuracy(X_test, y_test, x_raw, func_labels, output_file=args.output)

    print(f"\nDone! Open {args.output} to view the results.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())