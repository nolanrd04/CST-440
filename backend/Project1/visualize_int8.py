"""
TFLite INT8 Model with Derived Tan - Accuracy Visualizer
Author: CST-440 Team
Date: January 15, 2026

Creates visualizations showing the accuracy of trig_model_int8.tflite
which computes tan as sin/cos (derived approach).

Usage:
    python visualize_int8_derived_tan.py

    # With custom number of samples
    python visualize_int8_derived_tan.py --samples 500

    # Save to custom output file
    python visualize_int8_derived_tan.py --output my_results.png
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys


class TFLiteInt8DerivedTanVisualizer:
    """Visualizes accuracy of INT8 TFLite model with derived tan computation."""

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
        print("Loading TFLite Model (INT8 with Derived Tan)")
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

                print(f"  ✓ QUANTIZED MODEL (int8)")
                print(f"  Input quantization: scale={self.input_scale:.6f}, zero_point={self.input_zero_point}")
                print(f"  Output quantization: scale={self.output_scale:.6f}, zero_point={self.output_zero_point}")
            else:
                print(f"  ✓ Float32 model (not quantized)")

            size_bytes = os.path.getsize(self.model_path)
            print(f"  Model size: {size_bytes:,} bytes ({size_bytes/1024:.2f} KB)")
            print(f"  Note: Model only predicts sin and cos (tan is computed as sin/cos)")

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

        # Separate data structures for sin, cos, and tan
        sin_inputs = []
        cos_inputs = []
        tan_x_values = []

        y_sin = []
        y_cos = []
        y_tan = []

        x_sin_raw = []
        x_cos_raw = []
        x_tan_raw = []

        for x_val in x_base:
            x_norm = (x_val + 3.14) / (2 * 3.14)

            # sin - input format: [x_norm, is_sin, is_cos]
            sin_inputs.append([x_norm, 1, 0])
            y_sin.append(np.sin(x_val))
            x_sin_raw.append(x_val)

            # cos
            cos_inputs.append([x_norm, 0, 1])
            y_cos.append(np.cos(x_val))
            x_cos_raw.append(x_val)

            # tan (safe regions only) - store normalized x for later computation
            tan_val = np.tan(x_val)
            if np.abs(tan_val) < 3:
                tan_x_values.append(x_norm)
                y_tan.append(tan_val)
                x_tan_raw.append(x_val)

        print(f"✓ Generated {len(sin_inputs)} sin samples")
        print(f"✓ Generated {len(cos_inputs)} cos samples")
        print(f"✓ Generated {len(tan_x_values)} tan samples (derived from sin/cos)")

        return {
            'sin_inputs': np.array(sin_inputs, dtype=np.float32),
            'cos_inputs': np.array(cos_inputs, dtype=np.float32),
            'tan_x_values': np.array(tan_x_values, dtype=np.float32),
            'y_sin': np.array(y_sin, dtype=np.float32),
            'y_cos': np.array(y_cos, dtype=np.float32),
            'y_tan': np.array(y_tan, dtype=np.float32),
            'x_sin_raw': np.array(x_sin_raw, dtype=np.float32),
            'x_cos_raw': np.array(x_cos_raw, dtype=np.float32),
            'x_tan_raw': np.array(x_tan_raw, dtype=np.float32)
        }

    def visualize_accuracy(self, test_data, output_file='int8_derived_tan_accuracy.png'):
        """Create comprehensive accuracy visualizations."""
        print(f"\n{'='*70}")
        print("Running Model Inference")
        print(f"{'='*70}")

        # Predict sin and cos
        print("Predicting sin values...")
        y_pred_sin = self.predict(test_data['sin_inputs'])

        print("Predicting cos values...")
        y_pred_cos = self.predict(test_data['cos_inputs'])

        # Compute derived tan predictions
        print("Computing derived tan values (tan = sin/cos)...")
        y_pred_tan = []

        for x_norm in test_data['tan_x_values']:
            # Predict sin for this x
            sin_input = np.array([[x_norm, 1, 0]], dtype=np.float32)
            sin_pred = self.predict(sin_input)[0]

            # Predict cos for this x
            cos_input = np.array([[x_norm, 0, 1]], dtype=np.float32)
            cos_pred = self.predict(cos_input)[0]

            # Compute tan = sin/cos
            if np.abs(cos_pred) > 0.01:
                tan_pred = sin_pred / cos_pred
            else:
                tan_pred = np.sign(sin_pred) * 10  # Large value near asymptote

            y_pred_tan.append(tan_pred)

        y_pred_tan = np.array(y_pred_tan)

        print(f"✓ Inference complete")
        print(f"\nCreating visualizations...")

        # Calculate errors for all functions
        abs_errors_sin = np.abs(test_data['y_sin'] - y_pred_sin)
        abs_errors_cos = np.abs(test_data['y_cos'] - y_pred_cos)
        abs_errors_tan = np.abs(test_data['y_tan'] - y_pred_tan)

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))

        # Define colors for each function
        colors = {'sin': '#1f77b4', 'cos': '#ff7f0e', 'tan': '#2ca02c'}

        # 1. Predictions vs True Values (for each function)
        for idx, (func_name, x_raw, y_test, y_pred) in enumerate([
            ('sin', test_data['x_sin_raw'], test_data['y_sin'], y_pred_sin),
            ('cos', test_data['x_cos_raw'], test_data['y_cos'], y_pred_cos),
            ('tan', test_data['x_tan_raw'], test_data['y_tan'], y_pred_tan)
        ]):
            ax = plt.subplot(3, 4, idx + 1)

            # Sort by x for line plots
            sort_idx = np.argsort(x_raw)

            ax.plot(x_raw[sort_idx], y_test[sort_idx],
                   label='True', color='black', linewidth=2, alpha=0.7)
            ax.plot(x_raw[sort_idx], y_pred[sort_idx],
                   label='Predicted', color=colors[func_name],
                   linewidth=1.5, linestyle='--', alpha=0.8)

            ax.set_xlabel('x')
            ax.set_ylabel(f'{func_name}(x)')
            title = f'{func_name}(x): Prediction vs True'
            if func_name == 'tan':
                title += '\n(derived from sin/cos)'
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 2. Absolute Error vs x (for each function)
        for idx, (func_name, x_raw, abs_errors) in enumerate([
            ('sin', test_data['x_sin_raw'], abs_errors_sin),
            ('cos', test_data['x_cos_raw'], abs_errors_cos),
            ('tan', test_data['x_tan_raw'], abs_errors_tan)
        ]):
            ax = plt.subplot(3, 4, idx + 5)

            ax.scatter(x_raw, abs_errors, alpha=0.5,
                      color=colors[func_name], s=10)
            ax.axhline(y=0.05, color='red', linestyle='--',
                      linewidth=1, label='0.05 threshold')

            ax.set_xlabel('x')
            ax.set_ylabel('Absolute Error')
            ax.set_title(f'{func_name}(x): Absolute Error')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 3. Error Distribution Histograms (for each function)
        for idx, (func_name, abs_errors) in enumerate([
            ('sin', abs_errors_sin),
            ('cos', abs_errors_cos),
            ('tan', abs_errors_tan)
        ]):
            ax = plt.subplot(3, 4, idx + 9)

            ax.hist(abs_errors, bins=30, color=colors[func_name],
                   alpha=0.7, edgecolor='black')
            ax.axvline(x=np.mean(abs_errors), color='red',
                      linestyle='--', linewidth=2, label=f'Mean: {np.mean(abs_errors):.4f}')

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

        for func_name, abs_errors in [
            ('sin', abs_errors_sin),
            ('cos', abs_errors_cos),
            ('tan', abs_errors_tan)
        ]:
            accuracy = np.mean(abs_errors < 0.05) * 100
            mae = np.mean(abs_errors)

            accuracies.append(accuracy)
            mae_values.append(mae)
            function_names.append(func_name)

        x_pos = np.arange(len(function_names))
        bars = ax.bar(x_pos, accuracies, color=[colors[f] for f in function_names],
                     alpha=0.7, edgecolor='black')

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

        bars = ax.bar(x_pos, mae_values, color=[colors[f] for f in function_names],
                     alpha=0.7, edgecolor='black')

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

        # 6. Combined Error Plot
        ax = plt.subplot(3, 4, 12)

        for func_name, x_raw, abs_errors in [
            ('sin', test_data['x_sin_raw'], abs_errors_sin),
            ('cos', test_data['x_cos_raw'], abs_errors_cos),
            ('tan', test_data['x_tan_raw'], abs_errors_tan)
        ]:
            ax.scatter(x_raw, abs_errors, alpha=0.4,
                      color=colors[func_name], s=10, label=func_name)

        ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='0.05 threshold')
        ax.set_xlabel('x')
        ax.set_ylabel('Absolute Error')
        ax.set_title('Absolute Error: All Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('INT8 TFLite Model with Derived Tan - Accuracy Analysis',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_file}")

        # Calculate and print summary statistics
        print(f"\n{'='*70}")
        print("SUMMARY STATISTICS")
        print(f"{'='*70}")

        for func_name, abs_errors, num_samples in [
            ('sin', abs_errors_sin, len(abs_errors_sin)),
            ('cos', abs_errors_cos, len(abs_errors_cos)),
            ('tan', abs_errors_tan, len(abs_errors_tan))
        ]:
            accuracy = np.mean(abs_errors < 0.05) * 100
            mae = np.mean(abs_errors)
            max_error = np.max(abs_errors)

            print(f"\n{func_name}(x):")
            if func_name == 'tan':
                print(f"  (Derived from sin/cos division)")
            print(f"  Accuracy: {accuracy:.2f}% (within 0.05 absolute error)")
            print(f"  MAE: {mae:.6f}")
            print(f"  Max Error: {max_error:.6f}")
            print(f"  Samples: {num_samples}")

        print(f"\n{'='*70}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize INT8 TFLite model accuracy with derived tan',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', '-m', default='trig_model_int8.tflite',
                       help='Path to TFLite model (default: trig_model_int8.tflite)')
    parser.add_argument('--samples', '-s', type=int, default=500,
                       help='Number of x values to test per function (default: 500)')
    parser.add_argument('--output', '-o', default='int8_derived_tan_accuracy.png',
                       help='Output file for visualization (default: int8_derived_tan_accuracy.png)')

    args = parser.parse_args()

    # Create visualizer
    visualizer = TFLiteInt8DerivedTanVisualizer(args.model)

    # Generate test data
    test_data = visualizer.generate_test_data(num_samples=args.samples)

    # Create visualizations
    visualizer.visualize_accuracy(test_data, output_file=args.output)

    print(f"\nDone! Open {args.output} to view the results.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())