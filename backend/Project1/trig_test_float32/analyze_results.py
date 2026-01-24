#!/usr/bin/env python3
"""
Analyze Arduino trig model output and generate comparison graphs.
Reads CSV data from Arduino serial output and compares predictions to actual values.
"""

import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend for GUI window

import numpy as np
import matplotlib.pyplot as plt
import serial
import time
import sys

def read_arduino_data(port='/dev/ttyACM0', baudrate=115200, timeout=30):
    """Read data from Arduino serial port."""
    print(f"Connecting to {port}...")
    print("(If Arduino doesn't respond, press the reset button)")

    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(3)  # Wait for Arduino to reset after serial connection

        print("Waiting for data...")
        data_lines = []
        collecting = False
        start_time = time.time()

        while time.time() - start_time < timeout:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                if "DATA_START" in line:
                    collecting = True
                    print("Receiving data...")
                    continue
                elif "DATA_END" in line:
                    print(f"Data received! ({len(data_lines)} points)")
                    break
                elif collecting and line and ',' in line:
                    data_lines.append(line)

        ser.close()

        if len(data_lines) == 0:
            print("No data received within timeout.")
            return None

        return data_lines

    except serial.SerialException as e:
        print(f"Serial error: {e}")
        return None

def parse_data(data_lines):
    """Parse CSV data lines into numpy arrays."""
    angles = []
    sin_pred = []
    cos_pred = []
    tan_pred = []

    for line in data_lines:
        try:
            parts = line.split(',')
            if len(parts) == 4:
                angles.append(float(parts[0]))
                sin_pred.append(float(parts[1]))
                cos_pred.append(float(parts[2]))
                tan_pred.append(float(parts[3]))
        except ValueError:
            continue

    return (np.array(angles), np.array(sin_pred),
            np.array(cos_pred), np.array(tan_pred))

def calculate_errors(predicted, actual, name):
    """Calculate and print error metrics."""
    abs_errors = np.abs(predicted - actual)
    mae = np.mean(abs_errors)
    max_err = np.max(abs_errors)

    # Accuracy within tolerance
    tolerance = 0.01
    accuracy = np.mean(abs_errors < tolerance) * 100

    print(f"\n{name}:")
    print(f"  MAE: {mae:.6f}")
    print(f"  Max Error: {max_err:.6f}")
    print(f"  Accuracy (<{tolerance}): {accuracy:.1f}%")

    return mae, max_err, accuracy

def create_graphs(angles, sin_pred, cos_pred, tan_pred):
    """Create comparison graphs."""
    # Calculate actual values
    sin_actual = np.sin(angles)
    cos_actual = np.cos(angles)
    tan_actual = np.tan(angles)

    # Calculate errors
    print("\n" + "="*50)
    print("ERROR ANALYSIS")
    print("="*50)

    sin_mae, _, sin_acc = calculate_errors(sin_pred, sin_actual, "SIN")
    cos_mae, _, cos_acc = calculate_errors(cos_pred, cos_actual, "COS")

    # For tan, exclude near-asymptote values
    tan_mask = np.abs(tan_actual) < 3
    if np.any(tan_mask):
        tan_mae, _, tan_acc = calculate_errors(
            tan_pred[tan_mask], tan_actual[tan_mask], "TAN (|actual| < 3)")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sin plot
    ax1 = axes[0, 0]
    ax1.plot(angles, sin_actual, 'b-', linewidth=2, label='Actual sin(x)')
    ax1.plot(angles, sin_pred, 'r--', linewidth=2, label='Predicted sin(x)')
    ax1.set_xlabel('Angle (radians)')
    ax1.set_ylabel('Value')
    ax1.set_title(f'Sin(x) - MAE: {sin_mae:.4f}, Accuracy: {sin_acc:.1f}%')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-3.5, 3.5])

    # Cos plot
    ax2 = axes[0, 1]
    ax2.plot(angles, cos_actual, 'b-', linewidth=2, label='Actual cos(x)')
    ax2.plot(angles, cos_pred, 'r--', linewidth=2, label='Predicted cos(x)')
    ax2.set_xlabel('Angle (radians)')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Cos(x) - MAE: {cos_mae:.4f}, Accuracy: {cos_acc:.1f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-3.5, 3.5])

    # Tan plot (clipped for visibility)
    ax3 = axes[1, 0]
    tan_actual_clipped = np.clip(tan_actual, -5, 5)
    tan_pred_clipped = np.clip(tan_pred, -5, 5)
    ax3.plot(angles, tan_actual_clipped, 'b-', linewidth=2, label='Actual tan(x)')
    ax3.plot(angles, tan_pred_clipped, 'r--', linewidth=2, label='Predicted tan(x)')
    ax3.set_xlabel('Angle (radians)')
    ax3.set_ylabel('Value (clipped to [-5, 5])')
    ax3.set_title('Tan(x) = Sin(x)/Cos(x) [derived]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-3.5, 3.5])
    ax3.set_ylim([-5, 5])

    # Error plot
    ax4 = axes[1, 1]
    sin_errors = np.abs(sin_pred - sin_actual)
    cos_errors = np.abs(cos_pred - cos_actual)
    ax4.plot(angles, sin_errors, 'r-', linewidth=1.5, label='Sin error', alpha=0.8)
    ax4.plot(angles, cos_errors, 'b-', linewidth=1.5, label='Cos error', alpha=0.8)
    ax4.axhline(y=0.01, color='g', linestyle='--', label='Tolerance (0.01)')
    ax4.set_xlabel('Angle (radians)')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Prediction Errors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([-3.5, 3.5])

    plt.tight_layout()
    plt.savefig('trig_model_results.png', dpi=150)
    print(f"\nGraph saved to: trig_model_results.png")
    plt.show()

def main():
    # Read from Arduino
    data_lines = read_arduino_data()

    if data_lines is None or len(data_lines) == 0:
        print("Failed to receive data from Arduino.")
        print("Make sure Arduino is connected and press reset button.")
        sys.exit(1)

    angles, sin_pred, cos_pred, tan_pred = parse_data(data_lines)
    print(f"Parsed {len(angles)} data points")

    if len(angles) > 0:
        create_graphs(angles, sin_pred, cos_pred, tan_pred)
    else:
        print("No valid data to plot")

if __name__ == "__main__":
    main()
