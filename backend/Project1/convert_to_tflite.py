"""
TensorFlow to TensorFlow Lite Converter for Arduino Deployment
Author: CST-440 Team
Date: January 9, 2026

Converts trained TensorFlow/Keras models to TensorFlow Lite format
optimized for Arduino Nano 33 BLE Sense Rev2 (nRF52840, 1MB flash, 256KB RAM).

This script is configured for the trigonometric model (sin, cos, tan) trained
in backend/Project1/trig_model.py with 4-input architecture [x, is_sin, is_cos, is_tan].

Usage:
    # Basic conversion (optimized, with test data)
    python convert_to_tflite.py
    
    # With quantization for smaller size
    python convert_to_tflite.py --quantize
    
    # Custom output location
    python convert_to_tflite.py --output arduino_models/
    
    # Advanced usage
    python convert_to_tflite.py --model path/to/model.keras --output out/
"""

# Think of these as toolboxes we need to use:
import tensorflow as tf  # The AI brain toolkit - helps us work with our smart model
import numpy as np  # Math helper - does calculations with lots of numbers at once
import argparse  # Command reader - understands what you type in the terminal
import os  # File organizer - helps us find and save files on the computer
import sys  # System controller - can stop the program if something goes wrong
from pathlib import Path  # Path helper - makes working with file locations easier


class TFLiteConverter:
    """Converts TensorFlow models to TensorFlow Lite for Arduino."""
    
    # These are like size limits for our Arduino board - think of it like:
    # FLASH = long-term storage (like a backpack that holds 1 MB)
    # RAM = quick memory (like hands that can hold 256 KB)
    ARDUINO_FLASH_MB = 1.0  # Our Arduino can store 1 megabyte of model
    ARDUINO_RAM_KB = 256  # Our Arduino has 256 kilobytes of working memory
    
    def __init__(self, model_path, verbose=True):
        """
        Initialize converter with model path.
        
        Args:
            model_path: Path to trained Keras model (.h5, .keras, or SavedModel dir)
            verbose: Enable verbose output
        """
        self.model_path = model_path
        self.verbose = verbose
        self.model = None
        self._load_model()
    
    def _log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _load_model(self):
        """Load the TensorFlow/Keras model."""
        # This is like opening a saved game - we're loading our trained AI model from a file
        self._log(f"\n{'='*70}")
        self._log("Loading TensorFlow Model")
        self._log(f"{'='*70}")
        
        try:
            # Sometimes files are saved in different ways, so we try multiple methods
            # Think of it like trying different keys to open a locked box
            import os
            
            # Strategy 1: Try the simple way - just load the model without extra stuff
            try:
                # compile=False means "don't worry about how to train it, just load it"
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                self._log(f"✓ Model loaded (without compilation) from: {self.model_path}")
            except Exception as e1:
                self._log(f"  First attempt failed, trying alternative method...")
                
                # Strategy 2: Load as SavedModel format
                try:
                    # Check if it's a .keras file
                    if self.model_path.endswith('.keras'):
                        # Try converting to h5 format first
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        temp_path = os.path.join(temp_dir, 'temp_model')
                        
                        # Load and immediately save in SavedModel format
                        import keras
                        model_temp = keras.saving.load_model(self.model_path, compile=False)
                        model_temp.save(temp_path, save_format='tf')
                        
                        # Load from SavedModel format
                        self.model = tf.saved_model.load(temp_path)
                        self._log(f"✓ Model loaded via SavedModel format from: {self.model_path}")
                        
                        # Clean up
                        import shutil
                        shutil.rmtree(temp_dir)
                    else:
                        raise e1
                except Exception as e2:
                    raise Exception(f"All loading strategies failed. Original error: {e1}")
            
            # Show information about our model - like reading a recipe card
            if self.verbose and hasattr(self.model, 'summary'):
                self._log("\nModel Architecture:")
                self.model.summary()  # This shows all the layers (parts) of our AI brain
                
                # Count how many numbers (parameters) our model has
                if hasattr(self.model, 'count_params'):
                    total_params = self.model.count_params()  # Count all the number dials in our model
                    # Calculate size: each number takes 4 bytes of space
                    size_mb = (total_params * 4) / (1024 * 1024)  # Convert bytes to megabytes
                    self._log(f"\nModel Statistics:")
                    self._log(f"  Total parameters: {total_params:,}")
                    self._log(f"  Estimated size: {size_mb:.3f} MB")
                
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print(f"\nSolutions:")
            print(f"  1. Re-train the model with your current TensorFlow/Keras version:")
            print(f"     python trig_model.py")
            print(f"  2. Or ask your teammate to export as SavedModel format:")
            print(f"     model.save('trig_model_all', save_format='tf')")
            sys.exit(1)
    
    def convert(self, optimize=True, quantize=False, representative_data=None):
        """
        Convert model to TensorFlow Lite format.
        
        Args:
            optimize: Apply default optimizations (reduce size, improve speed)
            quantize: Apply int8 quantization (requires representative_data)
            representative_data: NumPy array for quantization calibration
            
        Returns:
            TFLite model as bytes
        """
        # This is the MAIN MAGIC STEP! We're shrinking our big AI model to fit on tiny Arduino
        # Think of it like packing a suitcase - we're making it smaller but keeping what's important
        self._log(f"\n{'='*70}")
        self._log("Converting to TensorFlow Lite")
        self._log(f"{'='*70}")
        
        # Create our converter tool - this is what will shrink our model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Make the model smaller and faster - like compressing a photo
        if optimize:
            self._log("✓ Applying default optimizations...")
            # This tells the converter to remove unnecessary stuff and make things smaller
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Make the model EVEN SMALLER by using smaller numbers (quantization)
        # Instead of using big precise numbers, we use smaller approximate numbers
        # Like saying "about 5" instead of "5.0000000001"
        if quantize:
            if representative_data is None:
                print("✗ Error: Quantization requires representative data")
                sys.exit(1)
            
            self._log("✓ Applying int8 quantization...")
            
            # We need to give the converter some example data so it knows what numbers to expect
            # This is like showing someone sample problems so they know what to prepare for
            def representative_dataset():
                # Give it up to 100 examples from our data
                for i in range(min(100, len(representative_data))):
                    sample = representative_data[i:i+1]  # Get one example
                    yield [sample.astype(np.float32)]  # Give it to the converter
            
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        # NOW DO THE ACTUAL CONVERSION! This is where the magic happens!
        try:
            tflite_model = converter.convert()  # Transform our big model into a tiny one!
            self._log(f"✓ Conversion successful!")
            
            # Check how big our new tiny model is
            size_kb = len(tflite_model) / 1024  # Convert from bytes to kilobytes
            size_mb = size_kb / 1024  # Convert from kilobytes to megabytes
            self._log(f"\nTFLite Model Size:")
            self._log(f"  Size: {size_kb:.2f} KB ({size_mb:.3f} MB)")
            
            # Check if our model will fit on the Arduino - like checking if clothes fit in a suitcase
            if size_mb < self.ARDUINO_FLASH_MB:
                self._log(f"  ✓ Fits in Arduino flash ({self.ARDUINO_FLASH_MB} MB available)")
            else:
                # Uh oh! Too big! Like trying to fit an elephant in a car
                self._log(f"  ✗ May be too large for Arduino flash!")
                self._log(f"    Consider enabling quantization to reduce size")
            
            return tflite_model
            
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
            sys.exit(1)
    
    def save_tflite(self, tflite_model, output_path):
        """
        Save TFLite model to file.
        
        Args:
            tflite_model: TFLite model bytes
            output_path: Path to save the .tflite file
        """
        # Create the folder if it doesn't exist - like making a new drawer for storage
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Save our tiny model to a file - like saving a game
        # 'wb' means "write binary" - we're saving raw computer data
        with open(output_path, 'wb') as f:
            f.write(tflite_model)  # Write the model data to the file
        
        self._log(f"\n✓ TFLite model saved: {output_path}")
    
    def generate_c_header(self, tflite_model, output_path, array_name="model"):
        """
        Generate C header file for Arduino.
        
        Args:
            tflite_model: TFLite model bytes
            output_path: Path to save the .h file
            array_name: Name of the C array variable
        """
        # Arduino speaks a language called C, not Python!
        # So we need to translate our model into C language - like translating Spanish to English
        self._log(f"\n{'='*70}")
        self._log("Generating C Header for Arduino")
        self._log(f"{'='*70}")
        
        # Create folder if needed
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Convert each byte of our model to hexadecimal format (0x00, 0x01, etc.)
        # Think of this like writing numbers in a special computer code that Arduino understands
        hex_array = [f"0x{byte:02x}" for byte in tflite_model]
        
        # Create the text for our C header file
        # This is like writing a letter that Arduino can read
        header_content = f"""// Auto-generated TensorFlow Lite model for Arduino
// Generated: {self._get_timestamp()}
// Model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.2f} KB)
// Target: Arduino Nano 33 BLE Sense Rev2 (nRF52840)

#ifndef {array_name.upper()}_H
#define {array_name.upper()}_H

// Model data array - this is our AI brain stored as a list of numbers!
const unsigned char {array_name}[] = {{
"""
        
        # Write all the model numbers in neat rows - like organizing crayons in a box
        # We put 12 numbers on each line to keep it readable
        for i in range(0, len(hex_array), 12):
            row = ", ".join(hex_array[i:i+12])  # Join 12 numbers with commas
            # Add comma at end unless it's the last row (grammar rules!)
            comma = "," if i + 12 < len(hex_array) else ""
            header_content += f"  {row}{comma}\n"  # Add this row to our file
        
        header_content += f"""
}};

// Model size constant
const unsigned int {array_name}_len = {len(tflite_model)};

#endif // {array_name.upper()}_H
"""
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(header_content)
        
        self._log(f"✓ C header saved: {output_path}")
        self._log(f"  Array name: {array_name}")
        self._log(f"  Array length: {array_name}_len")
        self._log(f"\nUsage in Arduino sketch:")
        self._log(f'  #include "{os.path.basename(output_path)}"')
        self._log(f"  const tflite::Model* model = tflite::GetModel({array_name});")
    
    def test_model(self, tflite_model, test_data_x, test_data_y=None, num_samples=10):
        """
        Test TFLite model inference and compare with original model.
        
        Args:
            tflite_model: TFLite model bytes
            test_data_x: Test input data (numpy array)
            test_data_y: Test output data (numpy array, optional)
            num_samples: Number of samples to test
        """
        # Time to test if our tiny model works! Like testing a toy after building it
        self._log(f"\n{'='*70}")
        self._log("Testing TFLite Model")
        self._log(f"{'='*70}")
        
        # Create an interpreter - this is like a translator that runs our tiny model
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()  # Set up memory space for the model to work
        
        # Find out what kind of data our model expects and returns
        # Like reading the instructions: "Put 4 numbers in, get 1 number out"
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        self._log(f"\nModel I/O Details:")
        self._log(f"  Input shape: {input_details[0]['shape']}")  # How many numbers go IN
        self._log(f"  Input type: {input_details[0]['dtype'].__name__}")  # What type of numbers
        self._log(f"  Output shape: {output_details[0]['shape']}")  # How many numbers come OUT
        self._log(f"  Output type: {output_details[0]['dtype'].__name__}")  # What type of numbers
        
        # Now let's test some examples and see if our tiny model gives the same answers!
        # Like comparing a copy to the original to see if it's accurate
        num_test = min(num_samples, len(test_data_x))
        self._log(f"\nRunning inference on {num_test} samples:")
        self._log("-" * 70)
        
        # Lists to store predictions from both models
        tflite_predictions = []  # Answers from our tiny model
        original_predictions = []  # Answers from the original big model
        
        # Test each example one by one
        for i in range(num_test):
            # Get one test example and make sure it's in the right format
            input_data = test_data_x[i:i+1].astype(input_details[0]['dtype'])
            
            # Ask the TINY model for an answer
            interpreter.set_tensor(input_details[0]['index'], input_data)  # Give it the question
            interpreter.invoke()  # Tell it to think and calculate
            tflite_output = interpreter.get_tensor(output_details[0]['index'])  # Get the answer
            
            # Ask the ORIGINAL big model for an answer (to compare)
            original_output = self.model.predict(test_data_x[i:i+1], verbose=0)
            
            tflite_predictions.append(tflite_output[0][0])
            original_predictions.append(original_output[0][0])
            
            # Display results with function type
            func_type = "sin" if test_data_x[i][1] == 1 else ("cos" if test_data_x[i][2] == 1 else "tan")
            x_val = test_data_x[i][0] * (2 * 3.14) - 3.14  # Denormalize x
            
            if test_data_y is not None:
                expected = test_data_y[i] if len(test_data_y.shape) == 1 else test_data_y[i][0]
                self._log(f"Sample {i+1} [{func_type}]: x={x_val:.3f} | "
                         f"Expected={expected:.4f} | "
                         f"TFLite={tflite_output[0][0]:.4f} | "
                         f"Original={original_output[0][0]:.4f}")
            else:
                self._log(f"Sample {i+1} [{func_type}]: x={x_val:.3f} | "
                         f"TFLite={tflite_output[0][0]:.4f} | "
                         f"Original={original_output[0][0]:.4f}")
        
        # Compare how different the tiny model's answers are from the original
        # Like checking how close a photocopy is to the original photo
        tflite_predictions = np.array(tflite_predictions)
        original_predictions = np.array(original_predictions)
        diff = np.abs(tflite_predictions - original_predictions)  # Calculate differences
        
        self._log("-" * 70)
        self._log(f"\nTFLite vs Original Model:")
        self._log(f"  Mean absolute difference: {np.mean(diff):.6f}")  # Average difference
        self._log(f"  Max absolute difference: {np.max(diff):.6f}")  # Biggest difference
        
        if np.mean(diff) < 0.001:
            self._log("  ✓ TFLite model is highly accurate!")
        elif np.mean(diff) < 0.01:
            self._log("  ✓ TFLite model has good accuracy")
        else:
            self._log("  ⚠ TFLite model may have significant accuracy loss")
    
    def _get_timestamp(self):
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_trig_test_data(num_samples=100):
    """
    Generate test data matching the trig model's input format.
    Input: [x_normalized, is_sin, is_cos, is_tan]
    Output: trig function value
    """
    # Create test data for our trig (sin, cos, tan) functions
    # This is like making practice problems to test if our model learned correctly
    x_base = np.linspace(-3.14, 3.14, num_samples)  # Create evenly spaced numbers from -π to π
    X_samples = []  # List to store input questions
    y_samples = []  # List to store correct answers
    
    # Create test examples for all three trig functions: sin, cos, and tan
    for x_val in x_base:
        # Squish the number to be between 0 and 1 (normalization)
        x_norm = (x_val + 3.14) / (2 * 3.14)  # Normalize to [0, 1]
        
        # Test sin: input is [x, 1, 0, 0] where the 1 means "calculate sine"
        X_samples.append([x_norm, 1, 0, 0])
        y_samples.append(np.sin(x_val))  # The correct answer
        
        # Test cos: input is [x, 0, 1, 0] where the 1 means "calculate cosine"
        X_samples.append([x_norm, 0, 1, 0])
        y_samples.append(np.cos(x_val))  # The correct answer
        
        # Test tan: input is [x, 0, 0, 1] where the 1 means "calculate tangent"
        # Only test tan when the answer isn't crazy big (avoid infinity)
        tan_val = np.tan(x_val)
        if np.abs(tan_val) < 3:  # Only if the answer is reasonable
            X_samples.append([x_norm, 0, 0, 1])
            y_samples.append(tan_val)  # The correct answer
    
    return np.array(X_samples, dtype=np.float32), np.array(y_samples, dtype=np.float32)


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Convert TensorFlow models to TensorFlow Lite for Arduino',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (uses default backend/Project1/trig_model_all.keras)
  python convert_to_tflite.py
  
  # With quantization (smaller model, faster inference)
  python convert_to_tflite.py --quantize
  
  # Custom model path
  python convert_to_tflite.py --model backend/Project1/trig_model_all.keras
  
  # Custom output directory
  python convert_to_tflite.py --output arduino_models/
  
  # Skip C header generation
  python convert_to_tflite.py --no-header
        """
    )
    
    # Default to the trained trig model
    default_model = 'trig_model_all.keras'
    
    # Required arguments (with default)
    parser.add_argument('--model', '-m', default=default_model,
                       help=f'Path to trained TensorFlow/Keras model (default: {default_model})')
    
    # Optional arguments
    parser.add_argument('--output', '-o', default='.',
                       help='Output directory for converted files (default: current directory)')
    parser.add_argument('--name', '-n', default=None,
                       help='Base name for output files (default: same as model)')
    parser.add_argument('--array-name', default='trig_model',
                       help='Name for C array variable (default: "trig_model")')
    
    # Conversion options
    parser.add_argument('--no-optimize', action='store_true',
                       help='Disable default optimizations')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply int8 quantization (reduces size, auto-generates test data)')
    parser.add_argument('--data', '-d', default=None,
                       help='Representative data for quantization (.npy file, auto-generated if not provided)')
    
    # Output options
    parser.add_argument('--no-header', action='store_true',
                       help='Skip C header file generation')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip model testing')
    parser.add_argument('--test-data', nargs=2, metavar=('X', 'Y'),
                       help='Test data files: input.npy output.npy')
    
    # Other options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print(f"Make sure the model has been trained first:")
        print(f"  python trig_model.py")
        sys.exit(1)
    
    # Determine output file names
    model_stem = Path(args.model).stem
    output_name = args.name or model_stem
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    tflite_path = os.path.join(output_dir, f"{output_name}.tflite")
    header_path = os.path.join(output_dir, f"{output_name}.h")
    
    # Initialize converter
    converter = TFLiteConverter(args.model, verbose=not args.quiet)
    
    # Generate or load test data for the trig model
    test_x, test_y = None, None
    if not args.no_test or args.quantize:
        if args.test_data:
            try:
                test_x = np.load(args.test_data[0])
                test_y = np.load(args.test_data[1])
                converter._log(f"✓ Loaded test data from files")
            except Exception as e:
                print(f"Warning: Could not load test data: {e}")
                test_x, test_y = generate_trig_test_data()
                converter._log(f"✓ Generated test data for trig model")
        else:
            test_x, test_y = generate_trig_test_data()
            converter._log(f"✓ Generated test data for trig model (300+ samples)")
    
    # Load or generate representative data for quantization
    representative_data = None
    if args.quantize:
        if args.data:
            try:
                representative_data = np.load(args.data)
                converter._log(f"✓ Loaded representative data: {args.data}")
                converter._log(f"  Shape: {representative_data.shape}")
            except Exception as e:
                print(f"Warning: Could not load representative data: {e}")
                representative_data = test_x
                converter._log(f"✓ Using generated test data for quantization")
        else:
            representative_data = test_x
            converter._log(f"✓ Using generated test data for quantization")
    
    # Convert model
    tflite_model = converter.convert(
        optimize=not args.no_optimize,
        quantize=args.quantize,
        representative_data=representative_data
    )
    
    # Save TFLite model
    converter.save_tflite(tflite_model, tflite_path)
    
    # Generate C header
    if not args.no_header:
        converter.generate_c_header(tflite_model, header_path, args.array_name)
    
    # Test model
    if not args.no_test and test_x is not None:
        converter.test_model(tflite_model, test_x, test_y, num_samples=15)
    
    # Summary
    print(f"\n{'='*70}")
    print("Conversion Complete!")
    print(f"{'='*70}")
    print(f"Output files:")
    print(f"  TFLite model: {tflite_path}")
    if not args.no_header:
        print(f"  C header: {header_path}")
    print(f"\nNext steps:")
    print(f"  1. Copy {os.path.basename(header_path)} to your Arduino sketch folder")
    print(f"  2. Include in your sketch: #include \"{os.path.basename(header_path)}\"")
    print(f"  3. Load model: tflite::GetModel({args.array_name})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
