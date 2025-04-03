"""ModelPort CLI main entry point."""

import sys
import argparse

def main():
    """Main entry point for the ModelPort CLI."""
    parser = argparse.ArgumentParser(description="ModelPort - ML Model Deployment and Compilation")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a model to ONNX format")
    export_parser.add_argument("model_path", help="Path to the PyTorch model")
    export_parser.add_argument("--output", "-o", help="Output path for the ONNX model")
    export_parser.add_argument("--input-shape", help="Input shape for the model (comma-separated)")
    
    # Compile command
    compile_parser = subparsers.add_parser("compile", help="Compile a model for a target architecture")
    compile_parser.add_argument("model_path", help="Path to the ONNX model")
    compile_parser.add_argument("--target-arch", help="Target architecture (x86_64, aarch64)", default="x86_64")
    compile_parser.add_argument("--target-device", help="Target device (cpu, cuda)", default="cpu")
    compile_parser.add_argument("--opt-level", type=int, help="Optimization level (0-3)", default=3)
    compile_parser.add_argument("--output-dir", help="Output directory for the compiled model")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run inference on a compiled model")
    run_parser.add_argument("model_dir", help="Directory containing the compiled model")
    run_parser.add_argument("--input", help="Input data file (NumPy .npy)")
    run_parser.add_argument("--output", help="Output file for the inference results")
    run_parser.add_argument("--device", help="Device to run inference on (cpu, cuda)", default="cpu")
    
    # Diagnostics command
    diag_parser = subparsers.add_parser("diagnostics", help="Print diagnostic information")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == "export":
        from modelport.export import to_onnx
        import torch
        
        # Parse input shape
        if args.input_shape:
            input_shape = tuple(map(int, args.input_shape.split(",")))
        else:
            input_shape = (1, 3, 224, 224)  # Default for image models
        
        # Load model
        try:
            model = torch.load(args.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return 1
        
        # Set output path
        if args.output is None:
            output_path = args.model_path.replace(".pt", ".onnx").replace(".pth", ".onnx")
            if output_path == args.model_path:
                output_path = args.model_path + ".onnx"
        else:
            output_path = args.output
        
        # Export model
        try:
            to_onnx(model, input_shape, output_path)
            print(f"Model exported to {output_path}")
        except Exception as e:
            print(f"Error exporting model: {e}")
            return 1
    
    elif args.command == "compile":
        from modelport.compile import compile_model
        
        try:
            output_dir = compile_model(
                args.model_path,
                target_arch=args.target_arch,
                target_device=args.target_device,
                opt_level=args.opt_level,
                output_dir=args.output_dir
            )
            print(f"Model compiled to {output_dir}")
        except Exception as e:
            print(f"Error compiling model: {e}")
            return 1
    
    elif args.command == "run":
        from modelport.inference import run
        import numpy as np
        
        if args.input is None:
            print("Error: Input data file is required")
            return 1
        
        try:
            # Load input data
            input_data = np.load(args.input)
            
            # Run inference
            outputs = run(args.model_dir, {"input": input_data}, device=args.device)
            
            # Save output
            if args.output:
                np.save(args.output, outputs[0])
                print(f"Output saved to {args.output}")
            else:
                print(f"Output shape: {outputs[0].shape}")
                print(f"Output preview: {outputs[0].flatten()[:5]}...")
        except Exception as e:
            print(f"Error running inference: {e}")
            return 1
    
    elif args.command == "diagnostics":
        from modelport.utils.diagnostics import print_diagnostics
        print_diagnostics()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 