import argparse
import os
import logging
import sys

# Ensure project root is in sys.path for consistent imports like `from src.module`
# This makes the script runnable as `python src/cli.py` from the project root.
current_script_dir = os.path.dirname(os.path.abspath(__file__)) # Should be /app/src
project_root_dir = os.path.dirname(current_script_dir) # Should be /app
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
    logger_init = logging.getLogger('audio_upscaler_cli_init') # Temp logger for this msg
    logger_init.debug(f"Added project root {project_root_dir} to sys.path for imports.")

# Basic config for initial logging, main() will refine it based on args.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('audio_upscaler_cli')

try:
    from src.inference import run_inference
    logger.debug("Successfully imported run_inference from src.inference")
except ImportError as e_import_real:
    logger.error(f"Critical Error: Could not import the real run_inference function. Details: {e_import_real}", exc_info=True)
    logger.error("This may be due to missing dependencies (like PyTorch) or an incorrect project structure.")
    logger.warning("Falling back to a DUMMY run_inference function. The CLI will parse args but not perform actual work.")
    def run_inference(*args, **kwargs): # Define dummy function in this scope
        # Need to access the global logger defined above, or re-initialize one for the dummy.
        # Using the global logger instance:
        logger.info(f"[Dummy run_inference] Called with: args={args}, kwargs={kwargs}")
        logger.warning("[Dummy run_inference] This is a placeholder because the real function could not be imported.")
        logger.warning("[Dummy run_inference] This usually means torch or its dependencies are missing or there's a path issue.")
        return False # Simulate a failure or unsuccessful operation

# Re-configure logger based on args in main, if needed.
# For now, the basicConfig above will handle initial logs.

def main():
    parser = argparse.ArgumentParser(description="Audio Upscaler CLI using a diffusion model.")

    parser.add_argument("input_file", type=str, help="Path to the low-resolution input audio file.")
    parser.add_argument("output_file", type=str, help="Path to save the upscaled high-resolution audio file.")

    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the model and inference configuration YAML file.")

    parser.add_argument("--target_sr", type=int, default=None,
                        help="Target sample rate for the output. If None, uses model's default from config.")
    parser.add_argument("--gain_db", type=float, default=0.0,
                        help="Gain in dB to apply to the output audio. Default: 0.0.")

    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.). Defaults to cuda if available, else cpu.")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose INFO logging for this CLI.")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG level logging for this CLI and imported modules.")

    args = parser.parse_args()

    # Configure logging level based on arguments
    if args.debug:
        # Set level for all loggers if desired, or target specific ones
        logging.getLogger().setLevel(logging.DEBUG) # Root logger
        logger.setLevel(logging.DEBUG) # This script's logger
        # For imported modules, their own loggers would need configuring if they don't inherit.
        # This can be done by iterating through logging.Logger.manager.loggerDict or being specific.
        logging.getLogger('src.inference').setLevel(logging.DEBUG) # Example for inference module
        logger.debug("Debug logging enabled.")
    elif args.verbose:
        logger.setLevel(logging.INFO)
        logging.getLogger('src.inference').setLevel(logging.INFO)
        logger.info("Verbose logging enabled.")
    else:
        # Default if not verbose/debug (already set by basicConfig, but can be more specific)
        logger.setLevel(logging.WARNING) # Less verbose by default for CLI tool
        logging.getLogger('src.inference').setLevel(logging.INFO) # Inference can be more chatty


    logger.info("Audio Upscaler CLI started.")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.debug(f"Model checkpoint: {args.checkpoint_path}") # Debug for potentially sensitive paths
    logger.debug(f"Config file: {args.config_path}")
    if args.target_sr: logger.info(f"Target sample rate: {args.target_sr}")
    if args.gain_db != 0.0: logger.info(f"Output gain: {args.gain_db} dB")
    if args.device: logger.info(f"Requested device: {args.device}")

    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}"); return 1
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Model checkpoint not found: {args.checkpoint_path}"); return 1
    if not os.path.exists(args.config_path):
        logger.error(f"Configuration file not found: {args.config_path}"); return 1

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        try: os.makedirs(output_dir); logger.info(f"Created output directory: {output_dir}")
        except OSError as e: logger.error(f"Could not create output directory {output_dir}: {e}"); return 1

    try:
        # The global run_inference (either real or dummy) is called here.
        success = run_inference(
            input_path=args.input_file, output_path=args.output_file,
            checkpoint_path=args.checkpoint_path, config_path=args.config_path,
            target_sr=args.target_sr, gain_db=args.gain_db,
            # device_str=args.device # run_inference needs to be updated to accept this
        )
        if success: logger.info("Upscaling process completed successfully."); return 0
        else: logger.error("Upscaling process failed or did not complete."); return 1

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        if isinstance(e, ModuleNotFoundError) and 'torch' in str(e).lower():
            logger.critical("This error is likely due to PyTorch not being installed correctly.")
        return 1

if __name__ == '__main__':
    return_code = main()
    sys.exit(return_code)
