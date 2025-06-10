import argparse
import logging
import os
import sys

# Setup basic logging VERY early, before other imports if possible,
# to catch issues during import or initial setup.
# The format can be refined later by load_config if a config file is successfully loaded.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Ensure logs go to stdout by default

# Get a logger for this main script
logger = logging.getLogger("OCRMain")

# Adjust Python path if modules are not found.
# This assumes main.py is in /app and other modules are also in /app or discoverable.
# For robust execution, ensure PYTHONPATH is set correctly in the environment.
# sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Usually not needed if run from repo root

try:
    from ocr_workflow_orchestrator import OCRWorkflowOrchestrator
    from custom_exceptions import OCRFileNotFoundError, OCRConfigurationError, OCRPipelineError
    # config_loader.load_config is called by OCRWorkflowOrchestrator, no direct call needed here.
except ImportError as e_import:
    logger.critical(f"Critical Error: Failed to import necessary OCR modules. Ensure all project files are present and PYTHONPATH is set correctly. Details: {e_import}", exc_info=True)
    print(f"Critical Error: Failed to import necessary modules. Check logs for details. Exiting.", file=sys.stderr)
    sys.exit(1)
except Exception as e_startup: # Catch any other unexpected error during startup/imports
    logger.critical(f"Critical Error: An unexpected error occurred during application startup: {e_startup}", exc_info=True)
    print(f"Critical Error: An unexpected error occurred during startup. Check logs for details. Exiting.", file=sys.stderr)
    sys.exit(1)


def main_cli():
    """
    Command-Line Interface for the OCR Application.
    """
    parser = argparse.ArgumentParser(
        description="OCR Application: Process an image and extract text.",
        epilog="Example usage:\n"
               "  python main.py path/to/your/image.png\n"
               "  python main.py --image path/to/your/image.jpg\n"
               "  python main.py --image path/to/your/image.tiff --config path/to/custom_config.yaml\n\n"
               "A default 'config_dev.yaml' will be used (and created if missing, with dummy models) if --config is not specified.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Consolidate image path argument
    parser.add_argument(
        "image_path",
        nargs='?', # Makes it optional if --image is used (though we'll enforce one is given)
        help="Path to the input image file."
    )
    parser.add_argument(
        "--image",
        dest='image_path_option',
        type=str,
        help="Path to the input image file (alternative to positional argument)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration YAML file. If not provided, defaults to 'config_dev.yaml'."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose DEBUG logging for the console. Overrides config file's console handler level."
    )

    args = parser.parse_args()

    # Resolve image_path (prefer --image if both given, though argparse usually handles this)
    final_image_path = args.image_path_option if args.image_path_option else args.image_path

    if not final_image_path:
        parser.error("Image path is required. Please provide it as a positional argument or using --image.")

    # If --verbose, set console logging to DEBUG.
    # This basicConfig call might reconfigure if load_config hasn't set up more complex handlers.
    # If load_config *has* set handlers, this might not have the desired effect on existing handlers.
    # A more robust way would be to get the root logger and adjust its handlers' levels.
    # For now, this is a simple approach.
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG) # Resets basic config to DEBUG
        logger.setLevel(logging.DEBUG) # Ensure our main logger is also DEBUG
        logger.info("Verbose DEBUG logging enabled via --verbose flag.")
        # Note: This might be too late if other modules already got their loggers
        # A better way: iterate over loggers/handlers after config load if verbose is set.
        # For this task, we'll assume load_config in orchestrator primarily sets levels.
        # This verbose flag in main.py is for main.py's own initial messages or if config fails early.


    logger.info(f"OCR CLI started. Image: '{final_image_path}', Config: '{args.config if args.config else 'Default (config_dev.yaml)'}'")

    if not os.path.exists(final_image_path):
        logger.error(f"Input image file not found: '{final_image_path}'")
        print(f"Error: Input image file not found at '{final_image_path}'. Please check the path.", file=sys.stderr)
        sys.exit(1)

    try:
        logger.info("Initializing OCR Workflow Orchestrator...")
        # Orchestrator's __init__ will call load_config. If args.config is None, orchestrator uses its default.
        orchestrator = OCRWorkflowOrchestrator(config_path=args.config)
        logger.info("Orchestrator initialized successfully.")

        logger.info(f"Processing document: '{final_image_path}'...")
        processing_result = orchestrator.process_document(final_image_path)

        logger.info("Retrieving formatted results...")
        formatted_results = orchestrator.get_results(processing_result)

        print("\n--- OCR Results ---")
        print(formatted_results)
        print("--- End of Results ---\n")
        logger.info("Processing complete. Results printed.")

    except OCRFileNotFoundError as e_fnf:
        logger.error(f"File Not Found Error: {e_fnf}", exc_info=False) # exc_info=False as error is usually clear
        print(f"\nError: A required file was not found.\nDetails: {e_fnf}\nPlease check file paths and configurations.", file=sys.stderr)
        sys.exit(1)
    except OCRConfigurationError as e_config:
        logger.error(f"Configuration Error: {e_config}", exc_info=True) # Show traceback for config issues
        print(f"\nError: There was a problem with the application configuration.\nDetails: {e_config}\nPlease check your configuration file or try with defaults.", file=sys.stderr)
        sys.exit(1)
    except OCRPipelineError as e_pipe:
        logger.error(f"OCR Pipeline Error: {e_pipe}", exc_info=True) # Show traceback for pipeline issues
        print(f"\nError: An error occurred during the OCR processing pipeline.\nDetails: {e_pipe}", file=sys.stderr)
        sys.exit(1)
    except Exception as e_unexpected: # Catch-all for any other unhandled exceptions
        logger.critical(f"An unexpected critical error occurred: {e_unexpected}", exc_info=True)
        print(f"\nAn unexpected critical error occurred. Please check the logs for details. Error: {e_unexpected}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main_cli()
