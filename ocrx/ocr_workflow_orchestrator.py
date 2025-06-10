"""
OCR Workflow Orchestrator for the OCR-X Project.

This module defines the main orchestrator class responsible for managing the
OCR pipeline, including image loading, preprocessing, engine management,
recognition, and postprocessing.
"""

import logging
import importlib
import os
import time # Added for processing time calculation
from typing import Any, Dict, List, Optional, Union

# Assuming these modules are in the same directory or accessible via PYTHONPATH
from ocrx.core.ocr_engine_interface import OCREngine
from ocrx.core.config_loader import load_config, DEFAULT_LOGGING_CONFIG, get_module_config
from ocrx.core.data_objects import DocumentContext, PageContext, TextRegion, RecognitionResult
from ocrx.core.module_base import OCRXModuleBase
from ocrx.core.exceptions import OCRXProcessingError, OCRXConfigurationError, OCRXInputError

from ocrx.modules.image_loader import ImageLoader # type: ignore
from ocrx.modules.preprocessor import AdaptivePreprocessor # type: ignore
from ocrx.modules.layout_analyzer import LayoutAnalyzer # type: ignore
from ocrx.modules.recognition_engine_manager import RecognitionCoreManager # type: ignore
from ocrx.modules.postprocessor import AdvancedPostprocessor # type: ignore
from ocrx.modules.output_generator import OutputGenerator # type: ignore

# For image loading - install Pillow: pip install Pillow
from PIL import Image, UnidentifiedImageError
import numpy as np


class OCREngineManager:
    """
    Manages the discovery, loading, and access of available OCR engines.
    """
    def __init__(self, app_config: Dict[str, Any], parent_logger: logging.Logger):
        """
        Initializes the OCREngineManager.

        :param app_config: The global application configuration dictionary.
        :param parent_logger: The parent logger instance (typically from OCRWorkflowOrchestrator).
        """
        self.engines: Dict[str, OCREngine] = {}
        self.engine_configs: Dict[str, Any] = app_config.get('ocr_engines', {})
        self.logger = parent_logger.getChild(self.__class__.__name__)
        self.logger.info(f"OCREngineManager initialized with {len(self.engine_configs)} engine configurations.")

    def discover_and_load_engines(self) -> None:
        """
        Discovers and loads OCR engines based on the application configuration.
        """
        self.logger.info("Starting OCR engine discovery and loading process...")
        for engine_name, engine_config_data in self.engine_configs.items():
            if not engine_config_data.get('enabled', False):
                self.logger.info(f"Engine '{engine_name}' is disabled in configuration. Skipping.")
                continue

            module_path = engine_config_data.get('module')
            class_name = engine_config_data.get('class')

            if not module_path or not class_name:
                self.logger.error(f"Engine '{engine_name}' configuration is missing 'module' or 'class' path. Skipping.")
                continue

            try:
                self.logger.debug(f"Attempting to load engine '{engine_name}' from module '{module_path}' class '{class_name}'.")
                engine_module = importlib.import_module(module_path)
                engine_class = getattr(engine_module, class_name)

                # Pass only the specific engine's config section, not the whole ocr_engines dict
                engine_instance_config = engine_config_data.get('config', {}) # Assuming engine-specific params are under 'config'
                if not engine_instance_config and engine_config_data: # Fallback for flat config structure
                    # Filter out common keys like 'enabled', 'module', 'class' to pass only specific params
                    engine_instance_config = {k: v for k, v in engine_config_data.items() if k not in ['enabled', 'module', 'class']}


                child_logger = self.logger.getChild(engine_name) # Create a child logger for the engine instance
                engine_instance: OCREngine = engine_class(engine_config=engine_instance_config, logger=child_logger)

                engine_instance.initialize() # Can raise exceptions

                if engine_instance.is_available():
                    self.engines[engine_name] = engine_instance
                    self.logger.info(f"Successfully loaded and initialized engine: '{engine_name}' ({engine_instance.get_engine_name()}).")
                else:
                    self.logger.warning(f"Engine '{engine_name}' loaded but reported not available after initialization.")

            except ImportError as e:
                self.logger.error(f"Failed to import module '{module_path}' for engine '{engine_name}': {e}", exc_info=True)
            except AttributeError as e:
                self.logger.error(f"Failed to get class '{class_name}' from module '{module_path}' for engine '{engine_name}': {e}", exc_info=True)
            except TypeError as e: # Catches errors in engine constructor call
                self.logger.error(f"TypeError instantiating engine '{engine_name}'. Check constructor arguments: {e}", exc_info=True)
            except Exception as e: # Catch exceptions from engine's initialize() or other instantiation issues
                self.logger.error(f"Failed to initialize engine '{engine_name}': {e}", exc_info=True)

        self.logger.info(f"Engine discovery complete. {len(self.engines)} engines are available.")

    def get_engine(self, engine_name: str) -> Optional[OCREngine]:
        """
        Returns the instantiated engine if it exists and is available.

        :param engine_name: The name of the engine to retrieve.
        :return: An instance of the OCREngine, or None if not found or unavailable.
        """
        engine = self.engines.get(engine_name)
        if engine and engine.is_available():
            return engine
        elif engine: # Exists but not available
            self.logger.warning(f"Engine '{engine_name}' was requested but is not currently available.")
        else: # Does not exist
            self.logger.warning(f"Engine '{engine_name}' was requested but not found (or not loaded).")
        return None

    def get_available_engines(self) -> List[str]:
        """
        Returns a list of names of successfully loaded and available engines.

        :return: A list of engine names.
        """
        return [name for name, engine in self.engines.items() if engine.is_available()]


class OCRWorkflowOrchestrator:
    """
    Orchestrates the entire OCR workflow, from loading images to returning processed text.
    """
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initializes the OCRWorkflowOrchestrator.

        :param config_path: Path to the main application configuration file.
        """
        # Load configuration first, which also sets up logging
        self.config: Dict[str, Any] = load_config(config_path)

        # Now that logging is configured by load_config, get the logger for this class
        self.logger = logging.getLogger(__name__) # Or a specific name like 'OCRWorkflowOrchestrator'
        self.logger.info("OCRWorkflowOrchestrator initializing...")

        # Initialize Engine Manager
        self.engine_manager = OCREngineManager(app_config=self.config, parent_logger=self.logger)
        self.engine_manager.discover_and_load_engines()

        # Initialize core processing modules
        # Their configurations are expected to be under a "modules" key in the main config file.
        # e.g., config = {"modules": {"image_loader": {...}, "preprocessor": {...}}}

        image_loader_config = get_module_config(self.config, "image_loader",
                                                default_if_missing={"module_id": "image_loader_default"})
        self.image_loader = ImageLoader(module_id="image_loader", config=image_loader_config)

        preprocessor_config = get_module_config(self.config, "preprocessor",
                                                default_if_missing={"module_id": "preprocessor_default"})
        self.preprocessor = AdaptivePreprocessor(module_id="preprocessor", config=preprocessor_config)

        layout_analyzer_config = get_module_config(self.config, "layout_analyzer",
                                                 default_if_missing={"module_id": "layout_analyzer_default"})
        self.layout_analyzer = LayoutAnalyzer(module_id="layout_analyzer", config=layout_analyzer_config)

        recognition_manager_config = get_module_config(self.config, "recognition_manager",
                                                       default_if_missing={"module_id": "recognition_manager_default"})
        # Note: OCREngineManager is separate for now. RecognitionCoreManager will use it or similar.
        # For MVP, RecognitionCoreManager might directly use OCREngineManager's engines or be merged.
        # The task implies RecognitionCoreManager is new and will handle the .process() call for regions.
        # It needs access to the engines loaded by OCREngineManager or load them itself.
        # For now, let's assume it gets its own config and might internally use OCREngineManager if needed,
        # or directly load engines as per its config.
        # The current OCREngineManager is initialized above.
        # Let's make RecognitionCoreManager use the existing self.engine_manager for MVP.
        # So, its own "engines_config" inside recognition_manager_config might be less relevant if it just uses self.engine_manager.
        # However, the task for RecognitionCoreManager says it iterates self.config.get("engines_config", {}).
        # This suggests RecognitionCoreManager *is* the replacement or higher level orchestrator for engines.
        # Let's stick to the task: RecognitionCoreManager loads its own set of engines via its config.
        # This means OCREngineManager above might become redundant or used by RecognitionCoreManager.
        # For now, let's instantiate RecognitionCoreManager as per its spec.
        self.recognition_manager = RecognitionCoreManager(module_id="recognition_manager", config=recognition_manager_config)

        postprocessor_config = get_module_config(self.config, "postprocessor",
                                                 default_if_missing={"module_id": "postprocessor_default"})
        self.postprocessor = AdvancedPostprocessor(module_id="postprocessor", config=postprocessor_config)

        output_generator_config = get_module_config(self.config, "output_generator",
                                                    default_if_missing={"module_id": "output_generator_default"})
        self.output_generator = OutputGenerator(module_id="output_generator", config=output_generator_config)


        self.logger.info(f"ImageLoader initialized with config: {image_loader_config}")
        self.logger.info(f"AdaptivePreprocessor initialized with config: {preprocessor_config}")
        self.logger.info(f"LayoutAnalyzer initialized with config: {layout_analyzer_config}")
        self.logger.info(f"RecognitionCoreManager initialized with config: {recognition_manager_config}")
        self.logger.info(f"AdvancedPostprocessor initialized with config: {postprocessor_config}")
        self.logger.info(f"OutputGenerator initialized with config: {output_generator_config}")


        available_engines = self.engine_manager.get_available_engines() # This refers to the old OCREngineManager
        if available_engines:
            self.logger.info(f"Available OCR engines: {', '.join(available_engines)}")
        else:
            self.logger.warning("No OCR engines available after discovery and loading.")

        self.logger.info("OCRWorkflowOrchestrator initialized.")

    def process_document(self, source: Union[str, bytes], runtime_config_override: Optional[Dict] = None) -> DocumentContext:
        """
        Orchestrates the full OCR pipeline for a document.

        Args:
            source: The document source, can be a file path (str) or image/PDF bytes.
            runtime_config_override: Optional dictionary to override parts of the global config for this run.

        Returns:
            A DocumentContext object containing all processing results and status.
        """
        # Determine document_id
        if isinstance(source, str):
            doc_id = os.path.basename(source)
        else: # bytes
            doc_id = f"bytes_input_{hash(source) & 0xffffff}" # Simple hash for bytes input

        # Merge global config with runtime overrides (deep merge might be needed for nested dicts)
        # For now, simple top-level override for 'app_settings' and 'modules'
        current_global_config = self.config.copy()
        if runtime_config_override:
            current_global_config.update(runtime_config_override)

        doc_context = DocumentContext(
            document_id=doc_id,
            source_path_or_id=source if isinstance(source, str) else doc_id, # Use doc_id if bytes
            global_config=current_global_config
        )
        doc_context.overall_status = "processing_started"
        self.logger.info(f"Starting document processing for: '{doc_id}'.")

        request_params = current_global_config.get("request_params", {})
        requested_engine_name = request_params.get("ocr_engine")
        language_hint = request_params.get("language_hint")


        try:
            # 1. Image Loading
            if not self.image_loader:
                raise OCRXConfigurationError("ImageLoader module is not initialized.")

            start_time = time.time()
            doc_context = self.image_loader.process(document_context=doc_context, source=source)
            # ImageLoader populates doc_context.pages and handles its own errors within doc_context
            if doc_context.document_errors or not doc_context.pages:
                self.logger.error(f"Image loading failed for {doc_id}. Errors: {doc_context.document_errors}")
                doc_context.overall_status = "image_loading_failed"
                return doc_context # Early exit if loading fails critically

            # Assuming image_loader adds processing time, or do it here.
            # For now, let's assume modules manage their own time in page_ctx.processing_times

            # 2. Preprocessing (per page)
            if not self.preprocessor:
                raise OCRXConfigurationError("AdaptivePreprocessor module is not initialized.")

            if self.preprocessor.is_enabled():
                for i, page_ctx in enumerate(doc_context.pages):
                    if page_ctx.original_image is None:
                        self.logger.warning(f"Skipping preprocessing for page {i} as original_image is missing.")
                        continue
                    try:
                        page_start_time = time.time()
                        # Pass module-specific config, allow potential runtime overrides for preprocessor
                        module_specific_override = doc_context.global_config.get("modules", {}).get(self.preprocessor.module_id, {})

                        page_ctx.preprocessed_image = self.preprocessor.process(
                            page_ctx.original_image,
                            config_override=module_specific_override
                        )
                        page_ctx.processing_times["preprocessing"] = time.time() - page_start_time
                        self.logger.info(f"Preprocessing completed for page {i} of {doc_id}.")
                    except Exception as e:
                        err_msg = f"Error during preprocessing page {i} of {doc_id}: {e}"
                        self.logger.error(err_msg, exc_info=True)
                        page_ctx.errors.append(err_msg)
                        # Decide if this page failure should halt all or mark as partial
            else:
                self.logger.info("Preprocessing module is disabled. Skipping.")
                # Copy original to preprocessed if preprocessing is skipped but later stages expect preprocessed_image
                for page_ctx in doc_context.pages:
                    if page_ctx.original_image is not None and page_ctx.preprocessed_image is None:
                        page_ctx.preprocessed_image = page_ctx.original_image.copy()


            # 3. Layout Analysis (Placeholder - would populate page_ctx.layout_regions)
            # Conceptual: if self.layout_analyzer: self.layout_analyzer.process(doc_context)
            if self.layout_analyzer and self.layout_analyzer.is_enabled():
                for i, page_ctx_la in enumerate(doc_context.pages):
                    if page_ctx_la.preprocessed_image is None:
                        self.logger.warning(f"Skipping layout analysis for page {i} as preprocessed_image is missing.")
                        continue
                    try:
                        page_la_start_time = time.time()
                        # Pass module-specific config from doc_context.global_config
                        la_module_config = doc_context.global_config.get("modules", {}).get(self.layout_analyzer.module_id, {})
                        self.layout_analyzer.process(
                            page_ctx_la.preprocessed_image,
                            page_ctx_la,
                            config_override=la_module_config
                        )
                        page_ctx_la.processing_times["layout_analysis"] = time.time() - page_la_start_time
                        self.logger.info(f"Layout analysis completed for page {i} of {doc_id}.")
                    except Exception as e_la:
                        err_msg = f"Error during layout analysis for page {i} of {doc_id}: {e_la}"
                        self.logger.error(err_msg, exc_info=True)
                        page_ctx_la.errors.append(err_msg)
            else:
                self.logger.info("LayoutAnalyzer module is disabled or not initialized. Populating with dummy full-page regions.")
                for page_ctx_la_dummy in doc_context.pages:
                    if not page_ctx_la_dummy.layout_regions:
                        if page_ctx_la_dummy.preprocessed_image is not None:
                            h, w = page_ctx_la_dummy.preprocessed_image.shape[:2]
                            page_ctx_la_dummy.layout_regions.append(
                                TextRegion(region_id=f"page_{page_ctx_la_dummy.page_number}_full_region_0", bounding_box=(0,0,w,h))
                            )
                            self.logger.info(f"Created dummy full-page TextRegion for page {page_ctx_la_dummy.page_number} as fallback.")
                        else:
                            self.logger.warning(f"Cannot create dummy region for page {page_ctx_la_dummy.page_number}, preprocessed_image is None.")

            # 4. Text Recognition (using RecognitionCoreManager)
            if not self.recognition_manager:
                 raise OCRXConfigurationError("RecognitionCoreManager module is not initialized.")

            if self.recognition_manager.is_enabled():
                for i, page_ctx_rec in enumerate(doc_context.pages):
                    if not page_ctx_rec.layout_regions:
                        self.logger.info(f"No layout regions to process for text recognition on page {i}. Skipping.")
                        continue
                    try:
                        page_rec_start_time = time.time()
                        rec_module_config = doc_context.global_config.get("modules", {}).get(self.recognition_manager.module_id, {})
                        # Pass engine and language hint via config_override for RecognitionCoreManager if needed
                        # Example: rec_module_config["engine_id_to_use"] = requested_engine_name
                        # rec_module_config["language_hint"] = language_hint
                        # These would be read by RecognitionCoreManager.process()
                        if requested_engine_name: rec_module_config["engine_id_to_use"] = requested_engine_name
                        if language_hint: rec_module_config["language_hint"] = language_hint

                        self.recognition_manager.process(page_ctx_rec, config_override=rec_module_config)
                        page_ctx_rec.processing_times["recognition"] = time.time() - page_rec_start_time
                        self.logger.info(f"Text recognition phase completed for page {i} of {doc_id}.")
                    except Exception as e_rec_mgr:
                        err_msg = f"Error during recognition manager processing for page {i} of {doc_id}: {e_rec_mgr}"
                        self.logger.error(err_msg, exc_info=True)
                        page_ctx_rec.errors.append(err_msg)
            else:
                self.logger.info("RecognitionCoreManager module is disabled. Skipping text recognition.")

            # 5. Postprocessing / Consolidation (Placeholder)
            self.logger.info("Text postprocessing/consolidation placeholder: No changes made to raw OCR results.")
            # Conceptual: if self.output_generator: self.output_generator.process(doc_context)

            # 5. Post-processing (per page)
            if not self.postprocessor:
                raise OCRXConfigurationError("AdvancedPostprocessor module is not initialized.")

            if self.postprocessor.is_enabled():
                for i, page_ctx_postp in enumerate(doc_context.pages):
                    if not page_ctx_postp.layout_regions: # Or if no recognition results to postprocess
                        self.logger.info(f"Skipping post-processing for page {i} as no layout regions or prior results.")
                        continue
                    try:
                        page_postp_start_time = time.time()
                        postp_module_config = doc_context.global_config.get("modules", {}).get(self.postprocessor.module_id, {})
                        self.postprocessor.process(page_ctx_postp, config_override=postp_module_config)
                        page_ctx_postp.processing_times["postprocessing"] = time.time() - page_postp_start_time
                        self.logger.info(f"Post-processing completed for page {i} of {doc_id}.")
                    except Exception as e_postp:
                        err_msg = f"Error during post-processing for page {i} of {doc_id}: {e_postp}"
                        self.logger.error(err_msg, exc_info=True)
                        page_ctx_postp.errors.append(err_msg)
            else:
                self.logger.info("AdvancedPostprocessor module is disabled. Skipping post-processing.")


            doc_context.overall_status = "completed"
            if any(p.errors for p in doc_context.pages) or doc_context.document_errors:
                 doc_context.overall_status = "completed_with_errors"
            self.logger.info(f"Core document processing for '{doc_id}' finished with status: {doc_context.overall_status}.")

        except (OCRXInputError, FileNotFoundError, ValueError, OCRXConfigurationError, OCRXProcessingError) as e:
            self.logger.error(f"Processing aborted for '{doc_id}': {e}", exc_info=True)
            doc_context.document_errors.append(str(e))
            doc_context.overall_status = "failed"
        except Exception as e: # Catch-all for unexpected errors during main pipeline
            crit_error = f"An unexpected critical error occurred during main processing pipeline for '{doc_id}': {e}"
            self.logger.critical(crit_error, exc_info=True)
            doc_context.document_errors.append(crit_error)
            doc_context.overall_status = "failed"

        # 6. Output Generation (runs even if there were errors, to output partial results/error info)
        if not self.output_generator:
            self.logger.error("OutputGenerator module is not initialized. Cannot generate outputs.")
            # Not raising OCRXConfigurationError here as main processing might be done.
        elif self.output_generator.is_enabled():
            try:
                output_gen_start_time = time.time()
                output_module_config = doc_context.global_config.get("modules", {}).get(self.output_generator.module_id, {})
                self.output_generator.process(doc_context, config_override=output_module_config)
                # Output generator might add its own time to doc_context or a global time log
                if "output_generation" not in doc_context.processing_times: # Ensure key exists if not set by module
                    doc_context.processing_times["output_generation"] = time.time() - output_gen_start_time
                self.logger.info(f"Output generation completed for document {doc_id}.")
            except Exception as e_out:
                err_msg = f"Error during output generation for document {doc_id}: {e_out}"
                self.logger.error(err_msg, exc_info=True)
                doc_context.document_errors.append(err_msg)
                if doc_context.overall_status != "failed": # Don't override if already failed
                    doc_context.overall_status = "output_generation_failed"
        else:
            self.logger.info("OutputGenerator module is disabled. Skipping output generation.")

        # Calculate total processing time from page times and output generation time
        total_time = 0
        for page_ctx_time in doc_context.pages:
            total_time += sum(pt for pt in page_ctx_time.processing_times.values() if isinstance(pt, (int, float)))
        total_time += doc_context.processing_times.get("output_generation", 0.0)
        doc_context.total_processing_time = total_time

        return doc_context


if __name__ == '__main__':
    # This block is now significantly simplified.
    # For detailed testing, use pytest and specific test files.
    # For a runnable demo, use scripts/main.py or a dedicated demo script.
    print("OCRWorkflowOrchestrator basic functionality can be tested via unit tests or scripts/main.py.")

    # Example of a very basic instantiation test (requires a config file)
    # Create a minimal dummy config for this:
    minimal_config_path = "temp_minimal_config_orchestrator.yaml"
    with open(minimal_config_path, "w", encoding="utf-8") as f:
        f.write("app_settings:\n  default_ocr_engine: \"any_engine\"\nocr_engines: {}\nlogging:\n  version: 1\n  handlers:\n    console:\n      class: logging.StreamHandler\n      level: INFO\n  root:\n    handlers: [console]\n    level: INFO")

    try:
        print(f"\nAttempting basic instantiation with '{minimal_config_path}'...")
        orchestrator = OCRWorkflowOrchestrator(config_path=minimal_config_path)
        print(f"Orchestrator instantiated: {orchestrator}")
        print(f"Available engines: {orchestrator.engine_manager.get_available_engines()}")
        # To actually run process_document, a dummy image and a loadable dummy engine config would be needed.
        # print("\nTo test process_document, ensure a valid image path and engine config are provided.")
    except Exception as e:
        print(f"Error during basic instantiation test: {e}")
    finally:
        if os.path.exists(minimal_config_path):
            os.remove(minimal_config_path)
