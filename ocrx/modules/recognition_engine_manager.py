import logging
import importlib
from typing import Any, Dict, Optional, List

from ocrx.core.module_base import OCRXModuleBase
from ocrx.core.data_objects import PageContext, TextRegion, RecognitionResult
from ocrx.core.ocr_engine_interface import OCREngine # For type hinting self.engines
from ocrx.core.exceptions import OCRXConfigurationError, OCRXProcessingError


class RecognitionCoreManager(OCRXModuleBase):
    """
    Manages and orchestrates one or more OCR engines for text recognition.
    """
    def __init__(self, module_id: str, config: Dict[str, Any]):
        self.engines: Dict[str, OCREngine] = {}
        # Config for this manager itself, not to be confused with individual engine configs
        # The actual engine configurations are nested under "engines_config"
        super().__init__(module_id, config)

    def _validate_config(self) -> None:
        super()._validate_config()
        if "engines_config" not in self.config or not isinstance(self.config["engines_config"], dict):
            self.logger.warning("'engines_config' not found or not a dict in RecognitionCoreManager config. No engines will be loaded.")
            self.config["engines_config"] = {} # Ensure it exists as empty dict

        self.config.setdefault("default_engine_id", None) # Optional: specify a default engine to use
        if self.config["default_engine_id"] and not isinstance(self.config["default_engine_id"], str):
            raise OCRXConfigurationError(f"{self.module_id}: 'default_engine_id' must be a string if provided.")

        self.logger.info(f"{self.module_id} validated config. Default engine ID: {self.config['default_engine_id']}")


    def _initialize_resources(self) -> None:
        """
        Initializes configured OCR engines.
        """
        super()._initialize_resources()
        self.logger.info("Initializing OCR engines based on 'engines_config'...")

        engines_to_load_config = self.config.get("engines_config", {})
        if not engines_to_load_config:
            self.logger.warning("No engines defined under 'engines_config'. RecognitionCoreManager will have no active engines.")
            return

        for engine_id, single_engine_config in engines_to_load_config.items():
            if not isinstance(single_engine_config, dict):
                self.logger.error(f"Configuration for engine_id '{engine_id}' is not a dictionary. Skipping.")
                continue

            if not single_engine_config.get("enabled", False):
                self.logger.info(f"Engine_id '{engine_id}' is disabled in configuration. Skipping.")
                continue

            class_path = single_engine_config.get("class_path")
            if not class_path:
                self.logger.error(f"Engine_id '{engine_id}' configuration is missing 'class_path'. Skipping.")
                continue

            try:
                module_name, class_name = class_path.rsplit(".", 1)
                self.logger.debug(f"Attempting to load engine '{engine_id}' from module '{module_name}', class '{class_name}'.")

                engine_module = importlib.import_module(module_name)
                engine_class = getattr(engine_module, class_name)

                # Pass the specific config for this engine instance
                # This config is used by the engine wrapper (e.g., PaddleOCREngineWrapper's __init__)
                instance_params = single_engine_config.get("params", {})

                child_logger = self.logger.getChild(engine_id) # Create a child logger for the engine instance
                engine_instance: OCREngine = engine_class(engine_config=instance_params, logger=child_logger)

                # Engine wrappers should handle their own internal initialization (e.g. loading models)
                # in their __init__ or a separate initialize method called by OCREngineManager or orchestrator.
                # OCREngine interface has an initialize() method.
                engine_instance.initialize()

                if engine_instance.is_available():
                    self.engines[engine_id] = engine_instance
                    self.logger.info(f"Successfully loaded and initialized engine_id: '{engine_id}' ({engine_instance.get_engine_name()}).")
                else:
                    self.logger.warning(f"Engine_id '{engine_id}' ({engine_instance.get_engine_name()}) loaded but reported NOT available after initialization.")

            except ImportError as e:
                self.logger.error(f"Failed to import module for engine_id '{engine_id}' (path: {class_path}): {e}", exc_info=True)
            except AttributeError as e:
                self.logger.error(f"Failed to get class from module for engine_id '{engine_id}' (path: {class_path}): {e}", exc_info=True)
            except Exception as e: # Catch exceptions from engine's constructor or initialize()
                self.logger.error(f"Failed to instantiate or initialize engine_id '{engine_id}': {e}", exc_info=True)

        self.logger.info(f"Engine loading complete for RecognitionCoreManager. {len(self.engines)} engines are available: {list(self.engines.keys())}")


    def process(self, page_ctx: PageContext, config_override: Optional[Dict] = None) -> None:
        """
        Performs text recognition on all text regions within a PageContext.

        Args:
            page_ctx: The PageContext object containing layout_regions with image_crops.
            config_override: Optional runtime configuration. Can specify `engine_id_to_use`.
        """
        if not self.is_enabled():
            self.logger.info(f"Module {self.module_id} is disabled. Skipping recognition.")
            return

        if not page_ctx.layout_regions:
            self.logger.info(f"No layout regions found on page {page_ctx.page_number}. Skipping recognition.")
            return

        current_config = {**self.config, **(config_override or {})}
        engine_id_to_use = current_config.get("engine_id_to_use", self.config.get("default_engine_id"))

        if not self.engines:
            self.logger.error("No OCR engines loaded/available in RecognitionCoreManager. Cannot process.")
            raise OCRXConfigurationError("No OCR engines available for recognition.")

        if not engine_id_to_use:
            # If no specific or default engine, pick the first available one
            if self.engines:
                engine_id_to_use = list(self.engines.keys())[0]
                self.logger.info(f"No specific or default engine ID provided. Using first available: '{engine_id_to_use}'.")
            else: # Should have been caught by the check above, but as a safeguard
                self.logger.error("No engine specified and no engines available to choose from.")
                raise OCRXConfigurationError("No OCR engine specified or available.")

        selected_engine = self.engines.get(engine_id_to_use)
        if not selected_engine or not selected_engine.is_available():
            self.logger.error(f"Requested/default OCR engine '{engine_id_to_use}' is not available. Available: {list(self.engines.keys())}")
            # Fallback: try any other available engine
            available_engines = [id for id, eng in self.engines.items() if eng.is_available()]
            if not available_engines:
                raise OCRXProcessingError(f"Engine '{engine_id_to_use}' unavailable and no fallback engines available.")

            original_choice = engine_id_to_use
            engine_id_to_use = available_engines[0]
            selected_engine = self.engines[engine_id_to_use]
            self.logger.warning(f"Falling back to engine '{engine_id_to_use}' as '{original_choice}' was unavailable.")

        self.logger.info(f"Processing page {page_ctx.page_number} with {len(page_ctx.layout_regions)} regions using engine: '{engine_id_to_use}' ({selected_engine.get_engine_name()})")

        regions_processed_count = 0
        for i, text_region in enumerate(page_ctx.layout_regions):
            if text_region.image_crop is None:
                self.logger.warning(f"Skipping region {text_region.region_id} on page {page_ctx.page_number}: image_crop is None.")
                page_ctx.errors.append(f"Region {text_region.region_id}: Missing image_crop for recognition.")
                continue

            try:
                # language_hint can be part of config_override or a TextRegion property in future
                language_hint = current_config.get("language_hint")

                # `recognize` method in OCREngine interface returns List[RecognitionResult]
                recognition_results: List[RecognitionResult] = selected_engine.recognize(
                    text_region.image_crop,
                    language_hint=language_hint
                )

                # Extend raw_ocr_results with all results from this engine for this region
                text_region.raw_ocr_results.extend(recognition_results)
                self.logger.debug(f"Region {text_region.region_id} processed by {engine_id_to_use}, got {len(recognition_results)} recognition segment(s).")
                regions_processed_count +=1

            except Exception as e:
                error_msg = f"Error recognizing text in region {text_region.region_id} (page {page_ctx.page_number}) using {engine_id_to_use}: {e}"
                self.logger.error(error_msg, exc_info=True)
                page_ctx.errors.append(error_msg)

        self.logger.info(f"Recognition completed for page {page_ctx.page_number}. Processed {regions_processed_count}/{len(page_ctx.layout_regions)} regions with '{engine_id_to_use}'.")
        if regions_processed_count < len(page_ctx.layout_regions):
            self.logger.warning(f"Not all regions were processed for page {page_ctx.page_number} due to missing crops or errors.")
