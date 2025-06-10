import pytest
import logging
from typing import Any, Dict
from ocrx.core.module_base import OCRXModuleBase
from ocrx.core.exceptions import OCRXConfigurationError

# --- Dummy Subclass for Testing ---
class DummyModule(OCRXModuleBase):
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)

    def _validate_config(self) -> None:
        super()._validate_config() # Call base validation
        # Add dummy-specific validation if needed for tests
        if "required_param" in self.config and self.config["required_param"] is None:
            raise OCRXConfigurationError(f"'{self.module_id}' config: 'required_param' cannot be None.")

    def _initialize_resources(self) -> None:
        super()._initialize_resources()
        if self.config.get("load_dummy_resource", False):
            self.logger.info(f"Dummy resource loaded for {self.module_id}")
            self.dummy_resource = "loaded"
        else:
            self.dummy_resource = None

    def load_model(self, model_path: str, **kwargs: Any) -> Any:
        # Override to provide a mock implementation for testing
        if not model_path:
            raise OCRXConfigurationError(f"Model path not provided for {self.module_id}")
        self.logger.info(f"Dummy loading model from '{model_path}' for {self.module_id} with {kwargs}...")
        return f"dummy_model_at_{model_path}"

    def process(self, data: Any, **kwargs: Any) -> Any:
        if not self.is_enabled():
            self.logger.warning(f"Module {self.module_id} process called while disabled.")
            return data # Or raise error, depending on desired behavior for disabled modules

        self.logger.info(f"DummyModule {self.module_id} processing data: {data}")
        if self.dummy_resource == "loaded":
            return f"Processed {data} with dummy_resource"
        return f"Processed {data}"

# --- Test Cases ---

def test_module_base_init_success():
    """Test successful initialization of a module derived from OCRXModuleBase."""
    module_id = "test_dummy_module"
    config = {"enabled": True, "some_param": "value"}
    module = DummyModule(module_id=module_id, config=config)

    assert module.module_id == module_id
    assert module.config == config
    assert module.is_enabled() is True
    assert isinstance(module.logger, logging.Logger)
    assert module.logger.name == f"ocrx.module.{module_id}"

def test_module_base_init_disabled():
    """Test initialization with module explicitly disabled in config."""
    module_id = "disabled_dummy"
    config = {"enabled": False}
    module = DummyModule(module_id=module_id, config=config)

    assert module.is_enabled() is False

def test_module_base_init_default_enabled():
    """Test that module is enabled by default if 'enabled' key is missing."""
    module_id = "default_enabled_dummy"
    config = {"some_other_param": "value"} # 'enabled' key missing
    module = DummyModule(module_id=module_id, config=config)

    assert module.is_enabled() is True

def test_module_base_repr():
    """Test the __repr__ method of OCRXModuleBase."""
    module = DummyModule(module_id="repr_test", config={})
    assert repr(module) == "<DummyModule(module_id='repr_test')>"

def test_validate_config_invalid_type():
    """Test _validate_config raises error if config is not a dict (though __init__ coerces)."""
    # __init__ currently ensures config is a dict. If that changes, this test is more relevant.
    # For now, we test the internal _validate_config's check if directly called or if __init__ changes.
    with pytest.raises(OCRXConfigurationError, match="must be a dictionary"):
        # Simulate scenario where config somehow bypasses __init__'s dict coercion
        module = DummyModule(module_id="bad_config_type", config={})
        module.config = "not a dict" # Manually set config to invalid type
        module._validate_config() # Call directly to test its logic

def test_validate_config_subclass_specific(caplog):
    """Test subclass-specific configuration validation."""
    with pytest.raises(OCRXConfigurationError, match="'required_param' cannot be None"):
        DummyModule(module_id="subclass_fail", config={"required_param": None})

    # Test successful subclass validation
    module = DummyModule(module_id="subclass_ok", config={"required_param": "valid"})
    assert module.config["required_param"] == "valid"

    # Test warning for disabled module during _validate_config
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        DummyModule(module_id="validate_disabled", config={"enabled": False})
    assert any(f"Module 'validate_disabled' is disabled via configuration." in record.message for record in caplog.records)


def test_initialize_resources_called_and_logged(caplog):
    """Test that _initialize_resources is called and logs debug message."""
    with caplog.at_level(logging.DEBUG):
        DummyModule(module_id="res_init_log", config={"load_dummy_resource": True})

    assert any(f"Initializing resources for res_init_log." in record.message for record in caplog.records)
    assert any(f"Dummy resource loaded for res_init_log" in record.message for record in caplog.records)


def test_load_model_base_not_implemented():
    """Test that calling load_model on base (if possible) or a subclass that hasn't implemented it raises error."""
    # This tests the placeholder in OCRXModuleBase itself.
    # Need a way to call it without a subclass overriding it.
    # One way: create a subclass that *doesn't* implement load_model
    class NoLoadModelModule(OCRXModuleBase):
        def process(self, data: Any, **kwargs: Any) -> Any: return data # Minimal implementation
        # No _validate_config, _initialize_resources, or load_model override

    module = NoLoadModelModule(module_id="no_load", config={})
    with pytest.raises(NotImplementedError, match="load_model() not implemented in NoLoadModelModule"):
        module.load_model(model_path="some/path.mod")

def test_load_model_dummy_implementation():
    """Test the dummy implementation of load_model in the DummyModule."""
    module = DummyModule(module_id="dummy_load", config={})
    model = module.load_model(model_path="dummy/model.pth", param="test")
    assert model == "dummy_model_at_dummy/model.pth"

def test_load_model_no_path():
    """Test load_model in DummyModule when model_path is not provided."""
    module = DummyModule(module_id="dummy_no_path", config={})
    with pytest.raises(OCRXConfigurationError, match="Model path not provided"):
        module.load_model(model_path="") # Empty path

def test_process_method_abstract():
    """Ensure process method is abstract and requires implementation."""
    # Attempting to instantiate OCRXModuleBase directly should fail if process is abstract
    # However, ABCs with abstract methods cannot be instantiated directly.
    # We test this by ensuring a subclass *must* implement it.

    # This will fail if process is not implemented by BadModule
    class BadModule(OCRXModuleBase):
        def __init__(self, module_id: str, config: Dict[str, Any]):
            super().__init__(module_id, config)
        # No process method

    with pytest.raises(TypeError) as excinfo:
        BadModule(module_id="bad", config={}) # type: ignore

    assert "Can't instantiate abstract class BadModule with abstract method process" in str(excinfo.value)


def test_dummy_module_process_method():
    """Test the process method of the DummyModule."""
    module_enabled = DummyModule(module_id="dummy_process_enabled", config={"load_dummy_resource": True})
    result = module_enabled.process("input_data")
    assert result == "Processed input_data with dummy_resource"

    module_disabled = DummyModule(module_id="dummy_process_disabled", config={"enabled": False})
    result_disabled = module_disabled.process("input_data_disabled")
    assert result_disabled == "input_data_disabled" # As per DummyModule's process logic for disabled

    module_no_resource = DummyModule(module_id="dummy_process_no_resource", config={"load_dummy_resource": False})
    result_no_res = module_no_resource.process("input_data_no_res")
    assert result_no_res == "Processed input_data_no_res"
