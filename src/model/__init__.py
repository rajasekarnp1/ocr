# src/model/__init__.py
"""
Makes the 'model' directory a Python package and centralizes model component imports.
"""

# Import model components to make them accessible via `from model import ...`
# For example, if you have a DiffusionModel class in diffusion.py:
# from .diffusion import DiffusionModel
# from .unet import UNet # Assuming you have a U-Net architecture
# from .condition import ConditionEncoder # If you have a separate conditioning module
# from .utils import PositionalEncoding, get_noise_schedule # Model-specific utilities


# You can also define an __all__ list to specify what gets imported with `from model import *`
# __all__ = ['DiffusionModel', 'UNet', 'ConditionEncoder', 'PositionalEncoding', 'get_noise_schedule']

print("src.model package initialized.")
