from setuptools import setup, find_packages
import os

# Function to read requirements from requirements.txt
def load_requirements(filename='requirements.txt'):
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. No requirements will be installed via setup.py.")
        return []
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Basic versioning
VERSION = "0.1.0-alpha"

setup(
    name="audio-upscaler",
    version=VERSION,
    author="AI Agent (Jules)",
    author_email="<placeholder_email@example.com>",
    description="A deep learning based audio upscaler (under development).",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="<placeholder_repository_url>", # TODO: Replace with actual URL
    packages=find_packages(where="src"), # Find packages in src
    package_dir={"": "src"},             # Tell distutils packages are under src
    # scripts=['src/cli.py'], # One way to make CLI accessible, but entry_points is better
    entry_points={
        'console_scripts': [
            'audio-upscaler=cli:main', # Assumes cli.py has a main() function
        ],
    },
    install_requires=load_requirements(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha", # Due to untested model
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # TODO: Add license classifier once chosen
    ],
    python_requires='>=3.8', # Based on common library compatibilities
    # include_package_data=True, # If you have non-code files inside your package
)
