from setuptools import setup, find_packages
import os

# Read package metadata from settings.py
metadata = {}
with open(os.path.join(os.path.dirname(__file__), "settings.py")) as f:
    exec(f.read(), metadata)

def read_requirements():
    with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def read_long_description():
    with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
        return f.read()

setup(
    name=metadata.get("PACKAGE_NAME", "graphrag_tagger"),
    version=metadata.get("VERSION", "0.1.1"),  # updated version
    description=metadata.get("DESCRIPTION", ""),
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author=metadata.get("AUTHOR", ""),
    author_email=metadata.get("AUTHOR_EMAIL", ""),
    url=metadata.get("URL", ""),
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.10",  # require Python 3.10+
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
