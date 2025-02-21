import os
import sys
sys.path.insert(0, os.path.abspath('../'))  # Adjust the path if needed


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'graphrag_tagger'
copyright = '2025, Alex Kameni'
author = 'Alex Kameni'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # If using Google-style or NumPy docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx'
]

# html_theme_options = {
#     "rightsidebar": "true",
#     "relbarbgcolor": "black"
# }

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'scrolls'
html_static_path = ['_static']
