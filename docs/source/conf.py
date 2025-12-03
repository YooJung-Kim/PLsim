# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../src/'))
sys.path.insert(0, os.path.abspath('../../src/PLsim'))


project = 'PLsim'
copyright = '2025, Yoo Jung Kim'
author = 'Yoo Jung Kim'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # Core library for API generation
    'sphinx.ext.mathjax',       # For LaTeX math
    'sphinx.ext.viewcode',      # "View Source" links
    'numpydoc',                 # For NumPy style docstrings
    'sphinx_automodapi.automodapi', # For the nice tables
    'nbsphinx',                 # For Jupyter Notebook support
    'sphinx.ext.githubpages',
]

numpydoc_show_class_members = False
templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

pygments_style = 'default'