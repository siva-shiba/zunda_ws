# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# プロジェクトルートをパスに追加（autodoc で zunda を読むため）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

project = "zunda"
copyright = "2025"
author = "zunda"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # NumPy/Google style docstrings
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "zunda ドキュメント"

# autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
}

# 重い依存関係はCIでインストールせずにモックする
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "sklearn",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "mmengine",
    "mmpretrain",
    "openmim",
    "wandb",
    "neptune",
    "PIL",
]
