# Configuration file for Sphinx documentation

import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------
project = 'Unsupervised Machine Learning'
copyright = f'{datetime.now().year}, Ashwini Kumar Mathur'
author = 'Ashwini Kumar Mathur'
release = '1.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.graphviz',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    "logo": {
        "text": "Unsupervised ML",
        "image_light": "_static/logos/course_logo.png",
        "image_dark": "_static/logos/course_logo.png",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 5,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/USERNAME/unsupervised-ml-docs",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
        {
            "name": "Colab",
            "url": "https://colab.research.google.com/",
            "icon": "fas fa-laptop-code",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": False,
    "show_toc_level": 2,
    "navigation_depth": 3,
    "show_nav_level": 2,
    "collapse_navigation": False,
    "navigation_with_keys": True,
    "show_prev_next": True,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "primary_sidebar_end": ["indices"],
    "announcement": "📚 Course Material for Unsupervised Machine Learning - Fall 2025",
}

html_context = {
    "default_mode": "auto",
    "github_user": "USERNAME",
    "github_repo": "unsupervised-ml-docs",
    "github_version": "main",
    "doc_path": "source",
}

html_title = "Unsupervised ML Course"
html_favicon = None
html_show_sourcelink = False

# -- Options for LaTeX output ------------------------------------------------
latex_engine = 'pdflatex'
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{algorithm}
\usepackage{algorithmic}
''',
}

latex_documents = [
    ('index', 'UnsupervisedML.tex', 
     'Unsupervised Machine Learning Documentation',
     'Ashwini Kumar Mathur', 'manual'),
]

# -- MathJax configuration ---------------------------------------------------
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'processEscapes': True,
        'processEnvironments': True,
    },
    'options': {
        'skipHtmlTags': ['script', 'noscript', 'style', 'textarea', 'pre'],
    },
}

# -- Graphviz configuration --------------------------------------------------
graphviz_output_format = 'svg'
graphviz_dot_args = [
    '-Gfontname=Arial',
    '-Nfontname=Arial',
    '-Efontname=Arial',
]

# -- Copy button configuration -----------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True

# -- Todo extension ----------------------------------------------------------
todo_include_todos = True
todo_emit_warnings = True

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

