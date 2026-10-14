# Configuration file for the Sphinx documentation builder.
#
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
import re
import sys
from types import ModuleType

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx.ext.autosummary import Autosummary

# -- Path setup --------------------------------------------------------------
# This file lives in <repo>/Zandpack/docs, so the importable package root is two
# levels up (<repo>). Anchoring on __file__ keeps the build independent of the
# process working directory.

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Project information -----------------------------------------------------
project = "Zandpack"
author = "Aleksander Bach Lorentzen"
copyright = "2025, Aleksander Bach Lorentzen"

try:
    release = _dist_version("Zandpack")
except PackageNotFoundError:
    # Fall back so the docs still build before an editable install exists.
    release = "1.1"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # NumPy / Google style docstring sections
    "sphinx_design",                # grid cards on the landing page
    "myst_parser",                  # parse and include README.md
    "nbsphinx",                     # notebook tutorials
    "sphinx_copybutton",            # copy button on code blocks
    "sphinxcontrib.video",          # .. video:: directives in cmdtoolsdoc
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"

# Autosummary writes one stub page per listed module at build time, into
# docs/api/ (not committed; see api.rst for the curated module list).
autosummary_generate = True

autodoc_typehints = "description"
autodoc_member_order = "groupwise"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# -- Autodoc docstring cleanup -----------------------------------------------
# Many modules carry an editor-generated header inside their module docstring:
# a "Created on <date>" line followed by an "@author" line. Neither says
# anything about the code, and the "@author" line was being picked up as the
# module summary in the API reference tables. Both are therefore dropped from
# the rendered docstring at build time, which also leaves the summary column
# free for real descriptions. The source files are left untouched.

_EDITOR_HEADER_LINE = re.compile(r"^\s*(Created on\s+\S+.*|@author:.*)$")


def _strip_editor_headers(app, what, name, obj, options, lines):
    """Drop editor-generated header lines from an autodoc docstring."""
    kept = [line for line in lines if not _EDITOR_HEADER_LINE.match(line)]
    # Removing the header usually leaves blank line(s) at the top.
    while kept and not kept[0].strip():
        kept.pop(0)
    lines[:] = kept

# Notebook tutorials are shipped pre-executed; never run them during the build.
nbsphinx_execute = "never"

todo_include_todos = True

# Markdown features needed by README.md (equations, links, lists).
myst_enable_extensions = [
    "dollarmath",
    "tasklist",
    "colon_fence",
    "deflist",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# README.md references its images with repo-root-relative paths
# ("./Zandpack/svg/..."). It is included into quickstart/readme.rst, so the
# browser resolves those under quickstart/. _extra/ mirrors that path so the
# images resolve in the built site without editing README.md.
html_extra_path = ["_extra"]
html_logo = "_static/logo3.svg"
html_title = "Zandpack"
html_theme_options = {
    "github_url": "https://github.com/AleksBL/Zandpack",
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "announcement": (
        "Zandpack v1.1 - see the CPC paper: doi:10.1016/j.cpc.2026.110087"
    ),
    "icon_links": [
        {
            "name": "CPC paper",
            "url": "https://doi.org/10.1016/j.cpc.2026.110087",
            "icon": "fa-solid fa-book",
            "type": "fontawesome",
        },
    ],
}

# Context for the "Edit this page" button (github_user/github_repo/github_version
# plus the path from the repository root to this documentation directory).
html_context = {
    "github_user": "AleksBL",
    "github_repo": "Zandpack",
    "github_version": "main",
    "doc_path": "Zandpack/docs",
}


# -- cmdhelp directive -------------------------------------------------------
# Renders a command-line tool's "*_help.txt" (shipped next to the tool) as a
# tidy keyword/default table, so the pages do not duplicate the defaults and
# stay in sync with the source of truth. Usage:
#
#     .. cmdhelp:: SCF
#     .. cmdhelp:: nozand
#        :caption: nozand keywords


def _find_help_file(tool):
    """Locate ``<tool>_help.txt`` under the package's tool directories."""
    base = Path(__file__).resolve().parents[1]
    for sub in ("cmdtools", "tools"):
        candidate = base / sub / f"{tool}_help.txt"
        if candidate.is_file():
            return candidate
    return None


def _help_keywords(tool):
    """Return the ``keyword, default`` pairs from a tool's help file."""
    path = _find_help_file(tool)
    if path is None:
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = 0
    for i, line in enumerate(lines):
        if "Keywords and default values" in line:
            start = i + 1
            break
    pairs = []
    for line in lines[start:]:
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*:\s*(.*?)\s*$", line)
        if match:
            pairs.append((match.group(1), match.group(2)))
    return pairs


class CmdHelpDirective(Directive):
    """Render a tool's help file as a keyword/default table."""

    required_arguments = 1
    has_content = False
    option_spec = {"caption": directives.unchanged}

    def run(self):
        tool = self.arguments[0]
        pairs = _help_keywords(tool)
        if not pairs:
            raise self.error(f"cmdhelp: no keyword help found for tool '{tool}'")
        caption = self.options.get(
            "caption", f"{tool}: keywords and default values"
        )
        rst = [
            f".. csv-table:: {caption}",
            '   :header: "Keyword", "Default"',
            "   :widths: 22 78",
            "   :class: cmdhelp-table",
            "",
        ]
        for key, value in pairs:
            key = key.replace('"', '""')
            value = value.replace('"', '""')
            rst.append(f'   "{key}", "{value}"')
        node = nodes.container()
        self.state.nested_parse(
            StringList(rst, source=self.block_text), 0, node
        )
        return node.children


# -- apimodules --------------------------------------------------------------
# The tables in api.rst list modules only. autosummary fills their second column
# with the first line of the module docstring, which for these modules is either
# missing or not a useful description. That column is left empty instead, ready
# for short module descriptions to be filled in later. Member tables (the ones
# on the generated module pages) keep their summaries.
#
# api.rst keeps writing ``.. autosummary::`` because that literal name is what
# the stub generator scans the sources for; the directive itself is overridden
# below, after the extension has registered its own version.

class ModuleListAutosummary(Autosummary):
    """``autosummary`` that blanks the summary column of module-only tables."""

    def get_items(self, names):
        items = super().get_items(names)
        if not all(
            isinstance(sys.modules.get(real_name), ModuleType)
            for _name, _sig, _summary, real_name in items
        ):
            return items
        return [
            (name, sig, "", real_name)
            for name, sig, _summary, real_name in items
        ]


def setup(app):
    app.add_directive("cmdhelp", CmdHelpDirective)
    app.add_directive("autosummary", ModuleListAutosummary, override=True)
    app.connect("autodoc-process-docstring", _strip_editor_headers)
    return {"parallel_read_safe": True}
