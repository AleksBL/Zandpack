# Configuration file for the Sphinx documentation builder.
#
# Full reference: https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import PackageNotFoundError, version as _dist_version
from html import escape
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


# -- tooldeps directive ------------------------------------------------------
# Renders what the tools depend on: for every command-line tool, the Zandpack
# modules and names it imports, as a graph and as a table. Everything is read
# from the tool sources at build time, so neither the picture nor the table can
# drift away from the code. Usage:
#
#     .. tooldeps::
#     .. tooldeps:: TimedependentTransport.py:TD_Transport wrapper.py:wrapper

# Tools are listed in the order they are introduced on the page; the remaining
# command-line tools follow alphabetically, and any file passed as an argument
# (a Python driver, say) comes last.
_TOOL_ORDER = ["SCF", "psinought", "modify_occupations", "nozand", "zand"]

# One colour per module in the graph, cycled. The first colours go to the
# modules that most tools depend on.
_TOOL_COLOURS = ["#e79a00", "#2f7ea8", "#3f8f6b", "#9b5ea6", "#c1553f", "#6b7a99"]

_TOOL_IMPORT_FROM = re.compile(
    r"from\s+(Zandpack[\w.]*)\s+import\s*(?:\(([^)]*)\)|([^\n(]+))"
)
_TOOL_IMPORT_MODULE = re.compile(r"^[ \t]*import\s+(Zandpack[\w.]*)", re.M)
# Comments and string literals are removed before parsing: some modules build
# their own Imports as text (wrapper.py writes ``from Zandpack.wrapper import
# ...`` lines into the files it generates), and those are not dependencies.
_TOOL_COMMENT = re.compile(r"#[^\n]*")
_TOOL_STRING = re.compile(
    r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|\'[^\'\n]*\''
)


def _tool_code(path):
    """Return a tool's source with comments and string literals removed."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return _TOOL_STRING.sub("", _TOOL_COMMENT.sub("", text))


def _tool_files(extra, skip=()):
    """Return the ``[(label, path)]`` of every tool, in presentation order."""
    package = Path(__file__).resolve().parents[1]
    tools = {}
    for path in sorted((package / "cmdtools").iterdir()):
        # The tools are extensionless executables; *_help.txt and *.npy are not.
        if path.is_file() and not path.suffix:
            tools[path.name] = path
    for name in ("nozand", "zand"):
        path = package / "mpi" / name
        if path.is_file():
            tools[name] = path
    for name in skip:
        tools.pop(name, None)

    ordered = [(name, tools.pop(name)) for name in _TOOL_ORDER if name in tools]
    ordered += sorted(tools.items(), key=lambda tool: tool[0].lower())

    for spec in extra:
        filename, _, label = spec.partition(":")
        path = package / filename
        if path.is_file():
            ordered.append((label or path.stem, path))
    return ordered


def _tool_dependencies(path):
    """Return ``{module: [imported names]}`` for one tool's source."""
    deps = {}
    for match in _TOOL_IMPORT_FROM.finditer(_tool_code(path)):
        module = match.group(1).removeprefix("Zandpack.")
        names = match.group(2) or match.group(3) or ""
        parsed = []
        for name in names.replace("\\", " ").split(","):
            name = name.split(" as ")[0].strip()
            if name:
                parsed.append(name)
        if module == "Zandpack":
            # "from Zandpack import k0nfig" pulls in a submodule.
            for name in parsed:
                deps.setdefault(name, [])
        elif parsed:
            deps.setdefault(module, []).extend(parsed)
        else:
            deps.setdefault(module, [])
    for match in _TOOL_IMPORT_MODULE.finditer(_tool_code(path)):
        deps.setdefault(match.group(1).removeprefix("Zandpack."), [])
    # Names imported more than once (a lazy import inside a function, say) are
    # recorded once.
    return {module: list(dict.fromkeys(names)) for module, names in deps.items()}


def _module_colours(tools):
    """Return ``(modules, colour)``: the modules used, most used first."""
    counts = {}
    for _label, deps in tools:
        for module in deps:
            counts[module] = counts.get(module, 0) + 1
    modules = sorted(counts, key=lambda module: (-counts[module], module))
    colour = {
        module: _TOOL_COLOURS[i % len(_TOOL_COLOURS)]
        for i, module in enumerate(modules)
    }
    return modules, colour


def _tool_graph_svg(tools):
    """Return an inline SVG of the tool -> module dependency graph."""
    modules, colour = _module_colours(tools)

    row, pad = 26, 10
    left_x, left_w = 10, 172
    right_x, right_w = 560, 250
    width = right_x + right_w + 10
    height = max(len(tools) * (row + 10), len(modules) * (row + 12)) + 2 * pad
    tool_y = {label: pad + i * (row + 10) for i, (label, _) in enumerate(tools)}
    module_y = {m: pad + i * (row + 12) for i, m in enumerate(modules)}

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" class="zd-toolgraph"'
        f' viewBox="0 0 {width} {height}"',
        f' width="{width}" height="{height}" role="img"',
        ' aria-label="Dependency graph of the Zandpack tools">',
        "<title>Dependency graph of the Zandpack tools</title>",
        # Every tool carries its own edges, so that CSS can fade the rest of the
        # graph while the pointer rests on one tool (see custom.css).
        '<g class="zd-rows">',
    ]
    for label, deps in tools:
        svg.append(f'<g class="zd-row" data-tool="{escape(label)}">')
        for module, names in deps.items():
            y1 = tool_y[label] + row / 2
            y2 = module_y[module] + row / 2
            # Thicker line = more names imported from that module.
            stroke = 1.6 if not names else min(1 + 0.35 * len(names), 3.5)
            svg.append(
                f'<line x1="{left_x + left_w}" y1="{y1:.1f}"'
                f' x2="{right_x}" y2="{y2:.1f}"'
                f' stroke="{colour[module]}" stroke-width="{stroke:.2f}"'
                ' opacity="0.55"/>'
            )
        y = tool_y[label]
        svg.append(
            f'<rect class="zd-box zd-tool" x="{left_x}" y="{y}"'
            f' width="{left_w}" height="{row}" rx="6"/>'
        )
        svg.append(
            f'<text x="{left_x + 8}" y="{y + row / 2 + 4:.1f}">{escape(label)}</text>'
        )
        svg.append("</g>")
    svg.append("</g>")
    for module in modules:
        y = module_y[module]
        svg.append(
            f'<rect class="zd-box" x="{right_x}" y="{y}" width="{right_w}"'
            f' height="{row}" rx="6" stroke="{colour[module]}"/>'
        )
        svg.append(
            f'<text x="{right_x + 8}" y="{y + row / 2 + 4:.1f}">'
            f"{escape(module)}</text>"
        )
    svg.append("</svg>")
    return "".join(svg)


def _tool_dependency_table(tools):
    """Return the RST of the table listing the imported names per tool."""
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 16 84",
        "   :class: tooldeps-table",
        "",
        "   * - Tool",
        "     - Imports from ``Zandpack``",
    ]
    for label, deps in tools:
        lines.append(f"   * - ``{label}``")
        if not deps:
            lines.append("     - nothing (standard library only)")
            continue
        for index, (module, names) in enumerate(deps.items()):
            imported = (
                ", ".join(f"``{name}``" for name in names)
                if names
                else "the module itself"
            )
            bullet = f"``{module}``: {imported}"
            lines.append(f"     - * {bullet}" if index == 0 else f"       * {bullet}")
    return lines


# -- the per-tool detail pictures --------------------------------------------
# The overview graph answers "which modules does this tool touch"; these smaller
# pictures go one level deeper, so a tool can be expanded into the exact names it
# imports -- linked to their pages in the API reference where those exist.

_PAGES_CACHE = {}


def _documented_pages():
    """Return the ``api`` pages of the documented modules, names and members.

    Read from the stub files autosummary generated for this build, so a link is
    only offered where the API reference really has a page for it.
    """
    if _PAGES_CACHE:
        return _PAGES_CACHE
    api = Path(__file__).resolve().parent / "api"
    members, modules = {}, {}
    for stub in api.glob("*/*.rst"):
        module, _, name = stub.stem.removeprefix("Zandpack.").partition(".")
        if module and name:
            members[(module, name)] = f"{stub.parent.name}/{stub.stem}.html#{stub.stem}"
    for stub in api.glob("*.rst"):
        modules[stub.stem.removeprefix("Zandpack.")] = f"{stub.stem}.html"
    _PAGES_CACHE.update(members=members, modules=modules)
    return _PAGES_CACHE


def _name_lines(entries, width):
    """Spread ``[(name, url)]`` over as many lines as *width* needs."""
    lines, current, used = [], [], 0.0
    for name, url in entries:
        size = len(name) * _DETAIL_CHAR + _DETAIL_GAP
        if current and used + size > width:
            lines.append(current)
            current, used = [], 0.0
        current.append((name, url))
        used += size
    if current:
        lines.append(current)
    return lines


_DETAIL_CHAR, _DETAIL_GAP = 6.6, 14


def _tool_detail_svg(label, deps, prefix, colour):
    """Picture of one tool: the modules it imports and the names it takes."""
    pages = _documented_pages()
    row_h, gap = 24, 8
    tree_x, module_x, module_w = 14, 30, 210
    names_x = module_x + module_w + 16
    width = 660
    names_w = width - names_x - 10

    rows, y = [], 34
    for module, names in deps.items():
        if names:
            entries = [
                (name, pages["members"].get((module, name))) for name in names
            ]
        else:
            entries = [("the module itself", None)]
        lines = _name_lines(entries, names_w)
        height = max(row_h, 13 * len(lines))
        rows.append(
            {
                "module": module,
                "url": pages["modules"].get(module),
                "lines": lines,
                "y": y,
                "h": height,
                "names": bool(names),
            }
        )
        y += height + gap

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" class="zd-tooldetail"'
        f' viewBox="0 0 {width} {y:.0f}" width="{width}" height="{y:.0f}"'
        f' role="img" aria-label="What {escape(label)} imports">',
        f"<title>What {escape(label)} imports</title>",
        f'<text class="zd-tools" x="{tree_x - 2}" y="18">{escape(label)}</text>',
    ]
    if rows:
        last = rows[-1]["y"] + rows[-1]["h"] / 2
        svg.append(
            f'<path class="zd-tree" d="M{tree_x} 28 L{tree_x} {last:.1f}"/>'
        )
    for row in rows:
        centre = row["y"] + row["h"] / 2
        svg.append(
            f'<path class="zd-tree" d="M{tree_x} {centre:.1f}'
            f' L{module_x} {centre:.1f}"/>'
        )
        box = (
            f'<rect class="zd-mod" x="{module_x}" y="{centre - 12:.1f}"'
            f' width="{module_w}" height="24" rx="6"'
            f' stroke="{colour.get(row["module"], _TOOL_COLOURS[0])}"/>'
        )
        text = (
            f'<text class="zd-tools" x="{module_x + 8}"'
            f' y="{centre + 4:.1f}">{escape(row["module"])}</text>'
        )
        if row["url"]:
            svg.append(f'<a href="{prefix}{row["url"]}">{box}{text}</a>')
        else:
            svg.append(box + text)
        for index, line in enumerate(row["lines"]):
            text_y = centre + 4 - (len(row["lines"]) - 1) * 6.5 + index * 13
            x = names_x
            for name, url in line:
                css = "zd-name" if row["names"] else "zd-note"
                text = (
                    f'<text class="{css}" x="{x:.1f}" y="{text_y:.1f}">'
                    f"{escape(name)}</text>"
                )
                svg.append(f'<a href="{prefix}{url}">{text}</a>' if url else text)
                x += len(name) * _DETAIL_CHAR + _DETAIL_GAP
    svg.append("</svg>")
    return "".join(svg)


def _tool_dependency_details(tools, prefix):
    """Return HTML with one expandable block per tool."""
    _modules, colour = _module_colours(tools)
    blocks = []
    for label, deps in tools:
        count = sum(len(names) for names in deps.values())
        summary = (
            f"no Zandpack imports"
            if not deps
            else f"{len(deps)} module{'s' if len(deps) != 1 else ''}, "
            f"{count} name{'s' if count != 1 else ''}"
        )
        picture = (
            _tool_detail_svg(label, deps, prefix, colour)
            if deps
            else "<p><em>Nothing outside the standard library.</em></p>"
        )
        blocks.append(
            f'<details class="zd-deps"><summary><code>{escape(label)}</code>'
            f' <span class="zd-count">{summary}</span></summary>'
            f'<div class="zd-deps-body">{picture}</div></details>'
        )
    return "".join(blocks)


class ToolDepsDirective(Directive):
    """Graph and table of what every tool imports from Zandpack."""

    required_arguments = 0
    optional_arguments = 10
    final_argument_whitespace = True
    option_spec = {"skip": directives.unchanged}
    has_content = False

    def run(self):
        specs = [spec for arg in self.arguments for spec in arg.split()]
        skip = {
            name.strip()
            for name in self.options.get("skip", "").split(",")
            if name.strip()
        }
        tools = [
            (label, _tool_dependencies(path))
            for label, path in _tool_files(specs, skip)
        ]
        # Links are written relative to the page holding the directive.
        docname = self.state.document.settings.env.docname
        prefix = "../" * docname.count("/") + "api/"
        html = _tool_graph_svg(tools) + _tool_dependency_details(tools, prefix)
        node = nodes.container()
        self.state.nested_parse(
            StringList([".. raw:: html", "", "   " + html], source=self.block_text),
            0,
            node,
        )
        return node.children


# -- toolflow directive ------------------------------------------------------
# Draws the workflow: which stage of a calculation follows which, which tools
# belong to a stage, and which tools hang off a stage as side branches. The
# stages are declared inline, in the section using the directive, so the text
# lives with the rest of the page; only the drawing happens here.
#
# Content syntax, one stage per line:
#
#     Step 1: <title> [| <branch tools> | <branch note>]
#       - <tools> | <note> [| <branch tools> | <branch note>]
#
# The optional branch fields hang a dashed box off that stage (or off the whole
# step, when they are given on the step line). A line
#
#     wrap <step number> | <tools> | <note>
#
# draws a frame around that whole step, for things that drive the step as a
# whole rather than sitting inside its flow. Tools and notes are plain text:
# they are drawn into the picture, not parsed as reStructuredText.

_FLOW_MAIN_X, _FLOW_MAIN_W = 14, 430
_FLOW_BRANCH_X, _FLOW_BRANCH_W = 600, 268
_FLOW_FORK_X = 520
_FLOW_HEAD_H = 30
_FLOW_STAGE_GAP = 18
_FLOW_GROUP_GAP = 26
_FLOW_GROUP_PAD = 14
_FLOW_WRAP_PAD = 12
# Rough character widths, used to wrap the notes: monospace tools vs. prose.
_FLOW_MONO_CHAR, _FLOW_NOTE_CHAR, _FLOW_BRANCH_CHAR = 7.4, 6.2, 6.0


def _flow_wrap(text, limit_chars):
    """Greedy word wrap, breaking words that do not fit on a line at all."""
    lines, current = [], ""
    for word in text.split():
        while len(word) > limit_chars:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:limit_chars])
            word = word[limit_chars:]
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _flow_branch(fields):
    """Return ``(tools, note, then_step)`` for an optional side branch.

    ``then_step`` is the step the flow continues into after the branch, taken
    from a trailing ``then step <n>`` field, or ``None``.
    """
    if not fields or not fields[0]:
        return None
    then = None
    if len(fields) > 2:
        match = re.match(r"then\s+step\s+(\d+)$", fields[2].strip(), re.I)
        if match:
            then = int(match.group(1))
    return fields[0], fields[1] if len(fields) > 1 else "", then


def _flow_steps(content):
    """Parse the declared flow into ``([(title, branch, [stages])], {step: wrap})``."""
    steps, wraps = [], {}
    for raw in content:
        if not raw.strip():
            continue
        fields = [field.strip() for field in raw.strip().removeprefix("- ").split("|")]
        if fields[0].lower().startswith("wrap "):
            wraps[int(fields[0].split()[1]) - 1] = _flow_branch(fields[1:])
        elif raw[:1].isspace() and steps:
            stage = [fields[0], fields[1] if len(fields) > 1 else "", _flow_branch(fields[2:])]
            steps[-1][2].append(stage)
        else:
            steps.append([fields[0], _flow_branch(fields[1:]), []])
    return steps, wraps


def _flow_frame(spec):
    """Geometry of the frame drawn around a whole step, or ``None``."""
    if not spec:
        return None
    tools, note, _then = spec
    tool_lines = _flow_wrap(tools, int((_FLOW_MAIN_W - 30) / _FLOW_MONO_CHAR))
    note_lines = _flow_wrap(note, int((_FLOW_MAIN_W - 30) / _FLOW_NOTE_CHAR))
    height = 10 + 15 * len(tool_lines) + 15 * len(note_lines)
    return {"tools": tool_lines, "note": note_lines, "h": height, "top": 0.0}


def _flow_merge(step, groups):
    """Where the flow leaving a side branch re-enters the diagram.

    The arrow points at the rectangle of the step itself, so it says what it
    joins; where the step is framed, it crosses the frame on the way in.
    """
    if not 1 <= step <= len(groups):
        return None
    group = groups[step - 1]
    return {
        "x": _FLOW_MAIN_X + _FLOW_MAIN_W,
        "y": min(group["y"] + 30, group["y"] + group["h"] - 10),
    }


def _flow_geometry(steps, also, wraps):
    """Work out where every box of the workflow picture goes."""
    inner_x = _FLOW_MAIN_X + 10
    inner_w = _FLOW_MAIN_W - 20
    groups, branches = [], []
    y, previous_bottom = 16, 0.0

    for index, (title, branch, stages) in enumerate(steps):
        frame = _flow_frame(wraps.get(index))
        if frame is not None:
            # Keep the frame clear of the group above it.
            y = max(y, previous_bottom + 12 + frame["h"] + 8)
            frame["top"] = y - frame["h"] - 8

        laid_out, stage_y = [], y + _FLOW_HEAD_H
        # Links from a stage inside a framed step leave from the frame, not from
        # the stage itself, so they do not cross the frames on the way out.
        link_x = (
            _FLOW_MAIN_X
            + _FLOW_MAIN_W
            + (_FLOW_WRAP_PAD if frame is not None else 0)
            + 2
        )
        for tools, note, stage_branch in stages:
            tool_lines = _flow_wrap(tools, int((inner_w - 12) / _FLOW_MONO_CHAR))
            note_lines = _flow_wrap(note, int((inner_w - 12) / _FLOW_NOTE_CHAR))
            height = 30 + 15 * (len(tool_lines) - 1) + 15 * len(note_lines)
            laid_out.append(
                {
                    "tools": tool_lines,
                    "note": note_lines,
                    "y": stage_y,
                    "h": height,
                }
            )
            if stage_branch:
                branches.append(
                    {
                        "spec": stage_branch,
                        "anchor": stage_y + height / 2,
                        "from_x": link_x,
                    }
                )
            stage_y += height + _FLOW_STAGE_GAP
        height = stage_y - _FLOW_STAGE_GAP + _FLOW_GROUP_PAD - y
        groups.append(
            {
                "title": title,
                "y": y,
                "h": height,
                "stages": laid_out,
                "frame": frame,
            }
        )
        if branch and laid_out:
            branches.append(
                {
                    "spec": branch,
                    "anchor": laid_out[-1]["y"] + laid_out[-1]["h"] / 2,
                    "from_x": link_x,
                }
            )
        previous_bottom = y + height
        y = previous_bottom + _FLOW_GROUP_GAP

    floor = 0
    for branch in branches:
        tools, note, then_step = branch["spec"]
        tool_lines = _flow_wrap(tools, int((_FLOW_BRANCH_W - 20) / _FLOW_MONO_CHAR))
        note_lines = _flow_wrap(note, int((_FLOW_BRANCH_W - 20) / _FLOW_BRANCH_CHAR))
        height = 24 + 15 * (len(tool_lines) - 1) + 14 * len(note_lines) + 8
        top = max(branch["anchor"] - height / 2, floor)
        branch.update(
            {
                "tools": tool_lines,
                "note": note_lines,
                "y": top,
                "h": height,
                "merge": _flow_merge(then_step, groups) if then_step else None,
            }
        )
        floor = top + height + 12

    also_lines = _flow_wrap(
        also, int((_FLOW_BRANCH_X + _FLOW_BRANCH_W - 40) / _FLOW_NOTE_CHAR)
    )
    also_box = {
        "y": max(y - _FLOW_GROUP_GAP, floor) + 18,
        "h": 24 + 15 * len(also_lines),
        "note": also_lines,
    }
    bottom = also_box["y"] + also_box["h"] + 16
    return {
        "groups": groups,
        "branches": branches,
        "also": also_box,
        "inner_x": inner_x,
        "inner_w": inner_w,
        "width": _FLOW_BRANCH_X + _FLOW_BRANCH_W + 12,
        "height": bottom,
    }


def _flow_svg(steps, also, wraps):
    """Return an inline SVG of the workflow picture."""
    plan = _flow_geometry(steps, also, wraps)
    inner_x, inner_w = plan["inner_x"], plan["inner_w"]
    mid_x = inner_x + inner_w / 2
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" class="zd-toolflow"'
        f' viewBox="0 0 {plan["width"]} {plan["height"]}"'
        f' width="{plan["width"]}" height="{plan["height"]}"'
        ' role="img" aria-label="How the Zandpack tools work together">',
        "<title>How the Zandpack tools work together</title>",
        "<defs>",
        '<marker id="zd-flow" viewBox="0 0 10 10" refX="9" refY="5"'
        ' markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#e79a00"/></marker>',
        '<marker id="zd-flow-link" viewBox="0 0 10 10" refX="9" refY="5"'
        ' markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#aab3bd"/></marker>',
        "</defs>",
    ]

    # Arrows of the main flow, drawn before the boxes. They stop at the frame of
    # a wrapped step rather than at the step's own rectangle, so the line meets
    # the frame instead of crossing it.
    previous_group = None
    for group in plan["groups"]:
        if previous_group is not None:
            frame = previous_group["frame"]
            from_y = previous_group["y"] + previous_group["h"]
            if frame is not None:
                from_y += _FLOW_WRAP_PAD
            to_y = group["frame"]["top"] if group["frame"] else group["y"]
            svg.append(
                f'<path class="zd-arrow" d="M{mid_x:.1f} {from_y:.1f}'
                f' L{mid_x:.1f} {to_y:.1f}" marker-end="url(#zd-flow)"/>'
            )
        for before, after in zip(group["stages"], group["stages"][1:]):
            gap = (after["y"] - (before["y"] + before["h"])) / 2
            svg.append(
                f'<path class="zd-arrow" d="M{mid_x:.1f}'
                f' {before["y"] + before["h"]:.1f} L{mid_x:.1f}'
                f' {before["y"] + before["h"] + gap:.1f}"'
                ' marker-end="url(#zd-flow)"/>'
            )
        previous_group = group

    # Dashed links from a stage to the tools that hang off it.
    for branch in plan["branches"]:
        svg.append(
            f'<path class="zd-link" d="M{branch["from_x"]} {branch["anchor"]:.1f}'
            f' L{_FLOW_FORK_X} {branch["anchor"]:.1f}'
            f' L{_FLOW_FORK_X} {branch["y"] + branch["h"] / 2:.1f}'
            f' L{_FLOW_BRANCH_X - 2} {branch["y"] + branch["h"] / 2:.1f}"'
            ' marker-end="url(#zd-flow-link)"/>'
        )

    # The flow of a side branch that carries on into a later step. It keeps the
    # grey of the side-branch links rather than the amber of the main flow.
    for branch in plan["branches"]:
        merge = branch["merge"]
        if merge is None:
            continue
        centre = _FLOW_BRANCH_X + _FLOW_BRANCH_W / 2
        svg.append(
            f'<path class="zd-link" d="M{centre} {branch["y"] + branch["h"]:.1f}'
            f' L{centre} {merge["y"]:.1f} L{merge["x"]:.1f} {merge["y"]:.1f}"'
            ' marker-end="url(#zd-flow-link)"/>'
        )

    for group in plan["groups"]:
        frame = group["frame"]
        if frame is not None:
            svg.append(
                f'<rect class="zd-frame" x="{_FLOW_MAIN_X - _FLOW_WRAP_PAD}"'
                f' y="{frame["top"]:.1f}"'
                f' width="{_FLOW_MAIN_W + 2 * _FLOW_WRAP_PAD}"'
                f' height="{group["y"] + group["h"] - frame["top"] + _FLOW_WRAP_PAD:.1f}"'
                ' rx="12"/>'
            )
            for index, line in enumerate(frame["tools"]):
                svg.append(
                    f'<text class="zd-tools"'
                    f' x="{_FLOW_MAIN_X - _FLOW_WRAP_PAD + 12}"'
                    f' y="{frame["top"] + 16 + 15 * index:.1f}">{escape(line)}</text>'
                )
            note_y = frame["top"] + 19 + 15 * len(frame["tools"])
            for index, line in enumerate(frame["note"]):
                svg.append(
                    f'<text class="zd-note" x="{_FLOW_MAIN_X - _FLOW_WRAP_PAD + 12}"'
                    f' y="{note_y + 15 * index:.1f}">{escape(line)}</text>'
                )
        svg.append(
            f'<rect class="zd-group" x="{_FLOW_MAIN_X}" y="{group["y"]}"'
            f' width="{_FLOW_MAIN_W}" height="{group["h"]:.1f}" rx="12"/>'
        )
        svg.append(
            f'<text class="zd-head" x="{inner_x}" y="{group["y"] + 20}">'
            f'{escape(group["title"])}</text>'
        )
        for stage in group["stages"]:
            svg.append(
                f'<rect class="zd-stage" x="{inner_x}" y="{stage["y"]}"'
                f' width="{inner_w}" height="{stage["h"]}" rx="8"/>'
            )
            for index, line in enumerate(stage["tools"]):
                svg.append(
                    f'<text class="zd-tools" x="{inner_x + 10}"'
                    f' y="{stage["y"] + 20 + 15 * index:.1f}">{escape(line)}</text>'
                )
            note_y = stage["y"] + 23 + 15 * len(stage["tools"])
            for index, line in enumerate(stage["note"]):
                svg.append(
                    f'<text class="zd-note" x="{inner_x + 10}"'
                    f' y="{note_y + 15 * index:.1f}">{escape(line)}</text>'
                )

    for branch in plan["branches"]:
        svg.append(
            f'<rect class="zd-branch" x="{_FLOW_BRANCH_X}" y="{branch["y"]}"'
            f' width="{_FLOW_BRANCH_W}" height="{branch["h"]}" rx="8"/>'
        )
        for index, line in enumerate(branch["tools"]):
            svg.append(
                f'<text class="zd-tools" x="{_FLOW_BRANCH_X + 10}"'
                f' y="{branch["y"] + 18 + 15 * index:.1f}">{escape(line)}</text>'
            )
        note_y = branch["y"] + 21 + 15 * len(branch["tools"])
        for index, line in enumerate(branch["note"]):
            svg.append(
                f'<text class="zd-note" x="{_FLOW_BRANCH_X + 10}"'
                f' y="{note_y + 14 * index:.1f}">{escape(line)}</text>'
            )

    box = plan["also"]
    svg.append(
        f'<rect class="zd-branch" x="{_FLOW_MAIN_X}" y="{box["y"]:.1f}"'
        f' width="{_FLOW_BRANCH_X + _FLOW_BRANCH_W - _FLOW_MAIN_X}"'
        f' height="{box["h"]}" rx="8"/>'
    )
    svg.append(
        f'<text class="zd-note" x="{_FLOW_MAIN_X + 10}" y="{box["y"] + 18:.1f}">'
        f'{escape(box["note"][0])}</text>'
    )
    for index, line in enumerate(box["note"][1:], start=1):
        svg.append(
            f'<text class="zd-note" x="{_FLOW_MAIN_X + 10}"'
            f' y="{box["y"] + 18 + 15 * index:.1f}">{escape(line)}</text>'
        )
    svg.append("</svg>")
    return "".join(svg)


class ToolFlowDirective(Directive):
    """Workflow picture of the tools, drawn from the declared steps."""

    required_arguments = 0
    optional_arguments = 0
    option_spec = {"also": directives.unchanged}
    has_content = True

    def run(self):
        steps, wraps = _flow_steps(self.content)
        svg = _flow_svg(steps, self.options.get("also", ""), wraps)
        rst = [".. raw:: html", "", "   " + svg]
        node = nodes.container()
        self.state.nested_parse(StringList(rst, source=self.block_text), 0, node)
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
    app.add_directive("tooldeps", ToolDepsDirective)
    app.add_directive("toolflow", ToolFlowDirective)
    app.add_directive("autosummary", ModuleListAutosummary, override=True)
    app.connect("autodoc-process-docstring", _strip_editor_headers)
    return {"parallel_read_safe": True}
