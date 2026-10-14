#!/usr/bin/env bash
#
# Build the Zandpack documentation.
#
#   Zandpack/docs/build.sh
#
# The output goes to docs/_build/html/index.html. Everything the build needs
# beyond the docs dependencies (namely an importable Zandpack and pandoc, which
# nbsphinx uses for the notebook tutorials) is checked below.

set -euo pipefail

docs_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${docs_dir}/../.." && pwd)"
out_dir="${docs_dir}/_build/html"

# -- Docs dependencies -------------------------------------------------------
if ! python -c "import sphinx, pydata_sphinx_theme, myst_parser, sphinx_design, nbsphinx, sphinx_copybutton, sphinxcontrib.video" >/dev/null 2>&1; then
    echo "==> Installing documentation dependencies"
    python -m pip install -r "${docs_dir}/requirements.txt"
fi

# -- Zandpack itself (autodoc imports it) ------------------------------------
if ! python -c "import Zandpack" >/dev/null 2>&1; then
    echo "==> Zandpack is not importable; installing it (editable)"
    python -m pip install -e "${repo_root}"
fi

# -- pandoc (nbsphinx needs it for the notebooks) ----------------------------
if ! command -v pandoc >/dev/null 2>&1; then
    echo "WARNING: pandoc was not found on your PATH." >&2
    echo "         The notebook tutorials will fail to build. Install it with" >&2
    echo "         your package manager, e.g.  sudo pacman -S pandoc" >&2
    echo "         (or apt/dnf/brew)." >&2
fi

# -- Build -------------------------------------------------------------------
echo "==> Building documentation into ${out_dir}"
rm -rf "${docs_dir}/_build" "${docs_dir}/api"
python -m sphinx -b html "${docs_dir}" "${out_dir}"

echo
echo "Done. Open the docs with:"
echo "    file://${out_dir}/index.html"
