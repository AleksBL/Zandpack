#!/usr/bin/env bash
#
# Build the Zandpack documentation.
#
#   Zandpack/docs/build.sh [--strict]
#
#     --strict   treat Sphinx warnings as errors (-W --keep-going). Use this in
#                CI; it only passes once the build is warning-free.
#
# Output: docs/_build/html/index.html

set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") [--strict]" >&2
    echo "  --strict   fail the build on Sphinx warnings" >&2
}

strict=0
case "${1:-}" in
    --strict) strict=1 ;;
    -h|--help) usage; exit 0 ;;
    "") ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
esac

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
sphinx_args=(-b html)
if (( strict )); then
    sphinx_args+=(-W --keep-going)
fi
sphinx_args+=("${docs_dir}" "${out_dir}")

echo "==> Building documentation into ${out_dir}"
rm -rf "${docs_dir}/_build" "${docs_dir}/api"
python -m sphinx "${sphinx_args[@]}"

echo
echo "Done. Open the docs with:"
echo "    file://${out_dir}/index.html"
