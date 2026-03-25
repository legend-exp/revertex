# AGENTS.md — Documentation

Sphinx-based. Build: `cd docs && make`. Config in `docs/source/conf.py`. See
`Makefile` for other useful targets.

## Conventions

- Source in `docs/source/`
- Pages written in Markdown
- Colon fences (`:::`) for sphinx directives like admonitions. Always add an
  empty line after the open and before the close of a `:::` block, otherwise
  `prettier` will re-wrap and break rendering
- Always enable code block highlighting. Use `console` for shell commands
  (prompt delimiter `>`)
- Document each function and Snakemake rule in its docstring; higher-level docs
  go in the user manual
- All new pages must appear in a toctree — orphan pages cause build warnings
- API reference is auto-generated; do not write it by hand

## Python docstring conventions

- NumPy-style docstrings
- Wrap prose at Ruff's default line length; never wrap the summary line
- Reference Python methods with Sphinx RST cross-reference roles
- Cross-reference methods from other packages via Intersphinx (mappings in
  `docs/source/conf.py`)
