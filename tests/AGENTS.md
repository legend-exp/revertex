# AGENTS.md — Testing

Python tests are stored in `tests/` and managed with Pytest. Run all tests with
`pytest`.

- `conftest.py`: fixtures to create mock configuration objects required to test
  package units
- `test_alpha_n.py`: tests for the alpha-n generator wrapper in
  `src/revertex/generators/alpha_n.py`. This includes tests for the wrapper's
  CLI interface, as well as tests for the wrapper's internal functions that
  generate the SaG4n input and parse the output.
- `test_beta.py`: tests for the beta decay generator in
  `src/revertex/generators/beta.py`.
- `test_borehole.py`: tests for the borehole position generator in
  `src/revertex/generators/borehole.py`.
- `test_surface.py`: tests for the surface position generator in
  `src/revertex/generators/surface.py`.
- `test_shell.py`: tests for the shell position generator in
  `src/revertex/generators/shell.py`.
- `test_utils.py`: tests for utility functions in `src/revertex/utils.py`.
- `test_core.py`: tests for core functions in `src/revertex/core.py`.
- `test_cli.py`: tests for command-line interface functions in
  `src/revertex/cli.py`.
