# AGENTS.md

## Project Overview

This repository hosts the code for revertex, a package part of the remage
ecosystem to run Geant4 simulations. This code in this package can be used to
generate vertices for primary particles in the simulation. These generators can
either generate positions, for example on specific surfaces, or generate
momenta, for example for beta decays or (alpha,n) reactions.

You can learn about the project from the markdown documentation source in
`docs/source/`:

- `index.md`: overview, key concepts
- `generators/alpha_n.md`: documentation for the alpha-n generator wrapper

## Dependencies & Tooling

Python package. See `pyproject.toml` for dependencies and build config.

## Common Commands

- Install (dev): `pip install -e ".[all]"` — run once after cloning or after
  dependency changes
- Test: `ptest` — run all tests before committing;
- Lint/format: `pre-commit run --all-files` — run before committing; every
  commit must pass
- Build docs: `cd docs && make` — verify after any documentation change

## Architecture

To understand the code structure, start with `src/revertex/cli.py`, which is the
entry point for the command line interface. This module shows how users should
interact with the package, and how the different generators are called.

To understand the implementation of the individual generators, look at the
modules under `src/revertex/generators/`.

- momenta generators:
  - `alpha_n.py` contains the implementation of the alpha-n generator wrapper.
  - `beta.py` contains the implementation of a beta decay generator.
- position generators:
  - `borehole.py` sampling on the borehole of a HPGe
  - `surface.py` sampling on the surface of a HPGe
  - `shell.py` sampling on the surface of a HPGe

Then, finally, look at `src/revertex/core.py`, `src/revertex/utils.py` and
`src/revertex/plot.py`. These modules contain helper functions for the
generators.

## Testing

See [`tests/AGENTS.md`](tests/AGENTS.md).

## Linting

- Formatting/linting is enforced by pre-commit, hooks are listed in
  `.pre-commit-config.yaml`, while further configuration is in `pyproject.toml`

## Documentation

See [`docs/AGENTS.md`](docs/AGENTS.md) for documentation conventions, docstring
style, and Snakemake rule docstring conventions.

## Code Conventions

See [`src/AGENTS.md`](src/AGENTS.md).

## Git Workflow

- Commit style: Conventional Commits (https://www.conventionalcommits.org)
- Every commit must pass linting and testing
- PRs require passing CI
- Make sure to include a concise description of the changes in a PR and link the
  relevant related PRs or issues

## Boundaries

- ALWAYS: make sure linting, testing and docs building succeed **before**
  committing
