# AGENTS.md — Code Conventions

## Python

- License statement at the top of every Python file (see existing files)
- Follow
  [scientific Python conventions](https://learn.scientific-python.org/development)
- Always add type annotations for input arguments and outputs
- Use as generic a type as possible for arguments, and return as specific a type
  as possible. Use `Iterable`/`Sequence` or `Mapping` (from `collections.abc`)
  instead of `list` or `dict` for function arguments, if appropriate.
- Prefer `dbetto.AttrsDict` over plain `dict` for non-trivial dictionaries that
  are frequently queried (enables attribute-style access)
- The `.on()` method from `dbetto.TextDB()` involves filesystem queries and can
  be slow. Avoid using it repeatedly in functions invoked at Snakemake DAG build
  time; consider caching strategies instead
- Other conventions are enforced by pre-commit
