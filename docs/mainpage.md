# igneous C++ API Docs

This site is generated from Doxygen comments in the C++ headers.

## Scope

- Public API in `include/igneous/`
- Internal/private implementation notes where available
- Concepts, structures, spaces, and operators

## Build

From the repo root:

```bash
cmake -S . -B build -G Ninja
cmake --build build --target docs
```

The generated HTML docs are written to:

- `build/docs/html/index.html`

## Style in This Repo

- Use `///` or `/** ... */` comments for declarations.
- Document templates/concepts, workspaces, and helper functions.
- Include private/internal field docs for complex stateful types.
