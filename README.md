# polaranges

Python bindings for `../polaranges`, built with `maturin`.

## Local benchmark install

Check out the private Rust crate and the Python bindings as sibling
directories:

```text
parent-dir/
  polaranges/
  polaranges_py/
```

Then run the local installer from the `polaranges_py` checkout:

```bash
./scripts/install-local.sh
```

The installer creates or reuses `.venv`, installs `maturin`, builds the Python
extension against the local `../polaranges` checkout, runs an import smoke test,
and then runs the test suite.

To install into the same Python environment used by a benchmark runner, pass
that interpreter explicitly:

```bash
./scripts/install-local.sh --python /path/to/benchmark/python
```

Or, after activating the benchmark environment:

```bash
./scripts/install-local.sh --current-python
```

For a quicker install without pytest:

```bash
./scripts/install-local.sh --skip-tests
```

The package exposes:

- a Rust extension for the interval operations implemented in `polaranges`
- top-level helpers such as `polaranges.overlap(df, other)` and `polaranges.nearest(df, other)`
- Polars namespaces registered on import:
  - `df.r` / `lf.r` for range operations
  - `df.g` / `lf.g` for genomic overlap and nearest
  - `df.bio` / `lf.bio` for broader chromosome / strand-aware operations
