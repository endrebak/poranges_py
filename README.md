# poranges

Python bindings for `../poranges`, built with `maturin`.

The package exposes:

- a Rust extension for the interval operations implemented in `poranges`
- top-level helpers such as `poranges.overlap(df, other)` and `poranges.nearest(df, other)`
- Polars namespaces registered on import:
  - `df.r` / `lf.r` for range operations
  - `df.g` / `lf.g` for genomic overlap and nearest
  - `df.bio` / `lf.bio` for broader chromosome / strand-aware operations
