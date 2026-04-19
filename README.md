# poranges

Python bindings for `../poranges`, built with `maturin`.

The package exposes:

- a Rust extension for the interval operations implemented in `poranges`
- direct `polars.DataFrame` methods such as `df.overlap_ranges(other)`
- a `df.bio` namespace for `Chromosome` / `Strand` defaults such as `df.bio.overlap_ranges(other)`
