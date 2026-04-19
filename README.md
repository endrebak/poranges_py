# poranges

Python bindings for `../poranges`, built with `maturin`.

The package exposes:

- a Rust extension for the five interval operations currently implemented in `poranges`
- a `RangeFrame` wrapper around `polars.DataFrame`
- a `PyRanges` wrapper with `Chromosome` and optional `Strand` defaults matching the `pyrunges` API

