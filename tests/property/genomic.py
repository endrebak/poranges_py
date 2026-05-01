from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl
from hypothesis import strategies as st


@dataclass(frozen=True)
class GenomicColumns:
    chromosome: str = "chromosome"
    start: str = "start"
    end: str = "end"
    strand: str = "strand"
    gene: str = "gene"
    transcript_id: str = "transcript_id"


DEFAULT_CHROMOSOMES = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX", "chrY", "chrM")
DEFAULT_STRANDS = ("+", "-", ".")


def chromosome_names(
    *, min_chromosomes: int = 1, max_chromosomes: int = 8
) -> st.SearchStrategy[tuple[str, ...]]:
    """Generate a non-empty subset of chromosome names."""
    return st.lists(
        st.sampled_from(DEFAULT_CHROMOSOMES),
        min_size=min_chromosomes,
        max_size=max_chromosomes,
        unique=True,
    ).map(tuple)


def genomic_record(
    *,
    chromosomes: tuple[str, ...] = DEFAULT_CHROMOSOMES,
    max_position: int = 1_000_000,
    max_interval_width: int = 10_000,
    columns: GenomicColumns = GenomicColumns(),
    include_strand: bool = True,
    include_gene: bool = True,
    include_transcript_id: bool = True,
) -> st.SearchStrategy[dict[str, Any]]:
    """Generate one valid half-open genomic interval record."""
    if max_position < 1:
        raise ValueError("max_position must be at least 1")
    if max_interval_width < 1:
        raise ValueError("max_interval_width must be at least 1")
    if not chromosomes:
        raise ValueError("chromosomes must contain at least one name")

    @st.composite
    def _record(draw: st.DrawFn) -> dict[str, Any]:
        start = draw(st.integers(min_value=0, max_value=max_position - 1))
        max_width = min(max_interval_width, max_position - start)
        width = draw(st.integers(min_value=1, max_value=max_width))

        record: dict[str, Any] = {
            columns.chromosome: draw(st.sampled_from(chromosomes)),
            columns.start: start,
            columns.end: start + width,
        }
        if include_strand:
            record[columns.strand] = draw(st.sampled_from(DEFAULT_STRANDS))
        if include_gene:
            record[columns.gene] = draw(_identifier("gene"))
        if include_transcript_id:
            record[columns.transcript_id] = draw(_identifier("tx"))
        return record

    return _record()


def genomic_records(
    *,
    min_size: int = 0,
    max_size: int = 100,
    chromosomes: tuple[str, ...] = DEFAULT_CHROMOSOMES,
    max_position: int = 1_000_000,
    max_interval_width: int = 10_000,
    columns: GenomicColumns = GenomicColumns(),
    include_strand: bool = True,
    include_gene: bool = True,
    include_transcript_id: bool = True,
) -> st.SearchStrategy[list[dict[str, Any]]]:
    """Generate a list of genomic interval records."""
    return st.lists(
        genomic_record(
            chromosomes=chromosomes,
            max_position=max_position,
            max_interval_width=max_interval_width,
            columns=columns,
            include_strand=include_strand,
            include_gene=include_gene,
            include_transcript_id=include_transcript_id,
        ),
        min_size=min_size,
        max_size=max_size,
    )


def genomic_dataframes(
    *,
    min_size: int = 0,
    max_size: int = 100,
    chromosomes: tuple[str, ...] = DEFAULT_CHROMOSOMES,
    max_position: int = 1_000_000,
    max_interval_width: int = 10_000,
    columns: GenomicColumns = GenomicColumns(),
    include_strand: bool = True,
    include_gene: bool = True,
    include_transcript_id: bool = True,
) -> st.SearchStrategy[pl.DataFrame]:
    """Generate Polars DataFrames containing valid genomic intervals."""
    schema = _schema_for(
        columns=columns,
        include_strand=include_strand,
        include_gene=include_gene,
        include_transcript_id=include_transcript_id,
    )
    return genomic_records(
        min_size=min_size,
        max_size=max_size,
        chromosomes=chromosomes,
        max_position=max_position,
        max_interval_width=max_interval_width,
        columns=columns,
        include_strand=include_strand,
        include_gene=include_gene,
        include_transcript_id=include_transcript_id,
    ).map(lambda records: pl.DataFrame(records, schema=schema))


def _identifier(prefix: str) -> st.SearchStrategy[str]:
    return st.integers(min_value=1, max_value=100_000).map(
        lambda value: f"{prefix}_{value}"
    )


def _schema_for(
    *,
    columns: GenomicColumns,
    include_strand: bool,
    include_gene: bool,
    include_transcript_id: bool,
) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        columns.chromosome: pl.String,
        columns.start: pl.Int64,
        columns.end: pl.Int64,
    }
    if include_strand:
        schema[columns.strand] = pl.String
    if include_gene:
        schema[columns.gene] = pl.String
    if include_transcript_id:
        schema[columns.transcript_id] = pl.String
    return schema
