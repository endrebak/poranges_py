from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from .genomic import GenomicColumns, genomic_dataframes

po = pytest.importorskip("polaranges")
pb = pytest.importorskip("polars_bio")


RANGE_COLUMNS = GenomicColumns(
    chromosome="chrom",
    start="start",
    end="end",
    strand="strand",
    gene="gene",
    transcript_id="transcript_id",
)


@dataclass(frozen=True)
class OverlapCase:
    name: str
    polaranges_kwargs: dict[str, Any]
    polars_bio_kwargs: dict[str, Any]


@pytest.fixture(scope="module")
def simple_overlap_case() -> OverlapCase:
    return OverlapCase(
        name="simple",
        polaranges_kwargs={"match_by": "chrom"},
        polars_bio_kwargs={
            "cols1": ["chrom", "start", "end"],
            "cols2": ["chrom", "start", "end"],
            "output_type": "polars.DataFrame",
        },
    )


@st.composite
def overlap_inputs(draw: st.DrawFn) -> tuple[pl.DataFrame, pl.DataFrame]:
    dataframe = genomic_dataframes(
        min_size=1,
        max_size=20,
        chromosomes=("chr1", "chr2", "chr3"),
        max_position=500,
        max_interval_width=50,
        columns=RANGE_COLUMNS,
        include_strand=False,
        include_gene=False,
        include_transcript_id=False,
    )
    left = draw(dataframe).with_row_index("left_id")
    right = draw(dataframe).with_row_index("right_id")
    return left, right


@given(overlap_inputs())
@settings(max_examples=50, deadline=None)
def test_overlap_pairs_match_polars_bio_for_simple_genomic_overlap(
    inputs: tuple[pl.DataFrame, pl.DataFrame],
    simple_overlap_case: OverlapCase,
) -> None:
    left, right = inputs

    expected = _polars_bio_overlap_pairs(left, right, simple_overlap_case)
    observed = _polaranges_overlap_pairs(left, right, simple_overlap_case)

    assert observed == expected


def _polaranges_overlap_pairs(
    left: pl.DataFrame, right: pl.DataFrame, case: OverlapCase
) -> set[tuple[int, int]]:
    left_indices, right_indices = po.overlap_pairs(
        left, right, **case.polaranges_kwargs
    )
    return set(zip(left_indices, right_indices, strict=True))


def _polars_bio_overlap_pairs(
    left: pl.DataFrame, right: pl.DataFrame, case: OverlapCase
) -> set[tuple[int, int]]:
    previous_coordinate_setting = pb.get_option(
        pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED
    )
    try:
        pb.set_option(pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED, "true")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Coordinate system metadata is missing.*",
                category=UserWarning,
            )
            overlaps = pb.overlap(left, right, **case.polars_bio_kwargs)
    finally:
        pb.set_option(
            pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED,
            previous_coordinate_setting,
        )

    if isinstance(overlaps, pl.LazyFrame):
        overlaps = overlaps.collect()

    return set(zip(overlaps["left_id_1"], overlaps["right_id_2"], strict=True))
