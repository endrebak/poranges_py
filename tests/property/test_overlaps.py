from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

po = pytest.importorskip("polaranges")
pb = pytest.importorskip("polars_bio")


CHROMOSOMES = ("chr1", "chr2", "chr3")
SMALL_GENES = ("gene_a", "gene_b", "gene_c")
LARGE_GENES = tuple(f"gene_{i}" for i in range(100))
STRANDS = ("+", "-", ".")
SCHEMA = {
    "chrom": pl.String,
    "Start": pl.Int64,
    "End": pl.Int64,
    "strand": pl.String,
    "gene": pl.String,
}


@dataclass(frozen=True)
class OverlapCase:
    name: str
    polaranges_kwargs: dict[str, Any]
    polars_bio_kwargs: dict[str, Any]
    strand_behavior: str = "ignore"
    match_by: tuple[str, ...] = ()


@pytest.fixture(
    scope="module",
    params=[
        OverlapCase(
            name="ignore_strand",
            polaranges_kwargs={"match_by": "chrom"},
            polars_bio_kwargs={
                "cols1": ["chrom", "Start", "End"],
                "cols2": ["chrom", "Start", "End"],
                "output_type": "polars.DataFrame",
            },
        ),
        OverlapCase(
            name="same_strand",
            polaranges_kwargs={"match_by": ["chrom", "strand"]},
            polars_bio_kwargs={
                "cols1": ["chrom", "Start", "End"],
                "cols2": ["chrom", "Start", "End"],
                "output_type": "polars.DataFrame",
            },
            strand_behavior="same",
        ),
        OverlapCase(
            name="match_by_gene",
            polaranges_kwargs={"match_by": ["chrom", "gene"]},
            polars_bio_kwargs={
                "cols1": ["chrom", "Start", "End"],
                "cols2": ["chrom", "Start", "End"],
                "output_type": "polars.DataFrame",
            },
            match_by=("gene",),
        ),
    ],
    ids=lambda case: case.name,
)
def overlap_case(request: pytest.FixtureRequest) -> OverlapCase:
    return request.param


@st.composite
def overlap_inputs(draw: st.DrawFn) -> tuple[pl.DataFrame, pl.DataFrame]:
    scenario = draw(
        st.sampled_from(
            (
                "guaranteed_overlap",
                "contained_overlap",
                "boundary_touch",
                "metadata_mismatch",
                "random",
            )
        )
    )

    left_records, right_records = _draw_anchor_records(draw, scenario)
    left_records.extend(
        _draw_random_record(draw) for _ in range(draw(st.integers(0, 6)))
    )
    right_records.extend(
        _draw_random_record(draw) for _ in range(draw(st.integers(0, 6)))
    )

    left = pl.DataFrame(left_records, schema=SCHEMA).with_row_index("left_id")
    right = pl.DataFrame(right_records, schema=SCHEMA).with_row_index("right_id")
    return left, right


@given(inputs=overlap_inputs())
@settings(deadline=None)
def test_overlap_pairs_match_polars_bio_for_simple_genomic_overlap(
    inputs: tuple[pl.DataFrame, pl.DataFrame],
    overlap_case: OverlapCase,
) -> None:
    left, right = inputs

    expected, polars_bio_overlaps = _polars_bio_overlap_pairs(
        left, right, overlap_case
    )
    observed = _polaranges_overlap_pairs(left, right, overlap_case)

    _print_overlap_example(
        case=overlap_case,
        left=left,
        right=right,
        polars_bio_overlaps=polars_bio_overlaps,
        expected=expected,
        observed=observed,
    )

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
) -> tuple[set[tuple[int, int]], pl.DataFrame]:
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

    if case.strand_behavior == "same":
        overlaps = overlaps.filter(pl.col("strand_1") == pl.col("strand_2"))
    elif case.strand_behavior != "ignore":
        raise ValueError(f"unsupported strand behavior: {case.strand_behavior}")

    for column in case.match_by:
        overlaps = overlaps.filter(pl.col(f"{column}_1") == pl.col(f"{column}_2"))

    return (
        set(zip(overlaps["left_id_1"], overlaps["right_id_2"], strict=True)),
        overlaps,
    )


def _draw_anchor_records(
    draw: st.DrawFn, scenario: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chrom = draw(st.sampled_from(CHROMOSOMES))
    strand = draw(st.sampled_from(STRANDS))
    gene = draw(st.sampled_from(SMALL_GENES))
    start = draw(st.integers(min_value=0, max_value=450))
    width = draw(st.integers(min_value=1, max_value=50))
    end = start + width

    if scenario == "guaranteed_overlap":
        right_start, right_end = _draw_overlapping_interval(draw, start, end)
        left = _record(chrom, start, end, strand, gene)
        right = _record(chrom, right_start, right_end, strand, gene)
    elif scenario == "contained_overlap":
        right_start = draw(st.integers(min_value=start, max_value=end - 1))
        right_end = draw(st.integers(min_value=right_start + 1, max_value=end))
        left = _record(chrom, start, end, strand, gene)
        right = _record(chrom, right_start, right_end, strand, gene)
    elif scenario == "boundary_touch":
        right_width = draw(st.integers(min_value=1, max_value=max(1, 500 - end)))
        left = _record(chrom, start, end, strand, gene)
        right = _record(chrom, end, end + right_width, strand, gene)
    elif scenario == "metadata_mismatch":
        right_start, right_end = _draw_overlapping_interval(draw, start, end)
        mismatch = draw(st.sampled_from(("gene", "strand")))
        right_gene = gene
        right_strand = strand
        if mismatch == "gene":
            right_gene = draw(
                st.sampled_from(tuple(g for g in SMALL_GENES if g != gene))
            )
        else:
            right_strand = draw(
                st.sampled_from(tuple(s for s in STRANDS if s != strand))
            )
        left = _record(chrom, start, end, strand, gene)
        right = _record(chrom, right_start, right_end, right_strand, right_gene)
    elif scenario == "random":
        left = _draw_random_record(draw)
        right = _draw_random_record(draw)
    else:
        raise ValueError(f"unsupported overlap input scenario: {scenario}")

    return [left], [right]


def _draw_overlapping_interval(
    draw: st.DrawFn, start: int, end: int
) -> tuple[int, int]:
    overlap_point = draw(st.integers(min_value=start, max_value=end - 1))
    right_start = draw(
        st.integers(min_value=max(0, overlap_point - 49), max_value=overlap_point)
    )
    right_end = draw(
        st.integers(min_value=overlap_point + 1, max_value=min(500, right_start + 50))
    )
    return right_start, right_end


def _draw_random_record(draw: st.DrawFn) -> dict[str, Any]:
    start = draw(st.integers(min_value=0, max_value=499))
    end = draw(st.integers(min_value=start + 1, max_value=min(500, start + 50)))
    return _record(
        chrom=draw(st.sampled_from(CHROMOSOMES)),
        start=start,
        end=end,
        strand=draw(st.sampled_from(STRANDS)),
        gene=draw(
            st.one_of(st.sampled_from(SMALL_GENES), st.sampled_from(LARGE_GENES))
        ),
    )


def _record(
    chrom: str,
    start: int,
    end: int,
    strand: str,
    gene: str,
) -> dict[str, Any]:
    return {
        "chrom": chrom,
        "Start": start,
        "End": end,
        "strand": strand,
        "gene": gene,
    }


def _print_overlap_example(
    *,
    case: OverlapCase,
    left: pl.DataFrame,
    right: pl.DataFrame,
    polars_bio_overlaps: pl.DataFrame,
    expected: set[tuple[int, int]],
    observed: set[tuple[int, int]],
) -> None:
    print(f"\n--- overlap case: {case.name} ---")
    print("left input:")
    print(left)
    print("right input:")
    print(right)
    print("polars-bio overlap output:")
    print(polars_bio_overlaps)
    print(f"expected pairs from polars-bio: {sorted(expected)}")
    print(f"observed pairs from polaranges: {sorted(observed)}")
    print(f"result: {'PASS' if observed == expected else 'FAIL'}")
