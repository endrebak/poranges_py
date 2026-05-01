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
STRANDS = ("+", "-", ".")
SCHEMA = {
    "chrom": pl.String,
    "Start": pl.Int64,
    "End": pl.Int64,
    "strand": pl.String,
    "gene": pl.String,
}


@dataclass(frozen=True)
class NearestCase:
    name: str
    polaranges_kwargs: dict[str, Any]
    polars_bio_kwargs: dict[str, Any]
    strand_behavior: str = "ignore"
    match_by: tuple[str, ...] = ()


@pytest.fixture(
    scope="module",
    params=[
        NearestCase(
            name="ignore_strand",
            polaranges_kwargs={"match_by": "chrom"},
            polars_bio_kwargs={
                "cols1": ["chrom", "Start", "End"],
                "cols2": ["chrom", "Start", "End"],
                "output_type": "polars.DataFrame",
            },
        ),
        NearestCase(
            name="same_strand",
            polaranges_kwargs={"match_by": ["chrom", "strand"]},
            polars_bio_kwargs={
                "cols1": ["chrom", "Start", "End"],
                "cols2": ["chrom", "Start", "End"],
                "output_type": "polars.DataFrame",
            },
            strand_behavior="same",
        ),
        NearestCase(
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
def nearest_case(request: pytest.FixtureRequest) -> NearestCase:
    return request.param


@st.composite
def nearest_inputs(draw: st.DrawFn) -> tuple[pl.DataFrame, pl.DataFrame]:
    scenario = draw(st.sampled_from(("before", "after", "overlap")))
    chrom = draw(st.sampled_from(CHROMOSOMES))
    noise_chrom = draw(st.sampled_from(tuple(c for c in CHROMOSOMES if c != chrom)))
    strand = draw(st.sampled_from(STRANDS))
    gene = draw(st.sampled_from(SMALL_GENES))
    start = draw(st.integers(min_value=100, max_value=350))
    width = draw(st.integers(min_value=5, max_value=50))
    end = start + width

    left_records = [_record(chrom, start, end, strand, gene)]
    right_records = [_nearest_anchor_right(draw, scenario, chrom, start, end, strand, gene)]

    left_records.extend(
        _draw_noise_record(draw, noise_chrom) for _ in range(draw(st.integers(0, 3)))
    )
    right_records.extend(
        _draw_noise_record(draw, noise_chrom) for _ in range(draw(st.integers(0, 3)))
    )

    left = pl.DataFrame(left_records, schema=SCHEMA).with_row_index("left_id")
    right = pl.DataFrame(right_records, schema=SCHEMA).with_row_index("right_id")
    return left, right


@given(inputs=nearest_inputs())
@settings(deadline=None)
def test_nearest_pairs_match_polars_bio(
    inputs: tuple[pl.DataFrame, pl.DataFrame],
    nearest_case: NearestCase,
) -> None:
    left, right = inputs

    expected, polars_bio_nearest = _polars_bio_nearest_pairs(left, right, nearest_case)
    observed, polaranges_nearest = _polaranges_nearest_pairs(left, right, nearest_case)

    _print_nearest_example(
        case=nearest_case,
        left=left,
        right=right,
        polars_bio_nearest=polars_bio_nearest,
        polaranges_nearest=polaranges_nearest,
        expected=expected,
        observed=observed,
    )

    assert observed == expected


def _polaranges_nearest_pairs(
    left: pl.DataFrame, right: pl.DataFrame, case: NearestCase
) -> tuple[set[tuple[int, int]], pl.DataFrame]:
    nearest = po.nearest(left, right, **case.polaranges_kwargs)
    return set(zip(nearest["left_id"], nearest["right_id_b"], strict=True)), nearest


def _polars_bio_nearest_pairs(
    left: pl.DataFrame, right: pl.DataFrame, case: NearestCase
) -> tuple[set[tuple[int, int]], pl.DataFrame]:
    nearest_frames: list[pl.DataFrame] = []
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
            for row in left.iter_rows(named=True):
                candidate_right = _candidate_right_rows(right, row, case)
                if candidate_right.is_empty():
                    continue

                nearest = pb.nearest(
                    pl.DataFrame([row], schema=left.schema),
                    candidate_right,
                    **case.polars_bio_kwargs,
                )
                if isinstance(nearest, pl.LazyFrame):
                    nearest = nearest.collect()
                nearest = _patch_polars_bio_null_nearest(
                    nearest=nearest,
                    left_row=row,
                    candidate_right=candidate_right,
                )
                nearest_frames.append(nearest)
    finally:
        pb.set_option(
            pb.POLARS_BIO_COORDINATE_SYSTEM_ZERO_BASED,
            previous_coordinate_setting,
        )

    if not nearest_frames:
        empty = pl.DataFrame()
        return set(), empty

    nearest = pl.concat(nearest_frames, how="vertical")
    return set(zip(nearest["left_id_1"], nearest["right_id_2"], strict=True)), nearest


def _patch_polars_bio_null_nearest(
    *,
    nearest: pl.DataFrame,
    left_row: dict[str, Any],
    candidate_right: pl.DataFrame,
) -> pl.DataFrame:
    if nearest.is_empty() or "right_id_2" not in nearest.columns:
        return nearest
    if not nearest["right_id_2"].is_null().any():
        return nearest

    # polars-bio 0.29 returns null nearest rows for zero-based single-base
    # intervals. See docs/polars-bio-differences.md.
    replacement = _nearest_candidate_by_interval_distance(left_row, candidate_right)
    if replacement is None:
        return nearest

    patched = nearest.to_dicts()
    patched[0].update(
        {
            "right_id_2": replacement["right_id"],
            "chrom_2": replacement["chrom"],
            "Start_2": replacement["Start"],
            "End_2": replacement["End"],
            "strand_2": replacement["strand"],
            "gene_2": replacement["gene"],
            "distance": _interval_distance(left_row, replacement),
        }
    )
    return pl.DataFrame(patched, schema=nearest.schema)


def _nearest_candidate_by_interval_distance(
    left_row: dict[str, Any], candidate_right: pl.DataFrame
) -> dict[str, Any] | None:
    candidates = candidate_right.to_dicts()
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            _interval_distance(left_row, row),
            row["right_id"],
        ),
    )


def _interval_distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    if left["Start"] < right["End"] and right["Start"] < left["End"]:
        return 0
    if left["End"] <= right["Start"]:
        return right["Start"] - left["End"]
    return left["Start"] - right["End"]


def _candidate_right_rows(
    right: pl.DataFrame, left_row: dict[str, Any], case: NearestCase
) -> pl.DataFrame:
    predicates = [pl.col("chrom") == left_row["chrom"]]
    if case.strand_behavior == "same":
        predicates.append(pl.col("strand") == left_row["strand"])
    elif case.strand_behavior != "ignore":
        raise ValueError(f"unsupported strand behavior: {case.strand_behavior}")
    for column in case.match_by:
        predicates.append(pl.col(column) == left_row[column])

    return right.filter(*predicates)


def _nearest_anchor_right(
    draw: st.DrawFn,
    scenario: str,
    chrom: str,
    start: int,
    end: int,
    strand: str,
    gene: str,
) -> dict[str, Any]:
    if scenario == "before":
        right_end = draw(st.integers(min_value=max(1, start - 60), max_value=start))
        right_width = draw(st.integers(min_value=1, max_value=min(50, right_end)))
        return _record(chrom, right_end - right_width, right_end, strand, gene)
    if scenario == "after":
        right_start = draw(st.integers(min_value=end, max_value=min(500, end + 60)))
        right_width = draw(st.integers(min_value=1, max_value=min(50, 500 - right_start)))
        return _record(chrom, right_start, right_start + right_width, strand, gene)
    if scenario == "overlap":
        overlap_point = draw(st.integers(min_value=start, max_value=end - 1))
        right_start = draw(
            st.integers(min_value=max(0, overlap_point - 49), max_value=overlap_point)
        )
        right_end = draw(
            st.integers(
                min_value=overlap_point + 1,
                max_value=min(500, right_start + 50),
            )
        )
        return _record(chrom, right_start, right_end, strand, gene)
    raise ValueError(f"unsupported nearest scenario: {scenario}")


def _draw_noise_record(draw: st.DrawFn, chrom: str) -> dict[str, Any]:
    start = draw(st.integers(min_value=0, max_value=499))
    end = draw(st.integers(min_value=start + 1, max_value=min(500, start + 50)))
    return _record(
        chrom=chrom,
        start=start,
        end=end,
        strand=draw(st.sampled_from(STRANDS)),
        gene=draw(st.sampled_from(SMALL_GENES)),
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


def _print_nearest_example(
    *,
    case: NearestCase,
    left: pl.DataFrame,
    right: pl.DataFrame,
    polars_bio_nearest: pl.DataFrame,
    polaranges_nearest: pl.DataFrame,
    expected: set[tuple[int, int]],
    observed: set[tuple[int, int]],
) -> None:
    print(f"\n--- nearest case: {case.name} ---")
    print("left input:")
    print(left)
    print("right input:")
    print(right)
    print("polars-bio nearest output:")
    print(polars_bio_nearest)
    print("polaranges nearest output:")
    print(polaranges_nearest)
    print(f"expected pairs from polars-bio: {sorted(expected)}")
    print(f"observed pairs from polaranges: {sorted(observed)}")
    print(f"result: {'PASS' if observed == expected else 'FAIL'}")
