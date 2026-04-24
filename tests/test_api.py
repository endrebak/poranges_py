import polars as pl

import poranges  # noqa: F401


def test_dataframe_merge_overlaps_matches_expected_projection() -> None:
    df = pl.DataFrame(
        {
            "Chrom": ["chr1", "chr1", "chr2"],
            "Start": [1, 4, 10],
            "End": [5, 8, 12],
            "Score": [1, 2, 3],
        }
    )

    result = df.merge_overlaps(count_col="Count", match_by="Chrom")

    assert result.to_dicts() == [
        {"Chrom": "chr1", "Start": 1, "End": 8, "Count": 2},
        {"Chrom": "chr2", "Start": 10, "End": 12, "Count": 1},
    ]


def test_dataframe_cluster_overlaps_adds_cluster_column() -> None:
    df = pl.DataFrame(
        {
            "Chrom": ["chr1", "chr1", "chr1", "chr2"],
            "Start": [1, 3, 10, 1],
            "End": [4, 5, 12, 2],
        }
    )

    result = df.cluster_overlaps(match_by="Chrom")

    assert result["Cluster"].to_list() == [0, 0, 1, 3]


def test_dataframe_overlap_and_subtract_overlaps_work() -> None:
    left = pl.DataFrame(
        {
            "Chrom": ["chr1", "chr1"],
            "Start": [1, 1],
            "End": [5, 9],
            "ID": ["a", "b"],
        }
    )
    right = pl.DataFrame(
        {
            "Chrom": ["chr1"],
            "Start": [2],
            "End": [3],
        }
    )

    overlap = left.overlap_ranges(right, match_by="Chrom")
    subtract = left.subtract_overlaps(right, match_by="Chrom")

    assert overlap["ID"].to_list() == ["a", "b"]
    assert subtract.to_dicts() == [
        {"Chrom": "chr1", "Start": 1, "End": 2, "ID": "a"},
        {"Chrom": "chr1", "Start": 3, "End": 5, "ID": "a"},
        {"Chrom": "chr1", "Start": 1, "End": 2, "ID": "b"},
        {"Chrom": "chr1", "Start": 3, "End": 9, "ID": "b"},
    ]


def test_dataframe_nearest_ranges_appends_other_columns() -> None:
    left = pl.DataFrame(
        {
            "Chrom": ["chr1", "chr1"],
            "Start": [1, 20],
            "End": [5, 25],
            "Name": ["a", "b"],
        }
    )
    right = pl.DataFrame(
        {
            "Chrom": ["chr1", "chr1"],
            "Start": [8, 30],
            "End": [10, 35],
            "Name": ["x", "y"],
        }
    )

    result = left.nearest_ranges(right, match_by="Chrom", direction="forward")

    assert result.columns == [
        "Chrom",
        "Start",
        "End",
        "Name",
        "Chrom_b",
        "Start_b",
        "End_b",
        "Name_b",
        "Distance",
    ]
    assert result["Distance"].to_list() == [4, 6]


def test_bio_merge_defaults_to_chromosome_matching() -> None:
    df = pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr2"],
            "Start": [1, 4, 1],
            "End": [5, 8, 2],
        }
    )

    result = df.bio.merge_overlaps()

    assert result.to_dicts() == [
        {"Chromosome": "chr1", "Start": 1, "End": 8},
        {"Chromosome": "chr2", "Start": 1, "End": 2},
    ]


def test_bio_overlap_supports_multiple_and_invert() -> None:
    left = pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr2"],
            "Start": [1, 1, 10],
            "End": [3, 3, 11],
            "ID": ["A", "a", "b"],
        }
    )
    right = pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [2, 2],
            "End": [3, 9],
        }
    )

    multiple = left.bio.overlap_ranges(right, multiple=True)
    inverted = left.bio.overlap_ranges(right, invert=True)

    assert multiple["ID"].to_list() == ["A", "A", "a", "a"]
    assert inverted["ID"].to_list() == ["b"]


def test_bio_nearest_ranges_respects_strand_and_downstream_direction() -> None:
    left = pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [10, 10],
            "End": [12, 12],
            "Strand": ["+", "-"],
            "Name": ["plus", "minus"],
        }
    )
    right = pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [20, 1],
            "End": [21, 2],
            "Strand": ["+", "-"],
            "Hit": ["plus_hit", "minus_hit"],
        }
    )

    result = left.bio.nearest_ranges(right, direction="downstream")

    assert result.sort("Name")["Hit_b"].to_list() == ["minus_hit", "plus_hit"]


def test_bio_has_valid_strand_uses_dataframe_namespace() -> None:
    df = pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [1, 2],
            "End": [3, 4],
            "Strand": ["+", "-"],
        }
    )

    assert df.bio.has_valid_strand() is True


def test_overlap_ranges_report_returns_dataframe_and_structured_timings() -> None:
    left = pl.DataFrame(
        {
            "Chrom": ["chr1", "chr1"],
            "Start": [1, 10],
            "End": [5, 15],
            "ID": ["a", "b"],
        }
    )
    right = pl.DataFrame(
        {
            "Chrom": ["chr1", "chr1"],
            "Start": [2, 12],
            "End": [4, 14],
        }
    )

    report = poranges.overlap_ranges_report(left, right, match_by="Chrom")

    assert isinstance(report.result, pl.DataFrame)
    assert report.result.to_dicts() == poranges.overlap_ranges(
        left, right, match_by="Chrom"
    ).to_dicts()
    assert report.timings.total > 0
    assert report.timings.kernel > 0
    assert "total" in report.timings.as_dict()
    assert "total" in report.timings.as_dict(milliseconds=True)


def test_parallel_config_and_dataframe_report_method_roundtrip() -> None:
    left = pl.DataFrame(
        {
            "Chrom": [f"chr{i}" for i in range(64) for _ in range(4)],
            "Start": [i * 10 for i in range(256)],
            "End": [i * 10 + 5 for i in range(256)],
        }
    )
    right = pl.DataFrame(
        {
            "Chrom": [f"chr{i}" for i in range(64) for _ in range(4)],
            "Start": [i * 10 + 1 for i in range(256)],
            "End": [i * 10 + 4 for i in range(256)],
        }
    )

    report = left.overlap_ranges_report(
        right, match_by="Chrom", parallel=poranges.ParallelConfig.auto()
    )

    assert isinstance(report, poranges.OperationReport)
    assert report.result.height == left.height
    assert report.timings.total > 0


def test_lazy_report_dispatch_returns_lazyframe_and_timings() -> None:
    left = pl.DataFrame(
        {
            "Chrom": ["chr1", "chr1"],
            "Start": [1, 20],
            "End": [5, 25],
            "Name": ["a", "b"],
        }
    ).lazy()
    right = pl.DataFrame(
        {
            "Chrom": ["chr1", "chr1"],
            "Start": [8, 30],
            "End": [10, 35],
            "Name": ["x", "y"],
        }
    ).lazy()

    report = poranges.nearest_ranges_report(
        left,
        right,
        match_by="Chrom",
        direction="forward",
        parallel="serial",
    )

    assert isinstance(report.result, pl.LazyFrame)
    collected = report.result.collect()
    expected = poranges.nearest_ranges(
        left.collect(),
        right.collect(),
        match_by="Chrom",
        direction="forward",
        parallel="serial",
    )
    assert collected.to_dicts() == expected.to_dicts()
    assert report.timings.total > 0


def test_range_and_genomic_namespace_aliases_work_for_eager_and_lazy() -> None:
    left = pl.DataFrame(
        {
            "Chromosome": ["chr1"],
            "Start": [10],
            "End": [12],
            "Strand": ["+"],
            "Name": ["plus"],
        }
    )
    right = pl.DataFrame(
        {
            "Chromosome": ["chr1"],
            "Start": [20],
            "End": [21],
            "Strand": ["+"],
            "Hit": ["plus_hit"],
        }
    )

    eager_report = left.g.nearest_ranges_report(right, direction="downstream")
    expected_eager = left.g.nearest_ranges(right, direction="downstream")
    lazy_report = left.lazy().r.overlap_ranges_report(
        right.lazy(),
        match_by="Chromosome",
    )

    assert eager_report.result.to_dicts() == expected_eager.to_dicts()
    assert isinstance(lazy_report.result, pl.LazyFrame)
    assert lazy_report.result.collect().height == 0
