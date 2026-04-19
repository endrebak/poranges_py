import polars as pl

import poranges as pr


def test_rangeframe_merge_overlaps_matches_expected_projection() -> None:
    rf = pr.RangeFrame(
        {
            "Chrom": ["chr1", "chr1", "chr2"],
            "Start": [1, 4, 10],
            "End": [5, 8, 12],
            "Score": [1, 2, 3],
        }
    )

    result = rf.merge_overlaps(count_col="Count", match_by="Chrom")

    assert result.to_polars().to_dicts() == [
        {"Chrom": "chr1", "Start": 1, "End": 8, "Count": 2},
        {"Chrom": "chr2", "Start": 10, "End": 12, "Count": 1},
    ]


def test_rangeframe_cluster_overlaps_adds_cluster_column() -> None:
    rf = pr.RangeFrame(
        {
            "Chrom": ["chr1", "chr1", "chr1", "chr2"],
            "Start": [1, 3, 10, 1],
            "End": [4, 5, 12, 2],
        }
    )

    result = rf.cluster_overlaps(match_by="Chrom")

    assert result.to_polars()["Cluster"].to_list() == [0, 0, 1, 3]


def test_rangeframe_overlap_and_subtract_overlaps_work() -> None:
    left = pr.RangeFrame(
        {
            "Chrom": ["chr1", "chr1"],
            "Start": [1, 1],
            "End": [5, 9],
            "ID": ["a", "b"],
        }
    )
    right = pr.RangeFrame(
        {
            "Chrom": ["chr1"],
            "Start": [2],
            "End": [3],
        }
    )

    overlap = left.overlap(right, match_by="Chrom")
    subtract = left.subtract_overlaps(right, match_by="Chrom")

    assert overlap.to_polars()["ID"].to_list() == ["a", "b"]
    assert subtract.to_polars().to_dicts() == [
        {"Chrom": "chr1", "Start": 1, "End": 2, "ID": "a"},
        {"Chrom": "chr1", "Start": 3, "End": 5, "ID": "a"},
        {"Chrom": "chr1", "Start": 1, "End": 2, "ID": "b"},
        {"Chrom": "chr1", "Start": 3, "End": 9, "ID": "b"},
    ]


def test_rangeframe_nearest_ranges_appends_other_columns() -> None:
    left = pr.RangeFrame(
        {
            "Chrom": ["chr1", "chr1"],
            "Start": [1, 20],
            "End": [5, 25],
            "Name": ["a", "b"],
        }
    )
    right = pr.RangeFrame(
        {
            "Chrom": ["chr1", "chr1"],
            "Start": [8, 30],
            "End": [10, 35],
            "Name": ["x", "y"],
        }
    )

    result = left.nearest_ranges(right, match_by="Chrom", direction="forward")

    assert result.to_polars().columns == [
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
    assert result.to_polars()["Distance"].to_list() == [4, 6]


def test_pyranges_defaults_to_chromosome_matching() -> None:
    gr = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1", "chr2"],
            "Start": [1, 4, 1],
            "End": [5, 8, 2],
        }
    )

    result = gr.merge_overlaps()

    assert result.to_polars().to_dicts() == [
        {"Chromosome": "chr1", "Start": 1, "End": 8},
        {"Chromosome": "chr2", "Start": 1, "End": 2},
    ]


def test_pyranges_overlap_supports_multiple_and_invert() -> None:
    left = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1", "chr2"],
            "Start": [1, 1, 10],
            "End": [3, 3, 11],
            "ID": ["A", "a", "b"],
        }
    )
    right = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [2, 2],
            "End": [3, 9],
        }
    )

    multiple = left.overlap(right, multiple=True)
    inverted = left.overlap(right, invert=True)

    assert multiple.to_polars()["ID"].to_list() == ["A", "A", "a", "a"]
    assert inverted.to_polars()["ID"].to_list() == ["b"]


def test_pyranges_nearest_ranges_respects_strand_and_downstream_direction() -> None:
    left = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [10, 10],
            "End": [12, 12],
            "Strand": ["+", "-"],
            "Name": ["plus", "minus"],
        }
    )
    right = pr.PyRanges(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [20, 1],
            "End": [21, 2],
            "Strand": ["+", "-"],
            "Hit": ["plus_hit", "minus_hit"],
        }
    )

    result = left.nearest_ranges(right, direction="downstream")

    assert result.to_polars().sort("Name")["Hit_b"].to_list() == ["minus_hit", "plus_hit"]
