import polars as pl

import poranges as po


EXPECTED_TOP_LEVEL = {
    "benchmark_version",
    "overlap_pairs",
    "overlap_pairs_report",
    "overlap",
    "overlap_report",
    "merge_overlaps",
    "cluster_overlaps",
    "count_overlaps",
    "join_overlaps",
    "intersect_overlaps",
    "set_intersect_overlaps",
    "set_union_overlaps",
    "sort_ranges",
    "extend_ranges",
    "tile_ranges",
    "clip_ranges",
    "group_cumsum",
    "max_disjoint_overlaps",
    "split_overlaps",
    "outer_ranges",
    "complement_ranges",
    "nearest",
    "nearest_report",
    "subtract_overlaps",
    "bio_overlap",
    "bio_overlap_report",
    "bio_nearest",
    "bio_nearest_report",
    "bio_merge_overlaps",
    "bio_cluster_overlaps",
    "bio_count_overlaps",
    "bio_join_overlaps",
    "bio_intersect_overlaps",
    "bio_set_intersect_overlaps",
    "bio_set_union_overlaps",
    "bio_sort_ranges",
    "bio_extend_ranges",
    "bio_tile_ranges",
    "bio_clip_ranges",
    "bio_group_cumsum",
    "bio_max_disjoint_overlaps",
    "bio_split_overlaps",
    "bio_outer_ranges",
    "bio_complement_ranges",
    "bio_subtract_overlaps",
    "bio_has_valid_strand",
}

EXPECTED_R_METHODS = {
    "overlap_pairs",
    "overlap_pairs_report",
    "overlap",
    "overlap_report",
    "merge_overlaps",
    "cluster_overlaps",
    "count_overlaps",
    "join_overlaps",
    "intersect_overlaps",
    "set_intersect_overlaps",
    "set_union_overlaps",
    "sort_ranges",
    "extend_ranges",
    "tile_ranges",
    "clip_ranges",
    "group_cumsum",
    "max_disjoint_overlaps",
    "split_overlaps",
    "outer_ranges",
    "complement_ranges",
    "nearest",
    "nearest_report",
    "subtract_overlaps",
}

EXPECTED_G_METHODS = {
    "overlap",
    "overlap_report",
    "nearest",
    "nearest_report",
}

EXPECTED_BIO_METHODS = {
    "overlap",
    "overlap_report",
    "nearest",
    "nearest_report",
    "merge_overlaps",
    "cluster_overlaps",
    "count_overlaps",
    "join_overlaps",
    "intersect_overlaps",
    "set_intersect_overlaps",
    "set_union_overlaps",
    "sort_ranges",
    "extend_ranges",
    "tile_ranges",
    "clip_ranges",
    "group_cumsum",
    "max_disjoint_overlaps",
    "split_overlaps",
    "outer_ranges",
    "complement_ranges",
    "subtract_overlaps",
    "has_valid_strand",
}


def test_import_registers_public_namespaces_and_top_level_functions() -> None:
    assert EXPECTED_TOP_LEVEL <= set(po.__all__)
    assert po.benchmark_version()

    df = pl.DataFrame({"Chromosome": ["chr1"], "Start": [1], "End": [2], "Strand": ["+"]})
    lf = df.lazy()

    assert EXPECTED_R_METHODS <= set(dir(df.r))
    assert EXPECTED_G_METHODS <= set(dir(df.g))
    assert EXPECTED_BIO_METHODS <= set(dir(df.bio))

    assert EXPECTED_R_METHODS <= set(dir(lf.r))
    assert EXPECTED_G_METHODS <= set(dir(lf.g))
    assert EXPECTED_BIO_METHODS <= set(dir(lf.bio))


def test_top_level_overlap_and_range_namespaces_match_for_eager_and_lazy() -> None:
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

    expected = po.overlap(left, right, match_by="Chrom")
    via_dataframe = left.r.overlap(right, match_by="Chrom")
    via_lazy = left.lazy().r.overlap(right.lazy(), match_by="Chrom")

    assert expected.to_dicts() == via_dataframe.to_dicts()
    assert expected.to_dicts() == via_lazy.to_dicts()

    report_frame, timings = po.overlap_report(left, right, match_by="Chrom")
    assert report_frame.to_dicts() == expected.to_dicts()
    assert timings["total"] > 0


def test_top_level_nearest_and_range_namespaces_match_for_eager_and_lazy() -> None:
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

    expected = po.nearest(left, right, match_by="Chrom", direction="forward")
    via_dataframe = left.r.nearest(right, match_by="Chrom", direction="forward")
    via_lazy = left.lazy().r.nearest(right.lazy(), match_by="Chrom", direction="forward")

    assert expected.to_dicts() == via_dataframe.to_dicts()
    assert expected.to_dicts() == via_lazy.to_dicts()
    assert expected["Distance"].to_list() == [4, 6]

    report_frame, timings = po.nearest_report(
        left, right, match_by="Chrom", direction="forward"
    )
    assert report_frame.to_dicts() == expected.to_dicts()
    assert timings["kernel"] > 0


def test_count_overlaps_and_subtract_overlaps_forward_through_range_namespace() -> None:
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

    assert po.count_overlaps(left, right, match_by="Chrom").to_list() == [1, 1]
    assert left.r.count_overlaps(right, match_by="Chrom").to_list() == [1, 1]

    subtract = left.r.subtract_overlaps(right, match_by="Chrom")
    assert subtract.to_dicts() == [
        {"Chrom": "chr1", "Start": 1, "End": 2, "ID": "a"},
        {"Chrom": "chr1", "Start": 3, "End": 5, "ID": "a"},
        {"Chrom": "chr1", "Start": 1, "End": 2, "ID": "b"},
        {"Chrom": "chr1", "Start": 3, "End": 9, "ID": "b"},
    ]


def test_genomic_and_bio_namespaces_forward_to_rust() -> None:
    left = pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr2"],
            "Start": [1, 1, 10],
            "End": [3, 3, 11],
            "Strand": ["+", "+", "-"],
            "ID": ["A", "a", "b"],
        }
    )
    right = pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1"],
            "Start": [2, 2],
            "End": [3, 9],
            "Strand": ["+", "+"],
        }
    )

    top_level = po.bio_overlap(left, right, multiple=True)
    genomic = left.g.overlap(right, multiple=True)
    bio = left.bio.overlap(right, multiple=True)
    inverted = left.bio.overlap(right, invert=True)

    assert top_level["ID"].to_list() == ["A", "A", "a", "a"]
    assert top_level.to_dicts() == genomic.to_dicts()
    assert top_level.to_dicts() == bio.to_dicts()
    assert inverted["ID"].to_list() == ["b"]


def test_bio_namespace_exposes_broader_genomic_operations() -> None:
    df = pl.DataFrame(
        {
            "Chromosome": ["chr1", "chr1", "chr2"],
            "Start": [1, 4, 1],
            "End": [5, 8, 2],
            "Strand": ["+", "+", "-"],
        }
    )

    merged = po.bio_merge_overlaps(df)
    clustered = df.bio.cluster_overlaps()

    assert merged.to_dicts() == [
        {"Chromosome": "chr1", "Start": 1, "End": 8, "Strand": "+"},
        {"Chromosome": "chr2", "Start": 1, "End": 2, "Strand": "-"},
    ]
    assert clustered["Cluster"].to_list() == [0, 0, 2]
    assert df.bio.has_valid_strand() is True
