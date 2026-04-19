from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from . import _poranges

JOIN_SUFFIX = "_b"


def _arg_to_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_match_by(value: str | Iterable[str] | None) -> list[str] | None:
    normalized = _arg_to_list(value)
    return normalized or None


def _coerce_df(data: Any) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        return data
    return pl.DataFrame(data)


def _normalize_use_strand(value: str | bool) -> str:
    if value == "auto":
        return "auto"
    if value is True:
        return "enabled"
    if value is False:
        return "disabled"
    if value in {"enabled", "disabled"}:
        return value

    msg = "use_strand must be one of 'auto', True, or False"
    raise ValueError(msg)


def _normalize_bio_multiple(value: str | bool) -> str:
    if value is True:
        return "all"
    if value is False:
        return "first"
    if value in {"all", "first", "last"}:
        return value

    msg = "multiple must be one of True, False, 'all', 'first', or 'last'"
    raise ValueError(msg)


def overlap_range_pairs(
    left: pl.DataFrame | Any,
    right: pl.DataFrame | Any,
    multiple: str = "all",
    slack: int = 0,
    *,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
) -> tuple[list[int], list[int]]:
    left_idx, right_idx = _poranges.overlap_range_pairs(
        _coerce_df(left),
        _coerce_df(right),
        multiple,
        slack,
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
    )
    return list(left_idx), list(right_idx)


def overlap_ranges(
    left: pl.DataFrame | Any,
    right: pl.DataFrame | Any,
    multiple: str = "all",
    slack: int = 0,
    *,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
) -> pl.DataFrame:
    return _poranges.overlap_ranges(
        _coerce_df(left),
        _coerce_df(right),
        multiple,
        slack,
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
    )


def merge_overlaps(
    df: pl.DataFrame | Any,
    *,
    count_col: str | None = None,
    match_by: str | Iterable[str] | None = None,
    slack: int = 0,
) -> pl.DataFrame:
    return _poranges.merge_overlaps(
        _coerce_df(df),
        count_col=count_col,
        match_by=_normalize_match_by(match_by),
        slack=slack,
    )


def cluster_overlaps(
    df: pl.DataFrame | Any,
    *,
    match_by: str | Iterable[str] | None = None,
    cluster_column: str = "Cluster",
    slack: int = 0,
) -> pl.DataFrame:
    return _poranges.cluster_overlaps(
        _coerce_df(df),
        match_by=_normalize_match_by(match_by),
        cluster_column=cluster_column,
        slack=slack,
    )


def nearest_ranges(
    left: pl.DataFrame | Any,
    right: pl.DataFrame | Any,
    *,
    match_by: str | Iterable[str] | None = None,
    suffix: str = JOIN_SUFFIX,
    exclude_overlaps: bool = False,
    k: int = 1,
    dist_col: str | None = "Distance",
    direction: str = "any",
    preserve_input_order: bool = True,
) -> pl.DataFrame:
    return _poranges.nearest_ranges(
        _coerce_df(left),
        _coerce_df(right),
        match_by=_normalize_match_by(match_by),
        suffix=suffix,
        exclude_overlaps=exclude_overlaps,
        k=k,
        dist_col=dist_col,
        direction=direction,
        preserve_input_order=preserve_input_order,
    )


def subtract_overlaps(
    left: pl.DataFrame | Any,
    right: pl.DataFrame | Any,
    *,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
) -> pl.DataFrame:
    return _poranges.subtract_overlaps(
        _coerce_df(left),
        _coerce_df(right),
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
    )


def bio_merge_overlaps(
    df: pl.DataFrame | Any,
    use_strand: str | bool = "auto",
    *,
    count_col: str | None = None,
    match_by: str | Iterable[str] | None = None,
    slack: int = 0,
) -> pl.DataFrame:
    return _poranges.bio_merge_overlaps(
        _coerce_df(df),
        use_strand=_normalize_use_strand(use_strand),
        count_col=count_col,
        match_by=_normalize_match_by(match_by),
        slack=slack,
    )


def bio_cluster_overlaps(
    df: pl.DataFrame | Any,
    use_strand: str | bool = "auto",
    *,
    match_by: str | Iterable[str] | None = None,
    cluster_column: str = "Cluster",
    slack: int = 0,
) -> pl.DataFrame:
    return _poranges.bio_cluster_overlaps(
        _coerce_df(df),
        use_strand=_normalize_use_strand(use_strand),
        match_by=_normalize_match_by(match_by),
        cluster_column=cluster_column,
        slack=slack,
    )


def bio_overlap_ranges(
    left: pl.DataFrame | Any,
    right: pl.DataFrame | Any,
    strand_behavior: str = "auto",
    slack: int = 0,
    *,
    multiple: str | bool = False,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    invert: bool = False,
    preserve_input_order: bool = True,
) -> pl.DataFrame:
    return _poranges.bio_overlap_ranges(
        _coerce_df(left),
        _coerce_df(right),
        _normalize_bio_multiple(multiple),
        slack,
        strand_behavior=strand_behavior,
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        invert=invert,
        preserve_input_order=preserve_input_order,
    )


def bio_nearest_ranges(
    left: pl.DataFrame | Any,
    right: pl.DataFrame | Any,
    strand_behavior: str = "auto",
    direction: str = "any",
    *,
    k: int = 1,
    match_by: str | Iterable[str] | None = None,
    suffix: str = JOIN_SUFFIX,
    exclude_overlaps: bool = False,
    dist_col: str | None = "Distance",
    preserve_input_order: bool = True,
) -> pl.DataFrame:
    return _poranges.bio_nearest_ranges(
        _coerce_df(left),
        _coerce_df(right),
        strand_behavior=strand_behavior,
        match_by=_normalize_match_by(match_by),
        suffix=suffix,
        exclude_overlaps=exclude_overlaps,
        k=k,
        dist_col=dist_col,
        direction=direction,
        preserve_input_order=preserve_input_order,
    )


def bio_subtract_overlaps(
    left: pl.DataFrame | Any,
    right: pl.DataFrame | Any,
    strand_behavior: str = "auto",
    *,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
) -> pl.DataFrame:
    return _poranges.bio_subtract_overlaps(
        _coerce_df(left),
        _coerce_df(right),
        strand_behavior=strand_behavior,
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
    )


def bio_has_valid_strand(df: pl.DataFrame | Any, *, strand_col: str = "Strand") -> bool:
    return _poranges.bio_has_valid_strand(_coerce_df(df), strand_col=strand_col)


class _BioNamespace:
    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def has_valid_strand(self, *, strand_col: str = "Strand") -> bool:
        return bio_has_valid_strand(self._df, strand_col=strand_col)

    def merge_overlaps(
        self,
        use_strand: str | bool = "auto",
        *,
        count_col: str | None = None,
        match_by: str | Iterable[str] | None = None,
        slack: int = 0,
    ) -> pl.DataFrame:
        return bio_merge_overlaps(
            self._df,
            use_strand=use_strand,
            count_col=count_col,
            match_by=match_by,
            slack=slack,
        )

    def cluster_overlaps(
        self,
        use_strand: str | bool = "auto",
        *,
        match_by: str | Iterable[str] | None = None,
        cluster_column: str = "Cluster",
        slack: int = 0,
    ) -> pl.DataFrame:
        return bio_cluster_overlaps(
            self._df,
            use_strand=use_strand,
            match_by=match_by,
            cluster_column=cluster_column,
            slack=slack,
        )

    def overlap_ranges(
        self,
        other: pl.DataFrame | Any,
        strand_behavior: str = "auto",
        slack: int = 0,
        *,
        multiple: str | bool = False,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        invert: bool = False,
        preserve_input_order: bool = True,
    ) -> pl.DataFrame:
        return bio_overlap_ranges(
            self._df,
            other,
            strand_behavior=strand_behavior,
            slack=slack,
            multiple=multiple,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            invert=invert,
            preserve_input_order=preserve_input_order,
        )

    def nearest_ranges(
        self,
        other: pl.DataFrame | Any,
        strand_behavior: str = "auto",
        direction: str = "any",
        *,
        k: int = 1,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        dist_col: str | None = "Distance",
        preserve_input_order: bool = True,
    ) -> pl.DataFrame:
        return bio_nearest_ranges(
            self._df,
            other,
            strand_behavior=strand_behavior,
            direction=direction,
            k=k,
            match_by=match_by,
            suffix=suffix,
            exclude_overlaps=exclude_overlaps,
            dist_col=dist_col,
            preserve_input_order=preserve_input_order,
        )

    def subtract_overlaps(
        self,
        other: pl.DataFrame | Any,
        strand_behavior: str = "auto",
        *,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
    ) -> pl.DataFrame:
        return bio_subtract_overlaps(
            self._df,
            other,
            strand_behavior=strand_behavior,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
        )


def _install_dataframe_methods() -> None:
    def overlap_ranges_method(
        self: pl.DataFrame,
        other: pl.DataFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
    ) -> pl.DataFrame:
        return overlap_ranges(
            self,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
        )

    def overlap_range_pairs_method(
        self: pl.DataFrame,
        other: pl.DataFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
    ) -> tuple[list[int], list[int]]:
        return overlap_range_pairs(
            self,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
        )

    def merge_overlaps_method(
        self: pl.DataFrame,
        *,
        count_col: str | None = None,
        match_by: str | Iterable[str] | None = None,
        slack: int = 0,
    ) -> pl.DataFrame:
        return merge_overlaps(self, count_col=count_col, match_by=match_by, slack=slack)

    def cluster_overlaps_method(
        self: pl.DataFrame,
        *,
        match_by: str | Iterable[str] | None = None,
        cluster_column: str = "Cluster",
        slack: int = 0,
    ) -> pl.DataFrame:
        return cluster_overlaps(
            self,
            match_by=match_by,
            cluster_column=cluster_column,
            slack=slack,
        )

    def nearest_ranges_method(
        self: pl.DataFrame,
        other: pl.DataFrame | Any,
        *,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        k: int = 1,
        dist_col: str | None = "Distance",
        direction: str = "any",
        preserve_input_order: bool = True,
    ) -> pl.DataFrame:
        return nearest_ranges(
            self,
            other,
            match_by=match_by,
            suffix=suffix,
            exclude_overlaps=exclude_overlaps,
            k=k,
            dist_col=dist_col,
            direction=direction,
            preserve_input_order=preserve_input_order,
        )

    def subtract_overlaps_method(
        self: pl.DataFrame,
        other: pl.DataFrame | Any,
        *,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
    ) -> pl.DataFrame:
        return subtract_overlaps(
            self,
            other,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
        )

    pl.DataFrame.overlap_ranges = overlap_ranges_method
    pl.DataFrame.overlap_range_pairs = overlap_range_pairs_method
    pl.DataFrame.merge_overlaps = merge_overlaps_method
    pl.DataFrame.cluster_overlaps = cluster_overlaps_method
    pl.DataFrame.nearest_ranges = nearest_ranges_method
    pl.DataFrame.subtract_overlaps = subtract_overlaps_method


if not hasattr(pl.DataFrame, "bio"):
    pl.api.register_dataframe_namespace("bio")(_BioNamespace)

_install_dataframe_methods()

# Backwards-compatible aliases for the older helper names.
range_overlap_pairs = overlap_range_pairs
range_overlap = overlap_ranges
