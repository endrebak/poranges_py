from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeAlias

import polars as pl

from . import _polaranges

FrameLike: TypeAlias = pl.DataFrame | pl.LazyFrame | Any
PairsResult: TypeAlias = tuple[list[int], list[int]]
PairsReportResult: TypeAlias = tuple[list[int], list[int], dict[str, float]]
FrameReportResult: TypeAlias = tuple[pl.DataFrame, dict[str, float]]


def _coerce_frame(data: FrameLike) -> pl.DataFrame:
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if isinstance(data, pl.DataFrame):
        return data
    return pl.DataFrame(data)


def _normalize_match_by(value: str | Iterable[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_multiple(value: str | bool) -> str:
    if isinstance(value, bool):
        return "all" if value else "first"
    return value


def _normalize_use_strand(value: str | bool) -> str:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    return value


def _normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(kwargs)
    if "match_by" in normalized:
        normalized["match_by"] = _normalize_match_by(normalized["match_by"])
    if "multiple" in normalized:
        normalized["multiple"] = _normalize_multiple(normalized["multiple"])
    if "use_strand" in normalized:
        normalized["use_strand"] = _normalize_use_strand(normalized["use_strand"])
    return normalized


def _call_unary(name: str, df: FrameLike, /, *args: Any, **kwargs: Any) -> Any:
    return getattr(_polaranges, name)(
        _coerce_frame(df), *args, **_normalize_kwargs(kwargs)
    )


def _call_binary(
    name: str, left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> Any:
    return getattr(_polaranges, name)(
        _coerce_frame(left), _coerce_frame(right), *args, **_normalize_kwargs(kwargs)
    )


def benchmark_version() -> str:
    return _polaranges.benchmark_version()


def overlap_pairs(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> PairsResult:
    return _call_binary("overlap_pairs", left, right, *args, **kwargs)


def overlap_pairs_report(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> PairsReportResult:
    return _call_binary("overlap_pairs_report", left, right, *args, **kwargs)


def overlap(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("overlap", left, right, *args, **kwargs)


def overlap_report(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> FrameReportResult:
    return _call_binary("overlap_report", left, right, *args, **kwargs)


def merge_overlaps(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("merge_overlaps", df, *args, **kwargs)


def cluster_overlaps(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("cluster_overlaps", df, *args, **kwargs)


def count_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.Series:
    return _call_binary("count_overlaps", left, right, *args, **kwargs)


def join_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("join_overlaps", left, right, *args, **kwargs)


def intersect_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("intersect_overlaps", left, right, *args, **kwargs)


def set_intersect_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("set_intersect_overlaps", left, right, *args, **kwargs)


def set_union_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("set_union_overlaps", left, right, *args, **kwargs)


def sort_ranges(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("sort_ranges", df, *args, **kwargs)


def extend_ranges(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("extend_ranges", df, *args, **kwargs)


def tile_ranges(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("tile_ranges", df, *args, **kwargs)


def clip_ranges(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("clip_ranges", df, *args, **kwargs)


def group_cumsum(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("group_cumsum", df, *args, **kwargs)


def max_disjoint_overlaps(
    df: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_unary("max_disjoint_overlaps", df, *args, **kwargs)


def split_overlaps(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("split_overlaps", df, *args, **kwargs)


def outer_ranges(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("outer_ranges", df, *args, **kwargs)


def complement_ranges(
    df: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_unary("complement_ranges", df, *args, **kwargs)


def nearest(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("nearest", left, right, *args, **kwargs)


def nearest_report(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> FrameReportResult:
    return _call_binary("nearest_report", left, right, *args, **kwargs)


def subtract_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("subtract_overlaps", left, right, *args, **kwargs)


def bio_overlap(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("bio_overlap", left, right, *args, **kwargs)


def bio_overlap_report(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> FrameReportResult:
    return _call_binary("bio_overlap_report", left, right, *args, **kwargs)


def bio_nearest(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("bio_nearest", left, right, *args, **kwargs)


def bio_nearest_report(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> FrameReportResult:
    return _call_binary("bio_nearest_report", left, right, *args, **kwargs)


def bio_merge_overlaps(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("bio_merge_overlaps", df, *args, **kwargs)


def bio_cluster_overlaps(
    df: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_unary("bio_cluster_overlaps", df, *args, **kwargs)


def bio_count_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("bio_count_overlaps", left, right, *args, **kwargs)


def bio_join_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("bio_join_overlaps", left, right, *args, **kwargs)


def bio_intersect_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("bio_intersect_overlaps", left, right, *args, **kwargs)


def bio_set_intersect_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("bio_set_intersect_overlaps", left, right, *args, **kwargs)


def bio_set_union_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("bio_set_union_overlaps", left, right, *args, **kwargs)


def bio_sort_ranges(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("bio_sort_ranges", df, *args, **kwargs)


def bio_extend_ranges(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("bio_extend_ranges", df, *args, **kwargs)


def bio_tile_ranges(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("bio_tile_ranges", df, *args, **kwargs)


def bio_clip_ranges(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("bio_clip_ranges", df, *args, **kwargs)


def bio_group_cumsum(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("bio_group_cumsum", df, *args, **kwargs)


def bio_max_disjoint_overlaps(
    df: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_unary("bio_max_disjoint_overlaps", df, *args, **kwargs)


def bio_split_overlaps(
    df: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_unary("bio_split_overlaps", df, *args, **kwargs)


def bio_outer_ranges(df: FrameLike, /, *args: Any, **kwargs: Any) -> pl.DataFrame:
    return _call_unary("bio_outer_ranges", df, *args, **kwargs)


def bio_complement_ranges(
    df: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_unary("bio_complement_ranges", df, *args, **kwargs)


def bio_subtract_overlaps(
    left: FrameLike, right: FrameLike, /, *args: Any, **kwargs: Any
) -> pl.DataFrame:
    return _call_binary("bio_subtract_overlaps", left, right, *args, **kwargs)


def bio_has_valid_strand(df: FrameLike, /, *args: Any, **kwargs: Any) -> bool:
    return _call_unary("bio_has_valid_strand", df, *args, **kwargs)


class _NamespaceBase:
    def __init__(self, frame: pl.DataFrame | pl.LazyFrame) -> None:
        self._frame = frame


class _RangeNamespace(_NamespaceBase):
    pass


class _GenomicNamespace(_NamespaceBase):
    pass


class _BioNamespace(_NamespaceBase):
    pass


def _attach_methods(namespace_cls: type[_NamespaceBase], methods: dict[str, Any]) -> None:
    for method_name, function in methods.items():
        def method(
            self: _NamespaceBase,
            /,
            *args: Any,
            __function: Any = function,
            **kwargs: Any,
        ) -> Any:
            return __function(self._frame, *args, **kwargs)

        method.__name__ = method_name
        method.__qualname__ = f"{namespace_cls.__name__}.{method_name}"
        method.__doc__ = function.__doc__
        setattr(namespace_cls, method_name, method)


_attach_methods(
    _RangeNamespace,
    {
        "overlap_pairs": overlap_pairs,
        "overlap_pairs_report": overlap_pairs_report,
        "overlap": overlap,
        "overlap_report": overlap_report,
        "merge_overlaps": merge_overlaps,
        "cluster_overlaps": cluster_overlaps,
        "count_overlaps": count_overlaps,
        "join_overlaps": join_overlaps,
        "intersect_overlaps": intersect_overlaps,
        "set_intersect_overlaps": set_intersect_overlaps,
        "set_union_overlaps": set_union_overlaps,
        "sort_ranges": sort_ranges,
        "extend_ranges": extend_ranges,
        "tile_ranges": tile_ranges,
        "clip_ranges": clip_ranges,
        "group_cumsum": group_cumsum,
        "max_disjoint_overlaps": max_disjoint_overlaps,
        "split_overlaps": split_overlaps,
        "outer_ranges": outer_ranges,
        "complement_ranges": complement_ranges,
        "nearest": nearest,
        "nearest_report": nearest_report,
        "subtract_overlaps": subtract_overlaps,
    },
)

_attach_methods(
    _GenomicNamespace,
    {
        "overlap": bio_overlap,
        "overlap_report": bio_overlap_report,
        "nearest": bio_nearest,
        "nearest_report": bio_nearest_report,
    },
)

_attach_methods(
    _BioNamespace,
    {
        "overlap": bio_overlap,
        "overlap_report": bio_overlap_report,
        "nearest": bio_nearest,
        "nearest_report": bio_nearest_report,
        "merge_overlaps": bio_merge_overlaps,
        "cluster_overlaps": bio_cluster_overlaps,
        "count_overlaps": bio_count_overlaps,
        "join_overlaps": bio_join_overlaps,
        "intersect_overlaps": bio_intersect_overlaps,
        "set_intersect_overlaps": bio_set_intersect_overlaps,
        "set_union_overlaps": bio_set_union_overlaps,
        "sort_ranges": bio_sort_ranges,
        "extend_ranges": bio_extend_ranges,
        "tile_ranges": bio_tile_ranges,
        "clip_ranges": bio_clip_ranges,
        "group_cumsum": bio_group_cumsum,
        "max_disjoint_overlaps": bio_max_disjoint_overlaps,
        "split_overlaps": bio_split_overlaps,
        "outer_ranges": bio_outer_ranges,
        "complement_ranges": bio_complement_ranges,
        "subtract_overlaps": bio_subtract_overlaps,
        "has_valid_strand": bio_has_valid_strand,
    },
)

if not hasattr(pl.DataFrame, "r"):
    pl.api.register_dataframe_namespace("r")(_RangeNamespace)
if not hasattr(pl.LazyFrame, "r"):
    pl.api.register_lazyframe_namespace("r")(_RangeNamespace)
if not hasattr(pl.DataFrame, "g"):
    pl.api.register_dataframe_namespace("g")(_GenomicNamespace)
if not hasattr(pl.LazyFrame, "g"):
    pl.api.register_lazyframe_namespace("g")(_GenomicNamespace)
if not hasattr(pl.DataFrame, "bio"):
    pl.api.register_dataframe_namespace("bio")(_BioNamespace)
if not hasattr(pl.LazyFrame, "bio"):
    pl.api.register_lazyframe_namespace("bio")(_BioNamespace)


__all__ = [
    "benchmark_version",
    "bio_clip_ranges",
    "bio_cluster_overlaps",
    "bio_complement_ranges",
    "bio_count_overlaps",
    "bio_extend_ranges",
    "bio_group_cumsum",
    "bio_has_valid_strand",
    "bio_intersect_overlaps",
    "bio_join_overlaps",
    "bio_max_disjoint_overlaps",
    "bio_merge_overlaps",
    "bio_nearest",
    "bio_nearest_report",
    "bio_outer_ranges",
    "bio_overlap",
    "bio_overlap_report",
    "bio_set_intersect_overlaps",
    "bio_set_union_overlaps",
    "bio_sort_ranges",
    "bio_split_overlaps",
    "bio_subtract_overlaps",
    "bio_tile_ranges",
    "clip_ranges",
    "cluster_overlaps",
    "complement_ranges",
    "count_overlaps",
    "extend_ranges",
    "group_cumsum",
    "intersect_overlaps",
    "join_overlaps",
    "max_disjoint_overlaps",
    "merge_overlaps",
    "nearest",
    "nearest_report",
    "outer_ranges",
    "overlap",
    "overlap_pairs",
    "overlap_pairs_report",
    "overlap_report",
    "set_intersect_overlaps",
    "set_union_overlaps",
    "sort_ranges",
    "split_overlaps",
    "subtract_overlaps",
    "tile_ranges",
]
