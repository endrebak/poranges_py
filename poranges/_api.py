from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import polars as pl

from . import _poranges

JOIN_SUFFIX = "_b"
ParallelMode = Literal["serial", "auto", "force"]


@dataclass(frozen=True, slots=True)
class ParallelConfig:
    enabled: bool = False
    force_parallel: bool = False
    target_batch_weight: int = 16_384
    min_total_weight: int = 65_536
    min_num_groups: int = 64
    min_num_batches: int = 2
    max_dominance_ratio: float = 0.85

    @classmethod
    def auto(cls) -> ParallelConfig:
        return cls(
            enabled=True,
            target_batch_weight=4_096,
            min_total_weight=24_000,
            min_num_groups=32,
            min_num_batches=2,
            max_dominance_ratio=0.80,
        )

    @classmethod
    def force(cls) -> ParallelConfig:
        cfg = cls.auto()
        return cls(**asdict(cfg), force_parallel=True)


@dataclass(frozen=True, slots=True)
class Timings:
    total: float
    prepare: float
    rechunk: float
    factorization: float
    group_task_build: float
    batch_build: float
    kernel: float
    reconstruction: float

    def as_dict(self, *, milliseconds: bool = False) -> dict[str, float]:
        values = asdict(self)
        if not milliseconds:
            return values
        return {name: value * 1000.0 for name, value in values.items()}


@dataclass(frozen=True, slots=True)
class OperationReport:
    result: pl.DataFrame | pl.LazyFrame | tuple[list[int], list[int]]
    timings: Timings


def _arg_to_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _normalize_match_by(value: str | Iterable[str] | None) -> list[str] | None:
    normalized = _arg_to_list(value)
    return normalized or None


def _is_lazyframe(value: Any) -> bool:
    return isinstance(value, pl.LazyFrame)


def _coerce_frame(data: Any) -> pl.DataFrame | pl.LazyFrame:
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return data
    return pl.DataFrame(data)


def _coerce_df(data: Any) -> pl.DataFrame:
    frame = _coerce_frame(data)
    if isinstance(frame, pl.LazyFrame):
        msg = "this operation currently requires an eager DataFrame"
        raise TypeError(msg)
    return frame


def _coerce_binary_frames(
    left: Any, right: Any
) -> tuple[pl.DataFrame | pl.LazyFrame, pl.DataFrame | pl.LazyFrame]:
    left_frame = _coerce_frame(left)
    right_frame = _coerce_frame(right)
    if _is_lazyframe(left_frame) or _is_lazyframe(right_frame):
        if isinstance(left_frame, pl.DataFrame):
            left_frame = left_frame.lazy()
        if isinstance(right_frame, pl.DataFrame):
            right_frame = right_frame.lazy()
    return left_frame, right_frame


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


def _normalize_parallel(
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None,
) -> ParallelConfig:
    if parallel in (None, False, "serial"):
        return ParallelConfig()
    if parallel in (True, "auto"):
        return ParallelConfig.auto()
    if parallel == "force":
        return ParallelConfig.force()
    if isinstance(parallel, ParallelConfig):
        return parallel
    if isinstance(parallel, Mapping):
        return ParallelConfig(**dict(parallel))

    msg = (
        "parallel must be one of None, False, True, 'serial', 'auto', 'force', "
        "a ParallelConfig, or a mapping of ParallelConfig fields"
    )
    raise ValueError(msg)


def _parallel_kwargs(
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None,
) -> dict[str, Any]:
    cfg = _normalize_parallel(parallel)
    return {
        "parallel_enabled": cfg.enabled,
        "force_parallel": cfg.force_parallel,
        "target_batch_weight": cfg.target_batch_weight,
        "min_total_weight": cfg.min_total_weight,
        "min_num_groups": cfg.min_num_groups,
        "min_num_batches": cfg.min_num_batches,
        "max_dominance_ratio": cfg.max_dominance_ratio,
    }


def _build_timings(values: Mapping[str, float]) -> Timings:
    return Timings(**dict(values))


def _build_report(
    result: pl.DataFrame | pl.LazyFrame | tuple[list[int], list[int]],
    timings: Mapping[str, float],
) -> OperationReport:
    return OperationReport(result=result, timings=_build_timings(timings))


def overlap_range_pairs(
    left: pl.DataFrame | Any,
    right: pl.DataFrame | Any,
    multiple: str = "all",
    slack: int = 0,
    *,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
) -> tuple[list[int], list[int]]:
    left_df = _coerce_frame(left)
    right_df = _coerce_frame(right)
    if _is_lazyframe(left_df) or _is_lazyframe(right_df):
        msg = "overlap_range_pairs currently requires eager DataFrame inputs"
        raise TypeError(msg)

    left_idx, right_idx = _poranges.overlap_range_pairs(
        left_df,
        right_df,
        multiple,
        slack,
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
        **_parallel_kwargs(parallel),
    )
    return list(left_idx), list(right_idx)


def overlap_range_pairs_report(
    left: pl.DataFrame | Any,
    right: pl.DataFrame | Any,
    multiple: str = "all",
    slack: int = 0,
    *,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
) -> OperationReport:
    left_df = _coerce_frame(left)
    right_df = _coerce_frame(right)
    if _is_lazyframe(left_df) or _is_lazyframe(right_df):
        msg = "overlap_range_pairs_report currently requires eager DataFrame inputs"
        raise TypeError(msg)

    left_idx, right_idx, timings = _poranges.overlap_range_pairs_report(
        left_df,
        right_df,
        multiple,
        slack,
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
        **_parallel_kwargs(parallel),
    )
    return _build_report((list(left_idx), list(right_idx)), timings)


def overlap_ranges(
    left: pl.DataFrame | pl.LazyFrame | Any,
    right: pl.DataFrame | pl.LazyFrame | Any,
    multiple: str = "all",
    slack: int = 0,
    *,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    left_frame, right_frame = _coerce_binary_frames(left, right)
    kwargs = dict(
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
        **_parallel_kwargs(parallel),
    )
    if _is_lazyframe(left_frame):
        return _poranges.overlap_ranges_lazy(left_frame, right_frame, multiple, slack, **kwargs)
    return _poranges.overlap_ranges(left_frame, right_frame, multiple, slack, **kwargs)


def overlap_ranges_report(
    left: pl.DataFrame | pl.LazyFrame | Any,
    right: pl.DataFrame | pl.LazyFrame | Any,
    multiple: str = "all",
    slack: int = 0,
    *,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
) -> OperationReport:
    left_frame, right_frame = _coerce_binary_frames(left, right)
    kwargs = dict(
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
        **_parallel_kwargs(parallel),
    )
    if _is_lazyframe(left_frame):
        result, timings = _poranges.overlap_ranges_lazy_report(
            left_frame, right_frame, multiple, slack, **kwargs
        )
        return _build_report(result, timings)

    result, timings = _poranges.overlap_ranges_report(
        left_frame, right_frame, multiple, slack, **kwargs
    )
    return _build_report(result, timings)


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
    left: pl.DataFrame | pl.LazyFrame | Any,
    right: pl.DataFrame | pl.LazyFrame | Any,
    *,
    match_by: str | Iterable[str] | None = None,
    suffix: str = JOIN_SUFFIX,
    exclude_overlaps: bool = False,
    k: int = 1,
    dist_col: str | None = "Distance",
    direction: str = "any",
    preserve_input_order: bool = True,
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    left_frame, right_frame = _coerce_binary_frames(left, right)
    kwargs = dict(
        match_by=_normalize_match_by(match_by),
        suffix=suffix,
        exclude_overlaps=exclude_overlaps,
        k=k,
        dist_col=dist_col,
        direction=direction,
        preserve_input_order=preserve_input_order,
        **_parallel_kwargs(parallel),
    )
    if _is_lazyframe(left_frame):
        return _poranges.nearest_ranges_lazy(left_frame, right_frame, **kwargs)
    return _poranges.nearest_ranges(left_frame, right_frame, **kwargs)


def nearest_ranges_report(
    left: pl.DataFrame | pl.LazyFrame | Any,
    right: pl.DataFrame | pl.LazyFrame | Any,
    *,
    match_by: str | Iterable[str] | None = None,
    suffix: str = JOIN_SUFFIX,
    exclude_overlaps: bool = False,
    k: int = 1,
    dist_col: str | None = "Distance",
    direction: str = "any",
    preserve_input_order: bool = True,
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
) -> OperationReport:
    left_frame, right_frame = _coerce_binary_frames(left, right)
    kwargs = dict(
        match_by=_normalize_match_by(match_by),
        suffix=suffix,
        exclude_overlaps=exclude_overlaps,
        k=k,
        dist_col=dist_col,
        direction=direction,
        preserve_input_order=preserve_input_order,
        **_parallel_kwargs(parallel),
    )
    if _is_lazyframe(left_frame):
        result, timings = _poranges.nearest_ranges_lazy_report(left_frame, right_frame, **kwargs)
        return _build_report(result, timings)

    result, timings = _poranges.nearest_ranges_report(left_frame, right_frame, **kwargs)
    return _build_report(result, timings)


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
    left: pl.DataFrame | pl.LazyFrame | Any,
    right: pl.DataFrame | pl.LazyFrame | Any,
    strand_behavior: str = "auto",
    slack: int = 0,
    *,
    multiple: str | bool = False,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    invert: bool = False,
    preserve_input_order: bool = True,
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    left_frame, right_frame = _coerce_binary_frames(left, right)
    kwargs = dict(
        strand_behavior=strand_behavior,
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        invert=invert,
        preserve_input_order=preserve_input_order,
        **_parallel_kwargs(parallel),
    )
    if _is_lazyframe(left_frame):
        return _poranges.bio_overlap_ranges_lazy(
            left_frame, right_frame, _normalize_bio_multiple(multiple), slack, **kwargs
        )
    return _poranges.bio_overlap_ranges(
        left_frame, right_frame, _normalize_bio_multiple(multiple), slack, **kwargs
    )


def bio_overlap_ranges_report(
    left: pl.DataFrame | pl.LazyFrame | Any,
    right: pl.DataFrame | pl.LazyFrame | Any,
    strand_behavior: str = "auto",
    slack: int = 0,
    *,
    multiple: str | bool = False,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    invert: bool = False,
    preserve_input_order: bool = True,
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
) -> OperationReport:
    left_frame, right_frame = _coerce_binary_frames(left, right)
    kwargs = dict(
        strand_behavior=strand_behavior,
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        invert=invert,
        preserve_input_order=preserve_input_order,
        **_parallel_kwargs(parallel),
    )
    if _is_lazyframe(left_frame):
        result, timings = _poranges.bio_overlap_ranges_lazy_report(
            left_frame, right_frame, _normalize_bio_multiple(multiple), slack, **kwargs
        )
        return _build_report(result, timings)

    result, timings = _poranges.bio_overlap_ranges_report(
        left_frame, right_frame, _normalize_bio_multiple(multiple), slack, **kwargs
    )
    return _build_report(result, timings)


def bio_nearest_ranges(
    left: pl.DataFrame | pl.LazyFrame | Any,
    right: pl.DataFrame | pl.LazyFrame | Any,
    strand_behavior: str = "auto",
    direction: str = "any",
    *,
    k: int = 1,
    match_by: str | Iterable[str] | None = None,
    suffix: str = JOIN_SUFFIX,
    exclude_overlaps: bool = False,
    dist_col: str | None = "Distance",
    preserve_input_order: bool = True,
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    left_frame, right_frame = _coerce_binary_frames(left, right)
    kwargs = dict(
        strand_behavior=strand_behavior,
        match_by=_normalize_match_by(match_by),
        suffix=suffix,
        exclude_overlaps=exclude_overlaps,
        k=k,
        dist_col=dist_col,
        direction=direction,
        preserve_input_order=preserve_input_order,
        **_parallel_kwargs(parallel),
    )
    if _is_lazyframe(left_frame):
        return _poranges.bio_nearest_ranges_lazy(left_frame, right_frame, **kwargs)
    return _poranges.bio_nearest_ranges(left_frame, right_frame, **kwargs)


def bio_nearest_ranges_report(
    left: pl.DataFrame | pl.LazyFrame | Any,
    right: pl.DataFrame | pl.LazyFrame | Any,
    strand_behavior: str = "auto",
    direction: str = "any",
    *,
    k: int = 1,
    match_by: str | Iterable[str] | None = None,
    suffix: str = JOIN_SUFFIX,
    exclude_overlaps: bool = False,
    dist_col: str | None = "Distance",
    preserve_input_order: bool = True,
    parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
) -> OperationReport:
    left_frame, right_frame = _coerce_binary_frames(left, right)
    kwargs = dict(
        strand_behavior=strand_behavior,
        match_by=_normalize_match_by(match_by),
        suffix=suffix,
        exclude_overlaps=exclude_overlaps,
        k=k,
        dist_col=dist_col,
        direction=direction,
        preserve_input_order=preserve_input_order,
        **_parallel_kwargs(parallel),
    )
    if _is_lazyframe(left_frame):
        result, timings = _poranges.bio_nearest_ranges_lazy_report(
            left_frame, right_frame, **kwargs
        )
        return _build_report(result, timings)

    result, timings = _poranges.bio_nearest_ranges_report(left_frame, right_frame, **kwargs)
    return _build_report(result, timings)


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


class _RangeNamespace:
    def __init__(self, frame: pl.DataFrame | pl.LazyFrame) -> None:
        self._frame = frame

    def overlap_ranges(
        self,
        other: pl.DataFrame | pl.LazyFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        return overlap_ranges(
            self._frame,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def overlap_ranges_report(
        self,
        other: pl.DataFrame | pl.LazyFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> OperationReport:
        return overlap_ranges_report(
            self._frame,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def overlap_range_pairs(
        self,
        other: pl.DataFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> tuple[list[int], list[int]]:
        return overlap_range_pairs(
            self._frame,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def overlap_range_pairs_report(
        self,
        other: pl.DataFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> OperationReport:
        return overlap_range_pairs_report(
            self._frame,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def nearest_ranges(
        self,
        other: pl.DataFrame | pl.LazyFrame | Any,
        *,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        k: int = 1,
        dist_col: str | None = "Distance",
        direction: str = "any",
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        return nearest_ranges(
            self._frame,
            other,
            match_by=match_by,
            suffix=suffix,
            exclude_overlaps=exclude_overlaps,
            k=k,
            dist_col=dist_col,
            direction=direction,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def nearest_ranges_report(
        self,
        other: pl.DataFrame | pl.LazyFrame | Any,
        *,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        k: int = 1,
        dist_col: str | None = "Distance",
        direction: str = "any",
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> OperationReport:
        return nearest_ranges_report(
            self._frame,
            other,
            match_by=match_by,
            suffix=suffix,
            exclude_overlaps=exclude_overlaps,
            k=k,
            dist_col=dist_col,
            direction=direction,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )


class _BioNamespace:
    def __init__(self, frame: pl.DataFrame | pl.LazyFrame) -> None:
        self._frame = frame

    def has_valid_strand(self, *, strand_col: str = "Strand") -> bool:
        return bio_has_valid_strand(self._frame, strand_col=strand_col)

    def merge_overlaps(
        self,
        use_strand: str | bool = "auto",
        *,
        count_col: str | None = None,
        match_by: str | Iterable[str] | None = None,
        slack: int = 0,
    ) -> pl.DataFrame:
        return bio_merge_overlaps(
            self._frame,
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
            self._frame,
            use_strand=use_strand,
            match_by=match_by,
            cluster_column=cluster_column,
            slack=slack,
        )

    def overlap_ranges(
        self,
        other: pl.DataFrame | pl.LazyFrame | Any,
        strand_behavior: str = "auto",
        slack: int = 0,
        *,
        multiple: str | bool = False,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        invert: bool = False,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        return bio_overlap_ranges(
            self._frame,
            other,
            strand_behavior=strand_behavior,
            slack=slack,
            multiple=multiple,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            invert=invert,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def overlap_ranges_report(
        self,
        other: pl.DataFrame | pl.LazyFrame | Any,
        strand_behavior: str = "auto",
        slack: int = 0,
        *,
        multiple: str | bool = False,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        invert: bool = False,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> OperationReport:
        return bio_overlap_ranges_report(
            self._frame,
            other,
            strand_behavior=strand_behavior,
            slack=slack,
            multiple=multiple,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            invert=invert,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def nearest_ranges(
        self,
        other: pl.DataFrame | pl.LazyFrame | Any,
        strand_behavior: str = "auto",
        direction: str = "any",
        *,
        k: int = 1,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        dist_col: str | None = "Distance",
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        return bio_nearest_ranges(
            self._frame,
            other,
            strand_behavior=strand_behavior,
            direction=direction,
            k=k,
            match_by=match_by,
            suffix=suffix,
            exclude_overlaps=exclude_overlaps,
            dist_col=dist_col,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def nearest_ranges_report(
        self,
        other: pl.DataFrame | pl.LazyFrame | Any,
        strand_behavior: str = "auto",
        direction: str = "any",
        *,
        k: int = 1,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        dist_col: str | None = "Distance",
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> OperationReport:
        return bio_nearest_ranges_report(
            self._frame,
            other,
            strand_behavior=strand_behavior,
            direction=direction,
            k=k,
            match_by=match_by,
            suffix=suffix,
            exclude_overlaps=exclude_overlaps,
            dist_col=dist_col,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
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
            self._frame,
            other,
            strand_behavior=strand_behavior,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
        )


def _install_dataframe_methods() -> None:
    def overlap_ranges_method(
        self: pl.DataFrame,
        other: pl.DataFrame | pl.LazyFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
        return overlap_ranges(
            self,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def overlap_ranges_report_method(
        self: pl.DataFrame,
        other: pl.DataFrame | pl.LazyFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> OperationReport:
        return overlap_ranges_report(
            self,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
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
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> tuple[list[int], list[int]]:
        return overlap_range_pairs(
            self,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def overlap_range_pairs_report_method(
        self: pl.DataFrame,
        other: pl.DataFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> OperationReport:
        return overlap_range_pairs_report(
            self,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
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
        other: pl.DataFrame | pl.LazyFrame | Any,
        *,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        k: int = 1,
        dist_col: str | None = "Distance",
        direction: str = "any",
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> pl.DataFrame | pl.LazyFrame:
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
            parallel=parallel,
        )

    def nearest_ranges_report_method(
        self: pl.DataFrame,
        other: pl.DataFrame | pl.LazyFrame | Any,
        *,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        k: int = 1,
        dist_col: str | None = "Distance",
        direction: str = "any",
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> OperationReport:
        return nearest_ranges_report(
            self,
            other,
            match_by=match_by,
            suffix=suffix,
            exclude_overlaps=exclude_overlaps,
            k=k,
            dist_col=dist_col,
            direction=direction,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
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
    pl.DataFrame.overlap_ranges_report = overlap_ranges_report_method
    pl.DataFrame.overlap_range_pairs = overlap_range_pairs_method
    pl.DataFrame.overlap_range_pairs_report = overlap_range_pairs_report_method
    pl.DataFrame.merge_overlaps = merge_overlaps_method
    pl.DataFrame.cluster_overlaps = cluster_overlaps_method
    pl.DataFrame.nearest_ranges = nearest_ranges_method
    pl.DataFrame.nearest_ranges_report = nearest_ranges_report_method
    pl.DataFrame.subtract_overlaps = subtract_overlaps_method


def _install_lazyframe_methods() -> None:
    def overlap_ranges_method(
        self: pl.LazyFrame,
        other: pl.DataFrame | pl.LazyFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> pl.LazyFrame:
        out = overlap_ranges(
            self,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )
        assert isinstance(out, pl.LazyFrame)
        return out

    def overlap_ranges_report_method(
        self: pl.LazyFrame,
        other: pl.DataFrame | pl.LazyFrame | Any,
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> OperationReport:
        return overlap_ranges_report(
            self,
            other,
            multiple=multiple,
            slack=slack,
            contained_intervals_only=contained_intervals_only,
            match_by=match_by,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    def nearest_ranges_method(
        self: pl.LazyFrame,
        other: pl.DataFrame | pl.LazyFrame | Any,
        *,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        k: int = 1,
        dist_col: str | None = "Distance",
        direction: str = "any",
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> pl.LazyFrame:
        out = nearest_ranges(
            self,
            other,
            match_by=match_by,
            suffix=suffix,
            exclude_overlaps=exclude_overlaps,
            k=k,
            dist_col=dist_col,
            direction=direction,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )
        assert isinstance(out, pl.LazyFrame)
        return out

    def nearest_ranges_report_method(
        self: pl.LazyFrame,
        other: pl.DataFrame | pl.LazyFrame | Any,
        *,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        k: int = 1,
        dist_col: str | None = "Distance",
        direction: str = "any",
        preserve_input_order: bool = True,
        parallel: ParallelConfig | Mapping[str, Any] | ParallelMode | bool | None = None,
    ) -> OperationReport:
        return nearest_ranges_report(
            self,
            other,
            match_by=match_by,
            suffix=suffix,
            exclude_overlaps=exclude_overlaps,
            k=k,
            dist_col=dist_col,
            direction=direction,
            preserve_input_order=preserve_input_order,
            parallel=parallel,
        )

    pl.LazyFrame.overlap_ranges = overlap_ranges_method
    pl.LazyFrame.overlap_ranges_report = overlap_ranges_report_method
    pl.LazyFrame.nearest_ranges = nearest_ranges_method
    pl.LazyFrame.nearest_ranges_report = nearest_ranges_report_method


if not hasattr(pl.DataFrame, "bio"):
    pl.api.register_dataframe_namespace("bio")(_BioNamespace)
if not hasattr(pl.LazyFrame, "bio"):
    pl.api.register_lazyframe_namespace("bio")(_BioNamespace)
if not hasattr(pl.DataFrame, "r"):
    pl.api.register_dataframe_namespace("r")(_RangeNamespace)
if not hasattr(pl.LazyFrame, "r"):
    pl.api.register_lazyframe_namespace("r")(_RangeNamespace)
if not hasattr(pl.DataFrame, "g"):
    pl.api.register_dataframe_namespace("g")(_BioNamespace)
if not hasattr(pl.LazyFrame, "g"):
    pl.api.register_lazyframe_namespace("g")(_BioNamespace)

_install_dataframe_methods()
_install_lazyframe_methods()

# Backwards-compatible aliases for the older helper names.
range_overlap_pairs = overlap_range_pairs
range_overlap = overlap_ranges
