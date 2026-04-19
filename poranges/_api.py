from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

import polars as pl

from . import _poranges

CHROM_COL = "Chromosome"
START_COL = "Start"
END_COL = "End"
STRAND_COL = "Strand"
JOIN_SUFFIX = "_b"
TEMP_INDEX_COL = "__poranges_rowid__"
VALID_STRANDS = {"+", "-"}

T = TypeVar("T", bound="RangeFrame")


def _arg_to_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _normalize_match_by(value: str | Iterable[str] | None) -> list[str] | None:
    normalized = _arg_to_list(value)
    return normalized or None


def _ensure_polars_df(data: Any) -> pl.DataFrame:
    if isinstance(data, RangeFrame):
        return data.df
    if isinstance(data, pl.DataFrame):
        return data
    return pl.DataFrame(data)


def _ensure_pyranges(data: Any) -> "PyRanges":
    if isinstance(data, PyRanges):
        return data
    return PyRanges(data)


def range_overlap_pairs(
    left: "RangeFrame | pl.DataFrame | Any",
    right: "RangeFrame | pl.DataFrame | Any",
    multiple: str = "all",
    slack: int = 0,
    *,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
) -> tuple[list[int], list[int]]:
    left_idx, right_idx = _poranges.range_overlap_pairs(
        _ensure_polars_df(left),
        _ensure_polars_df(right),
        multiple,
        slack,
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
    )
    return list(left_idx), list(right_idx)


def range_overlap(
    left: "RangeFrame | pl.DataFrame | Any",
    right: "RangeFrame | pl.DataFrame | Any",
    multiple: str = "all",
    slack: int = 0,
    *,
    contained_intervals_only: bool = False,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
) -> pl.DataFrame:
    return _poranges.range_overlap(
        _ensure_polars_df(left),
        _ensure_polars_df(right),
        multiple,
        slack,
        contained_intervals_only=contained_intervals_only,
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
    )


def merge_overlaps(
    df: "RangeFrame | pl.DataFrame | Any",
    *,
    count_col: str | None = None,
    match_by: str | Iterable[str] | None = None,
    slack: int = 0,
) -> pl.DataFrame:
    return _poranges.merge_overlaps(
        _ensure_polars_df(df),
        count_col=count_col,
        match_by=_normalize_match_by(match_by),
        slack=slack,
    )


def cluster_overlaps(
    df: "RangeFrame | pl.DataFrame | Any",
    *,
    match_by: str | Iterable[str] | None = None,
    cluster_column: str = "Cluster",
    slack: int = 0,
) -> pl.DataFrame:
    return _poranges.cluster_overlaps(
        _ensure_polars_df(df),
        match_by=_normalize_match_by(match_by),
        cluster_column=cluster_column,
        slack=slack,
    )


def nearest_ranges(
    left: "RangeFrame | pl.DataFrame | Any",
    right: "RangeFrame | pl.DataFrame | Any",
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
        _ensure_polars_df(left),
        _ensure_polars_df(right),
        match_by=_normalize_match_by(match_by),
        suffix=suffix,
        exclude_overlaps=exclude_overlaps,
        k=k,
        dist_col=dist_col,
        direction=direction,
        preserve_input_order=preserve_input_order,
    )


def subtract_overlaps(
    left: "RangeFrame | pl.DataFrame | Any",
    right: "RangeFrame | pl.DataFrame | Any",
    *,
    match_by: str | Iterable[str] | None = None,
    preserve_input_order: bool = True,
) -> pl.DataFrame:
    return _poranges.subtract_overlaps(
        _ensure_polars_df(left),
        _ensure_polars_df(right),
        match_by=_normalize_match_by(match_by),
        preserve_input_order=preserve_input_order,
    )


class RangeFrame:
    def __init__(self, data: Any = None, *args: Any, **kwargs: Any) -> None:
        if isinstance(data, RangeFrame):
            if args or kwargs:
                msg = "cannot pass additional arguments when constructing from an existing RangeFrame"
                raise TypeError(msg)
            self.df = data.df.clone()
            return

        if isinstance(data, pl.DataFrame):
            if args or kwargs:
                msg = "cannot pass additional arguments when constructing from a polars.DataFrame"
                raise TypeError(msg)
            self.df = data
            return

        if data is None and not args and not kwargs:
            self.df = pl.DataFrame()
            return

        self.df = pl.DataFrame(data, *args, **kwargs)

    def __repr__(self) -> str:
        return repr(self.df)

    def __str__(self) -> str:
        return str(self.df)

    def __len__(self) -> int:
        return self.df.height

    def __getitem__(self, key: Any) -> Any:
        return self.df[key]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.df, name)

    @property
    def columns(self) -> list[str]:
        return self.df.columns

    @property
    def schema(self) -> dict[str, pl.DataType]:
        return dict(self.df.schema)

    @property
    def shape(self) -> tuple[int, int]:
        return self.df.shape

    def to_polars(self) -> pl.DataFrame:
        return self.df

    def clone(self: T) -> T:
        return type(self)(self.df.clone())

    copy = clone

    def overlap(
        self: T,
        other: "RangeFrame | pl.DataFrame | Any",
        multiple: str = "all",
        slack: int = 0,
        *,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
    ) -> T:
        return type(self)(
            range_overlap(
                self.df,
                other,
                multiple=multiple,
                slack=slack,
                contained_intervals_only=contained_intervals_only,
                match_by=match_by,
                preserve_input_order=preserve_input_order,
            )
        )

    def merge_overlaps(
        self: T,
        *,
        count_col: str | None = None,
        match_by: str | Iterable[str] | None = None,
        slack: int = 0,
    ) -> T:
        return type(self)(
            merge_overlaps(self.df, count_col=count_col, match_by=match_by, slack=slack)
        )

    def cluster_overlaps(
        self: T,
        *,
        match_by: str | Iterable[str] | None = None,
        cluster_column: str = "Cluster",
        slack: int = 0,
    ) -> T:
        return type(self)(
            cluster_overlaps(
                self.df,
                match_by=match_by,
                cluster_column=cluster_column,
                slack=slack,
            )
        )

    def nearest_ranges(
        self: T,
        other: "RangeFrame | pl.DataFrame | Any",
        *,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        k: int = 1,
        dist_col: str | None = "Distance",
        direction: str = "any",
        preserve_input_order: bool = True,
    ) -> T:
        return type(self)(
            nearest_ranges(
                self.df,
                other,
                match_by=match_by,
                suffix=suffix,
                exclude_overlaps=exclude_overlaps,
                k=k,
                dist_col=dist_col,
                direction=direction,
                preserve_input_order=preserve_input_order,
            )
        )

    def subtract_overlaps(
        self: T,
        other: "RangeFrame | pl.DataFrame | Any",
        match_by: str | Iterable[str] | None = None,
        *,
        preserve_input_order: bool = True,
    ) -> T:
        return type(self)(
            subtract_overlaps(
                self.df,
                other,
                match_by=match_by,
                preserve_input_order=preserve_input_order,
            )
        )


def _validate_and_convert_use_strand(gr: "PyRanges", use_strand: str | bool) -> bool:
    if use_strand not in {"auto", True, False}:
        msg = "use_strand must be one of 'auto', True, or False"
        raise ValueError(msg)
    if use_strand is True and not gr.has_strand:
        msg = "cannot set use_strand=True when the Strand column is missing"
        raise ValueError(msg)
    if use_strand == "auto":
        return gr.strand_valid
    return bool(use_strand)


def _validate_and_convert_strand_behavior(
    left: "PyRanges",
    right: "PyRanges",
    strand_behavior: str,
) -> str:
    if strand_behavior not in {"auto", "same", "opposite", "ignore"}:
        msg = "strand_behavior must be one of 'auto', 'same', 'opposite', or 'ignore'"
        raise ValueError(msg)

    if strand_behavior == "auto":
        return "same" if left.strand_valid and right.strand_valid else "ignore"

    if strand_behavior in {"same", "opposite"} and not (left.strand_valid and right.strand_valid):
        msg = f"can only do {strand_behavior} strand operations when both PyRanges have valid strand info"
        raise ValueError(msg)

    return strand_behavior


def _prepare_by_single(
    gr: "PyRanges",
    use_strand: str | bool,
    match_by: str | Iterable[str] | None,
) -> list[str]:
    default_cols = [CHROM_COL]
    if _validate_and_convert_use_strand(gr, use_strand):
        default_cols.append(STRAND_COL)
    return _dedupe_preserving_order([*default_cols, *_arg_to_list(match_by)])


def _flip_strand(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col(STRAND_COL) == "+")
        .then(pl.lit("-"))
        .when(pl.col(STRAND_COL) == "-")
        .then(pl.lit("+"))
        .otherwise(pl.col(STRAND_COL))
        .alias(STRAND_COL)
    )


def _prepare_by_binary(
    left: "PyRanges",
    right: "PyRanges",
    strand_behavior: str = "auto",
    match_by: str | Iterable[str] | None = None,
) -> tuple["PyRanges", list[str]]:
    normalized = _validate_and_convert_strand_behavior(left, right, strand_behavior)
    default_cols = [CHROM_COL] if normalized == "ignore" else [CHROM_COL, STRAND_COL]
    by = _dedupe_preserving_order([*default_cols, *_arg_to_list(match_by)])
    if normalized == "opposite":
        return PyRanges(_flip_strand(right.df)), by
    return right, by


def _split_on_strand(gr: "PyRanges") -> tuple["PyRanges", "PyRanges"]:
    if not gr.has_strand:
        msg = "PyRanges has no strand column"
        raise ValueError(msg)
    return (
        PyRanges(gr.df.filter(pl.col(STRAND_COL) == "+")),
        PyRanges(gr.df.filter(pl.col(STRAND_COL) == "-")),
    )


class PyRanges(RangeFrame):
    @property
    def has_strand(self) -> bool:
        return STRAND_COL in self.df.columns

    @property
    def strand_valid(self) -> bool:
        if not self.has_strand:
            return False
        strand = self.df.get_column(STRAND_COL)
        if strand.null_count() > 0:
            return False
        return set(strand.unique().to_list()).issubset(VALID_STRANDS)

    def merge_overlaps(
        self,
        use_strand: str | bool = "auto",
        *,
        count_col: str | None = None,
        match_by: str | Iterable[str] | None = None,
        slack: int = 0,
    ) -> "PyRanges":
        by = _prepare_by_single(self, use_strand, match_by)
        return PyRanges(merge_overlaps(self.df, count_col=count_col, match_by=by, slack=slack))

    def cluster_overlaps(
        self,
        use_strand: str | bool = "auto",
        *,
        match_by: str | Iterable[str] | None = None,
        slack: int = 0,
        cluster_column: str = "Cluster",
    ) -> "PyRanges":
        by = _prepare_by_single(self, use_strand, match_by)
        return PyRanges(
            cluster_overlaps(
                self.df,
                match_by=by,
                cluster_column=cluster_column,
                slack=slack,
            )
        )

    def nearest_ranges(
        self,
        other: Any,
        strand_behavior: str = "auto",
        direction: str = "any",
        *,
        k: int = 1,
        match_by: str | Iterable[str] | None = None,
        suffix: str = JOIN_SUFFIX,
        exclude_overlaps: bool = False,
        dist_col: str | None = "Distance",
        preserve_input_order: bool = True,
    ) -> "PyRanges":
        other_gr = _ensure_pyranges(other)
        prepared_other, by = _prepare_by_binary(
            self,
            other_gr,
            strand_behavior=strand_behavior,
            match_by=match_by,
        )

        if direction == "any":
            return PyRanges(
                nearest_ranges(
                    self.df,
                    prepared_other.df,
                    match_by=by,
                    suffix=suffix,
                    exclude_overlaps=exclude_overlaps,
                    k=k,
                    dist_col=dist_col,
                    direction="any",
                    preserve_input_order=preserve_input_order,
                )
            )

        validated_strand_behavior = _validate_and_convert_strand_behavior(
            self,
            other_gr,
            strand_behavior,
        )
        if validated_strand_behavior == "ignore":
            msg = "upstream/downstream nearest requires valid strand info on both PyRanges"
            raise ValueError(msg)
        if direction not in {"upstream", "downstream"}:
            msg = "direction must be one of 'any', 'upstream', or 'downstream'"
            raise ValueError(msg)

        with_index = PyRanges(self.df.with_row_index(TEMP_INDEX_COL))
        prepared_other, by = _prepare_by_binary(
            with_index,
            other_gr,
            strand_behavior=validated_strand_behavior,
            match_by=match_by,
        )
        fwd_self, rev_self = _split_on_strand(with_index)

        forward_direction = "forward" if direction == "downstream" else "backward"
        reverse_direction = "backward" if direction == "downstream" else "forward"

        frames: list[pl.DataFrame] = []
        if fwd_self.df.height > 0:
            frames.append(
                nearest_ranges(
                    fwd_self.df,
                    prepared_other.df,
                    match_by=by,
                    suffix=suffix,
                    exclude_overlaps=exclude_overlaps,
                    k=k,
                    dist_col=dist_col,
                    direction=forward_direction,
                    preserve_input_order=preserve_input_order,
                )
            )
        if rev_self.df.height > 0:
            frames.append(
                nearest_ranges(
                    rev_self.df,
                    prepared_other.df,
                    match_by=by,
                    suffix=suffix,
                    exclude_overlaps=exclude_overlaps,
                    k=k,
                    dist_col=dist_col,
                    direction=reverse_direction,
                    preserve_input_order=preserve_input_order,
                )
            )

        if not frames:
            return PyRanges(with_index.df.head(0).drop(TEMP_INDEX_COL))

        combined = frames[0] if len(frames) == 1 else pl.concat(frames)
        if preserve_input_order:
            combined = combined.sort(TEMP_INDEX_COL)
        return PyRanges(combined.drop(TEMP_INDEX_COL))

    def overlap(
        self,
        other: Any,
        strand_behavior: str = "auto",
        slack: int = 0,
        *,
        multiple: bool = False,
        contained_intervals_only: bool = False,
        match_by: str | Iterable[str] | None = None,
        invert: bool = False,
        preserve_input_order: bool = True,
    ) -> "PyRanges":
        other_gr = _ensure_pyranges(other)
        prepared_other, by = _prepare_by_binary(
            self,
            other_gr,
            strand_behavior=strand_behavior,
            match_by=match_by,
        )
        multiple_arg = "all" if multiple else "first"

        if invert:
            left_idx, _ = range_overlap_pairs(
                self.df,
                prepared_other.df,
                multiple=multiple_arg,
                slack=slack,
                contained_intervals_only=contained_intervals_only,
                match_by=by,
                preserve_input_order=preserve_input_order,
            )
            overlapping = set(left_idx)
            indexed = self.df.with_row_index(TEMP_INDEX_COL)
            result = indexed.filter(
                ~pl.col(TEMP_INDEX_COL).is_in(sorted(overlapping))
            ).drop(TEMP_INDEX_COL)
            return PyRanges(result)

        return PyRanges(
            range_overlap(
                self.df,
                prepared_other.df,
                multiple=multiple_arg,
                slack=slack,
                contained_intervals_only=contained_intervals_only,
                match_by=by,
                preserve_input_order=preserve_input_order,
            )
        )

    def subtract_overlaps(
        self,
        other: Any,
        strand_behavior: str = "auto",
        *,
        match_by: str | Iterable[str] | None = None,
        preserve_input_order: bool = True,
    ) -> "PyRanges":
        other_gr = _ensure_pyranges(other)
        prepared_other, by = _prepare_by_binary(
            self,
            other_gr,
            strand_behavior=strand_behavior,
            match_by=match_by,
        )
        return PyRanges(
            subtract_overlaps(
                self.df,
                prepared_other.df,
                match_by=by,
                preserve_input_order=preserve_input_order,
            )
        )

