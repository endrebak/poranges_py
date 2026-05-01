use std::collections::{BTreeMap, HashMap};

use polaranges::{
    benchmark, BioClipOptions, BioClusterOptions, BioComplementRangesOptions,
    BioCountOverlapsOptions, BioDataFrameRanges, BioExtendOptions, BioGroupCumsumOptions,
    BioIntersectOverlapsOptions, BioJoinOverlapsOptions, BioMaxDisjointOptions, BioMergeOptions,
    BioOuterRangesOptions, BioSetIntersectOverlapsOptions, BioSetUnionOverlapsOptions,
    BioSortOptions, BioSplitOptions, BioSubtractOptions, BioTileOptions, ClipRangesOptions,
    ClusterOptions, ComplementRangesOptions, CountOverlapsOptions, DataFrameIntervalAccessors,
    DataFrameRanges, ExecReport, ExtendRangesOptions, GenomicNearestDirection,
    GenomicNearestOptions, GenomicOverlapOptions, GroupCumsumOptions, IntersectOverlapsOptions,
    JoinOverlapsOptions, JoinType, MaxDisjointOptions, MergeOptions, NearestDirection,
    NearestOptions, OuterRangesOptions, OverlapMode, OverlapOptions, ParallelConfig, ProfileConfig,
    SetIntersectOverlapsOptions, SetUnionOverlapsOptions, SortRangesOptions, SplitOverlapsOptions,
    StrandBehavior, SubtractOptions, TileRangesOptions, Timings, UseStrand,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PySeries};

fn parse_overlap_mode(value: &str) -> PyResult<OverlapMode> {
    match value {
        "all" => Ok(OverlapMode::All),
        "first" => Ok(OverlapMode::First),
        "last" => Ok(OverlapMode::Last),
        other => Err(PyValueError::new_err(format!(
            "multiple must be one of 'all', 'first', or 'last'; got {other:?}",
        ))),
    }
}

fn parse_nearest_direction(value: &str) -> PyResult<NearestDirection> {
    match value {
        "any" => Ok(NearestDirection::Any),
        "forward" => Ok(NearestDirection::Forward),
        "backward" => Ok(NearestDirection::Backward),
        other => Err(PyValueError::new_err(format!(
            "direction must be one of 'any', 'forward', or 'backward'; got {other:?}",
        ))),
    }
}

fn parse_join_type(value: &str) -> PyResult<JoinType> {
    match value {
        "inner" => Ok(JoinType::Inner),
        "left" => Ok(JoinType::Left),
        "right" => Ok(JoinType::Right),
        "outer" => Ok(JoinType::Outer),
        other => Err(PyValueError::new_err(format!(
            "join_type must be one of 'inner', 'left', 'right', or 'outer'; got {other:?}",
        ))),
    }
}

fn parse_use_strand(value: &str) -> PyResult<UseStrand> {
    match value {
        "auto" => Ok(UseStrand::Auto),
        "enabled" => Ok(UseStrand::Enabled),
        "disabled" => Ok(UseStrand::Disabled),
        other => Err(PyValueError::new_err(format!(
            "use_strand must be one of 'auto', 'enabled', or 'disabled'; got {other:?}",
        ))),
    }
}

fn parse_strand_behavior(value: &str) -> PyResult<StrandBehavior> {
    match value {
        "auto" => Ok(StrandBehavior::Auto),
        "same" => Ok(StrandBehavior::Same),
        "opposite" => Ok(StrandBehavior::Opposite),
        "ignore" => Ok(StrandBehavior::Ignore),
        other => Err(PyValueError::new_err(format!(
            "strand_behavior must be one of 'auto', 'same', 'opposite', or 'ignore'; got {other:?}",
        ))),
    }
}

fn parse_bio_nearest_direction(value: &str) -> PyResult<GenomicNearestDirection> {
    match value {
        "any" => Ok(GenomicNearestDirection::Any),
        "upstream" => Ok(GenomicNearestDirection::Upstream),
        "downstream" => Ok(GenomicNearestDirection::Downstream),
        other => Err(PyValueError::new_err(format!(
            "direction must be one of 'any', 'upstream', or 'downstream'; got {other:?}",
        ))),
    }
}

fn build_parallel_config(
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> ParallelConfig {
    ParallelConfig {
        enabled: parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    }
}

fn overlap_options(
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    parallel: ParallelConfig,
    profile_enabled: bool,
) -> PyResult<OverlapOptions> {
    Ok(OverlapOptions {
        multiple: parse_overlap_mode(multiple)?,
        slack,
        contained_intervals_only,
        match_by: match_by.unwrap_or_default(),
        preserve_input_order,
        parallel,
        profile: ProfileConfig {
            enabled: profile_enabled,
        },
    })
}

fn nearest_options(
    match_by: Option<Vec<String>>,
    suffix: String,
    exclude_overlaps: bool,
    k: usize,
    dist_col: Option<String>,
    direction: &str,
    preserve_input_order: bool,
    parallel: ParallelConfig,
    profile_enabled: bool,
) -> PyResult<NearestOptions> {
    Ok(NearestOptions {
        match_by: match_by.unwrap_or_default(),
        suffix,
        exclude_overlaps,
        k,
        distance_column: dist_col,
        direction: parse_nearest_direction(direction)?,
        preserve_input_order,
        parallel,
        profile: ProfileConfig {
            enabled: profile_enabled,
        },
    })
}

fn bio_merge_options(
    use_strand: &str,
    count_col: Option<String>,
    match_by: Option<Vec<String>>,
    slack: i64,
) -> PyResult<BioMergeOptions> {
    Ok(BioMergeOptions {
        use_strand: parse_use_strand(use_strand)?,
        count_column: count_col,
        match_by: match_by.unwrap_or_default(),
        slack,
        ..BioMergeOptions::default()
    })
}

fn bio_cluster_options(
    use_strand: &str,
    match_by: Option<Vec<String>>,
    cluster_column: String,
    slack: i64,
) -> PyResult<BioClusterOptions> {
    Ok(BioClusterOptions {
        use_strand: parse_use_strand(use_strand)?,
        match_by: match_by.unwrap_or_default(),
        cluster_column,
        slack,
        ..BioClusterOptions::default()
    })
}

fn bio_overlap_options(
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    strand_behavior: &str,
    invert: bool,
    parallel: ParallelConfig,
    profile_enabled: bool,
) -> PyResult<GenomicOverlapOptions> {
    Ok(GenomicOverlapOptions {
        multiple: parse_overlap_mode(multiple)?,
        slack,
        contained_intervals_only,
        match_by: match_by
            .unwrap_or_default()
            .into_iter()
            .map(Into::into)
            .collect(),
        preserve_input_order,
        strand_behavior: parse_strand_behavior(strand_behavior)?,
        invert,
        parallel,
        profile: ProfileConfig {
            enabled: profile_enabled,
        },
        ..GenomicOverlapOptions::default()
    })
}

fn bio_nearest_options(
    match_by: Option<Vec<String>>,
    suffix: String,
    exclude_overlaps: bool,
    k: usize,
    dist_col: Option<String>,
    direction: &str,
    preserve_input_order: bool,
    strand_behavior: &str,
    parallel: ParallelConfig,
    profile_enabled: bool,
) -> PyResult<GenomicNearestOptions> {
    Ok(GenomicNearestOptions {
        strand_behavior: parse_strand_behavior(strand_behavior)?,
        match_by: match_by
            .unwrap_or_default()
            .into_iter()
            .map(Into::into)
            .collect(),
        suffix,
        exclude_overlaps,
        k,
        distance_column: dist_col,
        direction: parse_bio_nearest_direction(direction)?,
        preserve_input_order,
        parallel,
        profile: ProfileConfig {
            enabled: profile_enabled,
        },
        ..GenomicNearestOptions::default()
    })
}

fn bio_subtract_options(
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    strand_behavior: &str,
) -> PyResult<BioSubtractOptions> {
    Ok(BioSubtractOptions {
        strand_behavior: parse_strand_behavior(strand_behavior)?,
        match_by: match_by.unwrap_or_default(),
        preserve_input_order,
        ..BioSubtractOptions::default()
    })
}

fn join_overlaps_options(
    multiple: &str,
    slack: i64,
    match_by: Option<Vec<String>>,
    suffix: String,
    contained_intervals_only: bool,
    join_type: &str,
    report_overlap_column: Option<String>,
    preserve_input_order: bool,
) -> PyResult<JoinOverlapsOptions> {
    Ok(JoinOverlapsOptions {
        match_by: match_by.unwrap_or_default(),
        multiple: parse_overlap_mode(multiple)?,
        slack,
        suffix,
        contained_intervals_only,
        join_type: parse_join_type(join_type)?,
        report_overlap_column,
        preserve_input_order,
    })
}

fn intersect_overlaps_options(
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
) -> PyResult<IntersectOverlapsOptions> {
    Ok(IntersectOverlapsOptions {
        match_by: match_by.unwrap_or_default(),
        multiple: parse_overlap_mode(multiple)?,
        slack,
        contained_intervals_only,
        preserve_input_order,
    })
}

fn set_intersect_overlaps_options(
    multiple: &str,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
) -> PyResult<SetIntersectOverlapsOptions> {
    Ok(SetIntersectOverlapsOptions {
        match_by: match_by.unwrap_or_default(),
        multiple: parse_overlap_mode(multiple)?,
        preserve_input_order,
    })
}

fn bio_count_overlaps_options(
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    slack: i64,
    overlap_column: String,
) -> PyResult<BioCountOverlapsOptions> {
    Ok(BioCountOverlapsOptions {
        strand_behavior: parse_strand_behavior(strand_behavior)?,
        match_by: match_by.unwrap_or_default(),
        slack,
        overlap_column,
        ..BioCountOverlapsOptions::default()
    })
}

fn bio_join_overlaps_options(
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    multiple: &str,
    slack: i64,
    suffix: String,
    contained_intervals_only: bool,
    join_type: &str,
    report_overlap_column: Option<String>,
    preserve_input_order: bool,
) -> PyResult<BioJoinOverlapsOptions> {
    Ok(BioJoinOverlapsOptions {
        strand_behavior: parse_strand_behavior(strand_behavior)?,
        match_by: match_by.unwrap_or_default(),
        multiple: parse_overlap_mode(multiple)?,
        slack,
        suffix,
        contained_intervals_only,
        join_type: parse_join_type(join_type)?,
        report_overlap_column,
        preserve_input_order,
        ..BioJoinOverlapsOptions::default()
    })
}

fn bio_intersect_overlaps_options(
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    multiple: &str,
    preserve_input_order: bool,
) -> PyResult<BioIntersectOverlapsOptions> {
    Ok(BioIntersectOverlapsOptions {
        strand_behavior: parse_strand_behavior(strand_behavior)?,
        match_by: match_by.unwrap_or_default(),
        multiple: parse_overlap_mode(multiple)?,
        preserve_input_order,
        ..BioIntersectOverlapsOptions::default()
    })
}

fn bio_set_intersect_overlaps_options(
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    multiple: &str,
    preserve_input_order: bool,
) -> PyResult<BioSetIntersectOverlapsOptions> {
    Ok(BioSetIntersectOverlapsOptions {
        strand_behavior: parse_strand_behavior(strand_behavior)?,
        match_by: match_by.unwrap_or_default(),
        multiple: parse_overlap_mode(multiple)?,
        preserve_input_order,
        ..BioSetIntersectOverlapsOptions::default()
    })
}

fn bio_set_union_overlaps_options(
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
) -> PyResult<BioSetUnionOverlapsOptions> {
    Ok(BioSetUnionOverlapsOptions {
        strand_behavior: parse_strand_behavior(strand_behavior)?,
        match_by: match_by.unwrap_or_default(),
        ..BioSetUnionOverlapsOptions::default()
    })
}

fn to_py_err(err: polaranges::RangeFrameError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn duration_seconds(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64()
}

fn timings_to_map(timings: &Timings) -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("total".to_string(), duration_seconds(timings.total)),
        ("prepare".to_string(), duration_seconds(timings.prepare)),
        ("rechunk".to_string(), duration_seconds(timings.rechunk)),
        (
            "factorization".to_string(),
            duration_seconds(timings.factorization),
        ),
        (
            "group_task_build".to_string(),
            duration_seconds(timings.group_task_build),
        ),
        (
            "batch_build".to_string(),
            duration_seconds(timings.batch_build),
        ),
        ("kernel".to_string(), duration_seconds(timings.kernel)),
        (
            "reconstruction".to_string(),
            duration_seconds(timings.reconstruction),
        ),
    ])
}

fn overlap_pairs_report_tuple(
    report: ExecReport<polaranges::OverlapPairs>,
) -> (Vec<u32>, Vec<u32>, BTreeMap<String, f64>) {
    let timings = report.timings.unwrap_or_default();
    (
        report.result.left,
        report.result.right,
        timings_to_map(&timings),
    )
}

fn dataframe_report_tuple(
    report: ExecReport<polars_core::prelude::DataFrame>,
) -> (PyDataFrame, BTreeMap<String, f64>) {
    let timings = report.timings.unwrap_or_default();
    (PyDataFrame(report.result), timings_to_map(&timings))
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="all",
        slack=0,
        *,
        contained_intervals_only=false,
        match_by=None,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn overlap_range_pairs(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let options = overlap_options(
        multiple,
        slack,
        contained_intervals_only,
        match_by,
        preserve_input_order,
        build_parallel_config(
            parallel_enabled,
            force_parallel,
            target_batch_weight,
            min_total_weight,
            min_num_groups,
            min_num_batches,
            max_dominance_ratio,
        ),
        false,
    )?;

    let pairs = left
        .0
        .range_overlap_pairs(&right.0, options)
        .map_err(to_py_err)?;

    Ok((pairs.left, pairs.right))
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="all",
        slack=0,
        *,
        contained_intervals_only=false,
        match_by=None,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn overlap_range_pairs_report(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(Vec<u32>, Vec<u32>, BTreeMap<String, f64>)> {
    let options = overlap_options(
        multiple,
        slack,
        contained_intervals_only,
        match_by,
        preserve_input_order,
        build_parallel_config(
            parallel_enabled,
            force_parallel,
            target_batch_weight,
            min_total_weight,
            min_num_groups,
            min_num_batches,
            max_dominance_ratio,
        ),
        true,
    )?;

    let report = left
        .0
        .range_overlap_pairs_report(&right.0, options)
        .map_err(to_py_err)?;

    Ok(overlap_pairs_report_tuple(report))
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="all",
        slack=0,
        *,
        contained_intervals_only=false,
        match_by=None,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn overlap_ranges(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<PyDataFrame> {
    let options = overlap_options(
        multiple,
        slack,
        contained_intervals_only,
        match_by,
        preserve_input_order,
        build_parallel_config(
            parallel_enabled,
            force_parallel,
            target_batch_weight,
            min_total_weight,
            min_num_groups,
            min_num_batches,
            max_dominance_ratio,
        ),
        false,
    )?;

    let out = left.0.range_overlap(&right.0, options).map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="all",
        slack=0,
        *,
        contained_intervals_only=false,
        match_by=None,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn overlap_ranges_report(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(PyDataFrame, BTreeMap<String, f64>)> {
    let options = overlap_options(
        multiple,
        slack,
        contained_intervals_only,
        match_by,
        preserve_input_order,
        build_parallel_config(
            parallel_enabled,
            force_parallel,
            target_batch_weight,
            min_total_weight,
            min_num_groups,
            min_num_batches,
            max_dominance_ratio,
        ),
        true,
    )?;

    let report = left
        .0
        .range_overlap_report(&right.0, options)
        .map_err(to_py_err)?;
    Ok(dataframe_report_tuple(report))
}

#[pyfunction(signature = (df, *, use_strand="auto", count_col=None, match_by=None, slack=0))]
fn bio_merge_overlaps(
    df: PyDataFrame,
    use_strand: &str,
    count_col: Option<String>,
    match_by: Option<Vec<String>>,
    slack: i64,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_merge_overlaps(bio_merge_options(use_strand, count_col, match_by, slack)?)
            .map_err(to_py_err)?;

    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, count_col=None, match_by=None, slack=0))]
fn merge_overlaps(
    df: PyDataFrame,
    count_col: Option<String>,
    match_by: Option<Vec<String>>,
    slack: i64,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.merge_overlaps(MergeOptions {
            count_column: count_col,
            match_by: match_by.unwrap_or_default(),
            slack,
        })
        .map_err(to_py_err)?;

    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, use_strand="auto", match_by=None, cluster_column="Cluster".to_string(), slack=0))]
fn bio_cluster_overlaps(
    df: PyDataFrame,
    use_strand: &str,
    match_by: Option<Vec<String>>,
    cluster_column: String,
    slack: i64,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_cluster_overlaps(bio_cluster_options(
            use_strand,
            match_by,
            cluster_column,
            slack,
        )?)
        .map_err(to_py_err)?;

    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, match_by=None, cluster_column="Cluster".to_string(), slack=0))]
fn cluster_overlaps(
    df: PyDataFrame,
    match_by: Option<Vec<String>>,
    cluster_column: String,
    slack: i64,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.cluster_overlaps(ClusterOptions {
            match_by: match_by.unwrap_or_default(),
            cluster_column,
            slack,
        })
        .map_err(to_py_err)?;

    Ok(PyDataFrame(out))
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        strand_behavior="auto",
        match_by=None,
        suffix="_b".to_string(),
        exclude_overlaps=false,
        k=1,
        dist_col=Some("Distance".to_string()),
        direction="any",
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn bio_nearest_ranges(
    left: PyDataFrame,
    right: PyDataFrame,
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    suffix: String,
    exclude_overlaps: bool,
    k: usize,
    dist_col: Option<String>,
    direction: &str,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .g()
        .nearest_with(
            &right.0,
            bio_nearest_options(
                match_by,
                suffix,
                exclude_overlaps,
                k,
                dist_col,
                direction,
                preserve_input_order,
                strand_behavior,
                build_parallel_config(
                    parallel_enabled,
                    force_parallel,
                    target_batch_weight,
                    min_total_weight,
                    min_num_groups,
                    min_num_batches,
                    max_dominance_ratio,
                ),
                false,
            )?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        strand_behavior="auto",
        match_by=None,
        suffix="_b".to_string(),
        exclude_overlaps=false,
        k=1,
        dist_col=Some("Distance".to_string()),
        direction="any",
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn bio_nearest_ranges_report(
    left: PyDataFrame,
    right: PyDataFrame,
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    suffix: String,
    exclude_overlaps: bool,
    k: usize,
    dist_col: Option<String>,
    direction: &str,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(PyDataFrame, BTreeMap<String, f64>)> {
    let report = left
        .0
        .g()
        .nearest_report_with(
            &right.0,
            bio_nearest_options(
                match_by,
                suffix,
                exclude_overlaps,
                k,
                dist_col,
                direction,
                preserve_input_order,
                strand_behavior,
                build_parallel_config(
                    parallel_enabled,
                    force_parallel,
                    target_batch_weight,
                    min_total_weight,
                    min_num_groups,
                    min_num_batches,
                    max_dominance_ratio,
                ),
                true,
            )?,
        )
        .map_err(to_py_err)?;
    Ok(dataframe_report_tuple(report))
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        match_by=None,
        suffix="_b".to_string(),
        exclude_overlaps=false,
        k=1,
        dist_col=Some("Distance".to_string()),
        direction="any",
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn nearest_ranges(
    left: PyDataFrame,
    right: PyDataFrame,
    match_by: Option<Vec<String>>,
    suffix: String,
    exclude_overlaps: bool,
    k: usize,
    dist_col: Option<String>,
    direction: &str,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<PyDataFrame> {
    let options = nearest_options(
        match_by,
        suffix,
        exclude_overlaps,
        k,
        dist_col,
        direction,
        preserve_input_order,
        build_parallel_config(
            parallel_enabled,
            force_parallel,
            target_batch_weight,
            min_total_weight,
            min_num_groups,
            min_num_batches,
            max_dominance_ratio,
        ),
        false,
    )?;

    let out = left
        .0
        .nearest_ranges(&right.0, options)
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        match_by=None,
        suffix="_b".to_string(),
        exclude_overlaps=false,
        k=1,
        dist_col=Some("Distance".to_string()),
        direction="any",
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn nearest_ranges_report(
    left: PyDataFrame,
    right: PyDataFrame,
    match_by: Option<Vec<String>>,
    suffix: String,
    exclude_overlaps: bool,
    k: usize,
    dist_col: Option<String>,
    direction: &str,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(PyDataFrame, BTreeMap<String, f64>)> {
    let options = nearest_options(
        match_by,
        suffix,
        exclude_overlaps,
        k,
        dist_col,
        direction,
        preserve_input_order,
        build_parallel_config(
            parallel_enabled,
            force_parallel,
            target_batch_weight,
            min_total_weight,
            min_num_groups,
            min_num_batches,
            max_dominance_ratio,
        ),
        true,
    )?;

    let report = left
        .0
        .nearest_ranges_report(&right.0, options)
        .map_err(to_py_err)?;
    Ok(dataframe_report_tuple(report))
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="first",
        slack=0,
        *,
        strand_behavior="auto",
        contained_intervals_only=false,
        match_by=None,
        invert=false,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn bio_overlap_ranges(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    strand_behavior: &str,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    invert: bool,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .g()
        .overlap_with(
            &right.0,
            bio_overlap_options(
                multiple,
                slack,
                contained_intervals_only,
                match_by,
                preserve_input_order,
                strand_behavior,
                invert,
                build_parallel_config(
                    parallel_enabled,
                    force_parallel,
                    target_batch_weight,
                    min_total_weight,
                    min_num_groups,
                    min_num_batches,
                    max_dominance_ratio,
                ),
                false,
            )?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="first",
        slack=0,
        *,
        strand_behavior="auto",
        contained_intervals_only=false,
        match_by=None,
        invert=false,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn bio_overlap_ranges_report(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    strand_behavior: &str,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    invert: bool,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(PyDataFrame, BTreeMap<String, f64>)> {
    let report = left
        .0
        .g()
        .overlap_report_with(
            &right.0,
            bio_overlap_options(
                multiple,
                slack,
                contained_intervals_only,
                match_by,
                preserve_input_order,
                strand_behavior,
                invert,
                build_parallel_config(
                    parallel_enabled,
                    force_parallel,
                    target_batch_weight,
                    min_total_weight,
                    min_num_groups,
                    min_num_batches,
                    max_dominance_ratio,
                ),
                true,
            )?,
        )
        .map_err(to_py_err)?;
    Ok(dataframe_report_tuple(report))
}

#[pyfunction(signature = (left, right, *, strand_behavior="auto", match_by=None, preserve_input_order=true))]
fn bio_subtract_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .bio_subtract_overlaps(
            &right.0,
            bio_subtract_options(match_by, preserve_input_order, strand_behavior)?,
        )
        .map_err(to_py_err)?;

    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, strand_col="Strand"))]
fn bio_has_valid_strand(df: PyDataFrame, strand_col: &str) -> bool {
    df.0.bio_has_valid_strand(Some(strand_col))
}

#[pyfunction(signature = (left, right, *, match_by=None, preserve_input_order=true))]
fn subtract_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .subtract_overlaps(
            &right.0,
            SubtractOptions {
                match_by: match_by.unwrap_or_default(),
                preserve_input_order,
            },
        )
        .map_err(to_py_err)?;

    Ok(PyDataFrame(out))
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="all",
        slack=0,
        *,
        contained_intervals_only=false,
        match_by=None,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn overlap_pairs(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(Vec<u32>, Vec<u32>)> {
    overlap_range_pairs(
        left,
        right,
        multiple,
        slack,
        contained_intervals_only,
        match_by,
        preserve_input_order,
        parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    )
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="all",
        slack=0,
        *,
        contained_intervals_only=false,
        match_by=None,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn overlap_pairs_report(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(Vec<u32>, Vec<u32>, BTreeMap<String, f64>)> {
    overlap_range_pairs_report(
        left,
        right,
        multiple,
        slack,
        contained_intervals_only,
        match_by,
        preserve_input_order,
        parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    )
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="all",
        slack=0,
        *,
        contained_intervals_only=false,
        match_by=None,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn overlap(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<PyDataFrame> {
    overlap_ranges(
        left,
        right,
        multiple,
        slack,
        contained_intervals_only,
        match_by,
        preserve_input_order,
        parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    )
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="all",
        slack=0,
        *,
        contained_intervals_only=false,
        match_by=None,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn overlap_report(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(PyDataFrame, BTreeMap<String, f64>)> {
    overlap_ranges_report(
        left,
        right,
        multiple,
        slack,
        contained_intervals_only,
        match_by,
        preserve_input_order,
        parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    )
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        match_by=None,
        suffix="_b".to_string(),
        exclude_overlaps=false,
        k=1,
        dist_col=Some("Distance".to_string()),
        direction="any",
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn nearest(
    left: PyDataFrame,
    right: PyDataFrame,
    match_by: Option<Vec<String>>,
    suffix: String,
    exclude_overlaps: bool,
    k: usize,
    dist_col: Option<String>,
    direction: &str,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<PyDataFrame> {
    nearest_ranges(
        left,
        right,
        match_by,
        suffix,
        exclude_overlaps,
        k,
        dist_col,
        direction,
        preserve_input_order,
        parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    )
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        match_by=None,
        suffix="_b".to_string(),
        exclude_overlaps=false,
        k=1,
        dist_col=Some("Distance".to_string()),
        direction="any",
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn nearest_report(
    left: PyDataFrame,
    right: PyDataFrame,
    match_by: Option<Vec<String>>,
    suffix: String,
    exclude_overlaps: bool,
    k: usize,
    dist_col: Option<String>,
    direction: &str,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(PyDataFrame, BTreeMap<String, f64>)> {
    nearest_ranges_report(
        left,
        right,
        match_by,
        suffix,
        exclude_overlaps,
        k,
        dist_col,
        direction,
        preserve_input_order,
        parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    )
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="first",
        slack=0,
        *,
        strand_behavior="auto",
        contained_intervals_only=false,
        match_by=None,
        invert=false,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn bio_overlap(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    strand_behavior: &str,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    invert: bool,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<PyDataFrame> {
    bio_overlap_ranges(
        left,
        right,
        multiple,
        slack,
        strand_behavior,
        contained_intervals_only,
        match_by,
        invert,
        preserve_input_order,
        parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    )
}

#[pyfunction(
    signature = (
        left,
        right,
        multiple="first",
        slack=0,
        *,
        strand_behavior="auto",
        contained_intervals_only=false,
        match_by=None,
        invert=false,
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn bio_overlap_report(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    strand_behavior: &str,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    invert: bool,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(PyDataFrame, BTreeMap<String, f64>)> {
    bio_overlap_ranges_report(
        left,
        right,
        multiple,
        slack,
        strand_behavior,
        contained_intervals_only,
        match_by,
        invert,
        preserve_input_order,
        parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    )
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        strand_behavior="auto",
        match_by=None,
        suffix="_b".to_string(),
        exclude_overlaps=false,
        k=1,
        dist_col=Some("Distance".to_string()),
        direction="any",
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn bio_nearest(
    left: PyDataFrame,
    right: PyDataFrame,
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    suffix: String,
    exclude_overlaps: bool,
    k: usize,
    dist_col: Option<String>,
    direction: &str,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<PyDataFrame> {
    bio_nearest_ranges(
        left,
        right,
        strand_behavior,
        match_by,
        suffix,
        exclude_overlaps,
        k,
        dist_col,
        direction,
        preserve_input_order,
        parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    )
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        strand_behavior="auto",
        match_by=None,
        suffix="_b".to_string(),
        exclude_overlaps=false,
        k=1,
        dist_col=Some("Distance".to_string()),
        direction="any",
        preserve_input_order=true,
        parallel_enabled=false,
        force_parallel=false,
        target_batch_weight=16384,
        min_total_weight=65536,
        min_num_groups=64,
        min_num_batches=2,
        max_dominance_ratio=0.85
    )
)]
fn bio_nearest_report(
    left: PyDataFrame,
    right: PyDataFrame,
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    suffix: String,
    exclude_overlaps: bool,
    k: usize,
    dist_col: Option<String>,
    direction: &str,
    preserve_input_order: bool,
    parallel_enabled: bool,
    force_parallel: bool,
    target_batch_weight: usize,
    min_total_weight: usize,
    min_num_groups: usize,
    min_num_batches: usize,
    max_dominance_ratio: f64,
) -> PyResult<(PyDataFrame, BTreeMap<String, f64>)> {
    bio_nearest_ranges_report(
        left,
        right,
        strand_behavior,
        match_by,
        suffix,
        exclude_overlaps,
        k,
        dist_col,
        direction,
        preserve_input_order,
        parallel_enabled,
        force_parallel,
        target_batch_weight,
        min_total_weight,
        min_num_groups,
        min_num_batches,
        max_dominance_ratio,
    )
}

#[pyfunction(signature = (left, right, *, match_by=None, slack=0))]
fn count_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    match_by: Option<Vec<String>>,
    slack: i64,
) -> PyResult<PySeries> {
    let out = left
        .0
        .count_overlaps(
            &right.0,
            CountOverlapsOptions {
                match_by: match_by.unwrap_or_default(),
                slack,
            },
        )
        .map_err(to_py_err)?;
    Ok(PySeries(out))
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        match_by=None,
        multiple="all",
        slack=0,
        suffix="_b".to_string(),
        contained_intervals_only=false,
        join_type="inner",
        report_overlap_column=None,
        preserve_input_order=true
    )
)]
fn join_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    match_by: Option<Vec<String>>,
    multiple: &str,
    slack: i64,
    suffix: String,
    contained_intervals_only: bool,
    join_type: &str,
    report_overlap_column: Option<String>,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .join_overlaps(
            &right.0,
            join_overlaps_options(
                multiple,
                slack,
                match_by,
                suffix,
                contained_intervals_only,
                join_type,
                report_overlap_column,
                preserve_input_order,
            )?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        match_by=None,
        multiple="all",
        slack=0,
        contained_intervals_only=false,
        preserve_input_order=true
    )
)]
fn intersect_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    match_by: Option<Vec<String>>,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .intersect_overlaps(
            &right.0,
            intersect_overlaps_options(
                multiple,
                slack,
                contained_intervals_only,
                match_by,
                preserve_input_order,
            )?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (left, right, *, match_by=None, multiple="all", preserve_input_order=true))]
fn set_intersect_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    match_by: Option<Vec<String>>,
    multiple: &str,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .set_intersect_overlaps(
            &right.0,
            set_intersect_overlaps_options(multiple, match_by, preserve_input_order)?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (left, right, *, match_by=None))]
fn set_union_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    match_by: Option<Vec<String>>,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .set_union_overlaps(
            &right.0,
            SetUnionOverlapsOptions {
                match_by: match_by.unwrap_or_default(),
            },
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, match_by=None, negative_strand_column=None))]
fn sort_ranges(
    df: PyDataFrame,
    match_by: Option<Vec<String>>,
    negative_strand_column: Option<String>,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.sort_ranges(SortRangesOptions {
            match_by: match_by.unwrap_or_default(),
            negative_strand_column,
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, ext=None, *, match_by=None, ext_3=None, ext_5=None, negative_strand_column=None))]
fn extend_ranges(
    df: PyDataFrame,
    ext: Option<i64>,
    match_by: Option<Vec<String>>,
    ext_3: Option<i64>,
    ext_5: Option<i64>,
    negative_strand_column: Option<String>,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.extend_ranges(ExtendRangesOptions {
            match_by: match_by.unwrap_or_default(),
            ext,
            ext_3,
            ext_5,
            negative_strand_column,
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, tile_size, *, match_by=None, negative_strand_column=None, overlap_column=None))]
fn tile_ranges(
    df: PyDataFrame,
    tile_size: i64,
    match_by: Option<Vec<String>>,
    negative_strand_column: Option<String>,
    overlap_column: Option<String>,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.tile_ranges(TileRangesOptions {
            match_by: match_by.unwrap_or_default(),
            tile_size,
            negative_strand_column,
            overlap_column,
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, chromsizes=None, remove=false, only_right=false, group_sizes_col="Chromosome".to_string()))]
fn clip_ranges(
    df: PyDataFrame,
    chromsizes: Option<HashMap<String, i64>>,
    remove: bool,
    only_right: bool,
    group_sizes_col: String,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.clip_ranges(ClipRangesOptions {
            chromsizes: chromsizes.map(|values| values.into_iter().collect()),
            remove,
            only_right,
            group_sizes_col,
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, match_by=None, forward_strand_column=None, cumsum_start_column=None, cumsum_end_column=None, keep_order=true))]
fn group_cumsum(
    df: PyDataFrame,
    match_by: Option<Vec<String>>,
    forward_strand_column: Option<String>,
    cumsum_start_column: Option<String>,
    cumsum_end_column: Option<String>,
    keep_order: bool,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.group_cumsum(GroupCumsumOptions {
            match_by: match_by.unwrap_or_default(),
            forward_strand_column,
            cumsum_start_column,
            cumsum_end_column,
            keep_order,
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, match_by=None, slack=0, preserve_input_order=true))]
fn max_disjoint_overlaps(
    df: PyDataFrame,
    match_by: Option<Vec<String>>,
    slack: i64,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.max_disjoint_overlaps(MaxDisjointOptions {
            match_by: match_by.unwrap_or_default(),
            slack,
            preserve_input_order,
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, match_by=None, between=false))]
fn split_overlaps(
    df: PyDataFrame,
    match_by: Option<Vec<String>>,
    between: bool,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.split_overlaps(SplitOverlapsOptions {
            match_by: match_by.unwrap_or_default(),
            between,
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, match_by=None))]
fn outer_ranges(df: PyDataFrame, match_by: Option<Vec<String>>) -> PyResult<PyDataFrame> {
    let out =
        df.0.outer_ranges(OuterRangesOptions {
            match_by: match_by.unwrap_or_default(),
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, match_by=None, include_first_interval=false, chromsizes=None, group_sizes_col="Chromosome".to_string()))]
fn complement_ranges(
    df: PyDataFrame,
    match_by: Option<Vec<String>>,
    include_first_interval: bool,
    chromsizes: Option<HashMap<String, i64>>,
    group_sizes_col: String,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.complement_ranges(ComplementRangesOptions {
            match_by: match_by.unwrap_or_default(),
            include_first_interval,
            chromsizes: chromsizes.map(|values| values.into_iter().collect()),
            group_sizes_col,
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        strand_behavior="auto",
        match_by=None,
        slack=0,
        overlap_column="Count".to_string()
    )
)]
fn bio_count_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    slack: i64,
    overlap_column: String,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .bio_count_overlaps(
            &right.0,
            bio_count_overlaps_options(strand_behavior, match_by, slack, overlap_column)?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(
    signature = (
        left,
        right,
        *,
        strand_behavior="auto",
        match_by=None,
        multiple="all",
        slack=0,
        suffix="_b".to_string(),
        contained_intervals_only=false,
        join_type="inner",
        report_overlap_column=None,
        preserve_input_order=true
    )
)]
fn bio_join_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    multiple: &str,
    slack: i64,
    suffix: String,
    contained_intervals_only: bool,
    join_type: &str,
    report_overlap_column: Option<String>,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .bio_join_overlaps(
            &right.0,
            bio_join_overlaps_options(
                strand_behavior,
                match_by,
                multiple,
                slack,
                suffix,
                contained_intervals_only,
                join_type,
                report_overlap_column,
                preserve_input_order,
            )?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (left, right, *, strand_behavior="auto", match_by=None, multiple="all", preserve_input_order=true))]
fn bio_intersect_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    multiple: &str,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .bio_intersect_overlaps(
            &right.0,
            bio_intersect_overlaps_options(
                strand_behavior,
                match_by,
                multiple,
                preserve_input_order,
            )?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (left, right, *, strand_behavior="auto", match_by=None, multiple="all", preserve_input_order=true))]
fn bio_set_intersect_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
    multiple: &str,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .bio_set_intersect_overlaps(
            &right.0,
            bio_set_intersect_overlaps_options(
                strand_behavior,
                match_by,
                multiple,
                preserve_input_order,
            )?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (left, right, *, strand_behavior="auto", match_by=None))]
fn bio_set_union_overlaps(
    left: PyDataFrame,
    right: PyDataFrame,
    strand_behavior: &str,
    match_by: Option<Vec<String>>,
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .bio_set_union_overlaps(
            &right.0,
            bio_set_union_overlaps_options(strand_behavior, match_by)?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, use_strand="auto", match_by=None))]
fn bio_sort_ranges(
    df: PyDataFrame,
    use_strand: &str,
    match_by: Option<Vec<String>>,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_sort_ranges(BioSortOptions {
            use_strand: parse_use_strand(use_strand)?,
            match_by: match_by.unwrap_or_default(),
            ..BioSortOptions::default()
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, ext=None, *, use_strand="auto", match_by=None, ext_3=None, ext_5=None))]
fn bio_extend_ranges(
    df: PyDataFrame,
    ext: Option<i64>,
    use_strand: &str,
    match_by: Option<Vec<String>>,
    ext_3: Option<i64>,
    ext_5: Option<i64>,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_extend_ranges(BioExtendOptions {
            use_strand: parse_use_strand(use_strand)?,
            match_by: match_by.unwrap_or_default(),
            ext,
            ext_3,
            ext_5,
            ..BioExtendOptions::default()
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, tile_size, *, use_strand="disabled", match_by=None, overlap_column=None))]
fn bio_tile_ranges(
    df: PyDataFrame,
    tile_size: i64,
    use_strand: &str,
    match_by: Option<Vec<String>>,
    overlap_column: Option<String>,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_tile_ranges(BioTileOptions {
            use_strand: parse_use_strand(use_strand)?,
            match_by: match_by.unwrap_or_default(),
            tile_size,
            overlap_column,
            ..BioTileOptions::new(tile_size)
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, chromsizes=None, remove=false, only_right=false, chromosome_column="Chromosome".to_string()))]
fn bio_clip_ranges(
    df: PyDataFrame,
    chromsizes: Option<HashMap<String, i64>>,
    remove: bool,
    only_right: bool,
    chromosome_column: String,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_clip_ranges(BioClipOptions {
            chromsizes: chromsizes.map(|values| values.into_iter().collect()),
            remove,
            only_right,
            chromosome_column,
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, use_strand="auto", match_by=None, cumsum_start_column=None, cumsum_end_column=None, keep_order=true))]
fn bio_group_cumsum(
    df: PyDataFrame,
    use_strand: &str,
    match_by: Option<Vec<String>>,
    cumsum_start_column: Option<String>,
    cumsum_end_column: Option<String>,
    keep_order: bool,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_group_cumsum(BioGroupCumsumOptions {
            use_strand: parse_use_strand(use_strand)?,
            match_by: match_by.unwrap_or_default(),
            cumsum_start_column,
            cumsum_end_column,
            keep_order,
            ..BioGroupCumsumOptions::default()
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, use_strand="auto", match_by=None, slack=0, preserve_input_order=true))]
fn bio_max_disjoint_overlaps(
    df: PyDataFrame,
    use_strand: &str,
    match_by: Option<Vec<String>>,
    slack: i64,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_max_disjoint_overlaps(BioMaxDisjointOptions {
            use_strand: parse_use_strand(use_strand)?,
            match_by: match_by.unwrap_or_default(),
            slack,
            preserve_input_order,
            ..BioMaxDisjointOptions::default()
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, use_strand="auto", match_by=None, between=false))]
fn bio_split_overlaps(
    df: PyDataFrame,
    use_strand: &str,
    match_by: Option<Vec<String>>,
    between: bool,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_split_overlaps(BioSplitOptions {
            use_strand: parse_use_strand(use_strand)?,
            match_by: match_by.unwrap_or_default(),
            between,
            ..BioSplitOptions::default()
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, use_strand="auto", match_by=None))]
fn bio_outer_ranges(
    df: PyDataFrame,
    use_strand: &str,
    match_by: Option<Vec<String>>,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_outer_ranges(BioOuterRangesOptions {
            use_strand: parse_use_strand(use_strand)?,
            match_by: match_by.unwrap_or_default(),
            ..BioOuterRangesOptions::default()
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction(signature = (df, *, use_strand="auto", match_by=None, include_first_interval=false, chromsizes=None, group_sizes_col="Chromosome".to_string()))]
fn bio_complement_ranges(
    df: PyDataFrame,
    use_strand: &str,
    match_by: Option<Vec<String>>,
    include_first_interval: bool,
    chromsizes: Option<HashMap<String, i64>>,
    group_sizes_col: String,
) -> PyResult<PyDataFrame> {
    let out =
        df.0.bio_complement_ranges(BioComplementRangesOptions {
            use_strand: parse_use_strand(use_strand)?,
            match_by: match_by.unwrap_or_default(),
            include_first_interval,
            chromsizes: chromsizes.map(|values| values.into_iter().collect()),
            group_sizes_col,
            ..BioComplementRangesOptions::default()
        })
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
}

#[pyfunction]
fn benchmark_version() -> String {
    benchmark::benchmark_version_description()
}

#[pymodule]
fn _polaranges(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(benchmark_version, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_pairs_report, m)?)?;
    m.add_function(wrap_pyfunction!(overlap, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_report, m)?)?;
    m.add_function(wrap_pyfunction!(count_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(join_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(intersect_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(set_intersect_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(set_union_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(sort_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(extend_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(tile_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(clip_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(group_cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(max_disjoint_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(split_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(outer_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(complement_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(nearest, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_report, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_range_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_range_pairs_report, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_ranges_report, m)?)?;
    m.add_function(wrap_pyfunction!(bio_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(bio_overlap_report, m)?)?;
    m.add_function(wrap_pyfunction!(bio_merge_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(merge_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_cluster_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_count_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_join_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_intersect_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_set_intersect_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_set_union_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_sort_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_extend_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_tile_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_clip_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_group_cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(bio_max_disjoint_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_split_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_outer_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_complement_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_nearest, m)?)?;
    m.add_function(wrap_pyfunction!(bio_nearest_report, m)?)?;
    m.add_function(wrap_pyfunction!(bio_nearest_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_nearest_ranges_report, m)?)?;
    m.add_function(wrap_pyfunction!(bio_overlap_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_overlap_ranges_report, m)?)?;
    m.add_function(wrap_pyfunction!(bio_subtract_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_has_valid_strand, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_ranges_report, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_overlaps, m)?)?;
    Ok(())
}
