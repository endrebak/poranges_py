use std::collections::BTreeMap;

use poranges::{
    BioClusterOptions, BioDataFrameRanges, BioMergeOptions, BioSubtractOptions, ClusterOptions,
    DataFrameIntervalAccessors, DataFrameRanges, ExecReport, GenomicNearestDirection,
    GenomicNearestOptions,
    GenomicOverlapOptions, LazyFrameIntervalAccessors, LazyFrameRanges, MergeOptions,
    NearestDirection, NearestOptions, OverlapMode, OverlapOptions, ParallelConfig, ProfileConfig,
    StrandBehavior, SubtractOptions, Timings, UseStrand,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::{PyDataFrame, PyLazyFrame};

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

fn to_py_err(err: poranges::RangeFrameError) -> PyErr {
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
        ("batch_build".to_string(), duration_seconds(timings.batch_build)),
        ("kernel".to_string(), duration_seconds(timings.kernel)),
        (
            "reconstruction".to_string(),
            duration_seconds(timings.reconstruction),
        ),
    ])
}

fn overlap_pairs_report_tuple(
    report: ExecReport<poranges::OverlapPairs>,
) -> (Vec<u32>, Vec<u32>, BTreeMap<String, f64>) {
    let timings = report.timings.unwrap_or_default();
    (report.result.left, report.result.right, timings_to_map(&timings))
}

fn dataframe_report_tuple(
    report: ExecReport<polars_core::prelude::DataFrame>,
) -> (PyDataFrame, BTreeMap<String, f64>) {
    let timings = report.timings.unwrap_or_default();
    (PyDataFrame(report.result), timings_to_map(&timings))
}

fn lazyframe_report_tuple(
    report: ExecReport<polars_lazy::frame::LazyFrame>,
) -> (PyLazyFrame, BTreeMap<String, f64>) {
    let timings = report.timings.unwrap_or_default();
    (PyLazyFrame(report.result), timings_to_map(&timings))
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
fn overlap_ranges_lazy(
    left: PyLazyFrame,
    right: PyLazyFrame,
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
) -> PyResult<PyLazyFrame> {
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

    let out = left.0.range_overlap(right.0, options).map_err(to_py_err)?;
    Ok(PyLazyFrame(out))
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
fn overlap_ranges_lazy_report(
    left: PyLazyFrame,
    right: PyLazyFrame,
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
) -> PyResult<(PyLazyFrame, BTreeMap<String, f64>)> {
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
        .range_overlap_report(right.0, options)
        .map_err(to_py_err)?;
    Ok(lazyframe_report_tuple(report))
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
fn bio_nearest_ranges_lazy(
    left: PyLazyFrame,
    right: PyLazyFrame,
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
) -> PyResult<PyLazyFrame> {
    let out = left
        .0
        .g()
        .nearest_with(
            right.0,
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
    Ok(PyLazyFrame(out))
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
fn bio_nearest_ranges_lazy_report(
    left: PyLazyFrame,
    right: PyLazyFrame,
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
) -> PyResult<(PyLazyFrame, BTreeMap<String, f64>)> {
    let report = left
        .0
        .g()
        .nearest_report_with(
            right.0,
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
    Ok(lazyframe_report_tuple(report))
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

    let out = left.0.nearest_ranges(&right.0, options).map_err(to_py_err)?;
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
fn nearest_ranges_lazy(
    left: PyLazyFrame,
    right: PyLazyFrame,
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
) -> PyResult<PyLazyFrame> {
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

    let out = left.0.nearest_ranges(right.0, options).map_err(to_py_err)?;
    Ok(PyLazyFrame(out))
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
fn nearest_ranges_lazy_report(
    left: PyLazyFrame,
    right: PyLazyFrame,
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
) -> PyResult<(PyLazyFrame, BTreeMap<String, f64>)> {
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
        .nearest_ranges_report(right.0, options)
        .map_err(to_py_err)?;
    Ok(lazyframe_report_tuple(report))
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
fn bio_overlap_ranges_lazy(
    left: PyLazyFrame,
    right: PyLazyFrame,
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
) -> PyResult<PyLazyFrame> {
    let out = left
        .0
        .g()
        .overlap_with(
            right.0,
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
    Ok(PyLazyFrame(out))
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
fn bio_overlap_ranges_lazy_report(
    left: PyLazyFrame,
    right: PyLazyFrame,
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
) -> PyResult<(PyLazyFrame, BTreeMap<String, f64>)> {
    let report = left
        .0
        .g()
        .overlap_report_with(
            right.0,
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
    Ok(lazyframe_report_tuple(report))
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

#[pymodule]
fn _poranges(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(overlap_range_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_range_pairs_report, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_ranges_report, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_ranges_lazy, m)?)?;
    m.add_function(wrap_pyfunction!(overlap_ranges_lazy_report, m)?)?;
    m.add_function(wrap_pyfunction!(bio_merge_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(merge_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_cluster_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_nearest_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_nearest_ranges_report, m)?)?;
    m.add_function(wrap_pyfunction!(bio_nearest_ranges_lazy, m)?)?;
    m.add_function(wrap_pyfunction!(bio_nearest_ranges_lazy_report, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_ranges_report, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_ranges_lazy, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_ranges_lazy_report, m)?)?;
    m.add_function(wrap_pyfunction!(bio_overlap_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_overlap_ranges_report, m)?)?;
    m.add_function(wrap_pyfunction!(bio_overlap_ranges_lazy, m)?)?;
    m.add_function(wrap_pyfunction!(bio_overlap_ranges_lazy_report, m)?)?;
    m.add_function(wrap_pyfunction!(bio_subtract_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_has_valid_strand, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_overlaps, m)?)?;
    Ok(())
}
