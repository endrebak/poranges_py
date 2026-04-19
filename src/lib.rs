use poranges::{
    ClusterOptions, DataFrameRanges, MergeOptions, NearestDirection, NearestOptions, OverlapMode,
    OverlapOptions, SubtractOptions,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

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

fn overlap_options(
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
) -> PyResult<OverlapOptions> {
    Ok(OverlapOptions {
        multiple: parse_overlap_mode(multiple)?,
        slack,
        contained_intervals_only,
        match_by: match_by.unwrap_or_default(),
        preserve_input_order,
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
) -> PyResult<NearestOptions> {
    Ok(NearestOptions {
        match_by: match_by.unwrap_or_default(),
        suffix,
        exclude_overlaps,
        k,
        distance_column: dist_col,
        direction: parse_nearest_direction(direction)?,
        preserve_input_order,
    })
}

fn to_py_err(err: poranges::RangeFrameError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction(signature = (left, right, multiple="all", slack=0, *, contained_intervals_only=false, match_by=None, preserve_input_order=true))]
fn range_overlap_pairs(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
) -> PyResult<(Vec<u32>, Vec<u32>)> {
    let options = overlap_options(
        multiple,
        slack,
        contained_intervals_only,
        match_by,
        preserve_input_order,
    )?;

    let pairs = left
        .0
        .range_overlap_pairs(&right.0, options)
        .map_err(to_py_err)?;

    Ok((pairs.left, pairs.right))
}

#[pyfunction(signature = (left, right, multiple="all", slack=0, *, contained_intervals_only=false, match_by=None, preserve_input_order=true))]
fn range_overlap(
    left: PyDataFrame,
    right: PyDataFrame,
    multiple: &str,
    slack: i64,
    contained_intervals_only: bool,
    match_by: Option<Vec<String>>,
    preserve_input_order: bool,
) -> PyResult<PyDataFrame> {
    let options = overlap_options(
        multiple,
        slack,
        contained_intervals_only,
        match_by,
        preserve_input_order,
    )?;

    let out = left.0.range_overlap(&right.0, options).map_err(to_py_err)?;
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

#[pyfunction(signature = (left, right, *, match_by=None, suffix="_b".to_string(), exclude_overlaps=false, k=1, dist_col=Some("Distance".to_string()), direction="any", preserve_input_order=true))]
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
) -> PyResult<PyDataFrame> {
    let options = nearest_options(
        match_by,
        suffix,
        exclude_overlaps,
        k,
        dist_col,
        direction,
        preserve_input_order,
    )?;

    let out = left
        .0
        .nearest_ranges(&right.0, options)
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
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
    m.add_function(wrap_pyfunction!(range_overlap_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(range_overlap, m)?)?;
    m.add_function(wrap_pyfunction!(merge_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_overlaps, m)?)?;
    Ok(())
}
