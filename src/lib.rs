use poranges::{
    BioClusterOptions, BioDataFrameRanges, BioMergeOptions, BioNearestDirection, BioNearestOptions,
    BioOverlapOptions, BioSubtractOptions, ClusterOptions, DataFrameRanges, MergeOptions,
    NearestDirection, NearestOptions, OverlapMode, OverlapOptions, StrandBehavior, SubtractOptions,
    UseStrand,
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

fn parse_bio_nearest_direction(value: &str) -> PyResult<BioNearestDirection> {
    match value {
        "any" => Ok(BioNearestDirection::Any),
        "upstream" => Ok(BioNearestDirection::Upstream),
        "downstream" => Ok(BioNearestDirection::Downstream),
        other => Err(PyValueError::new_err(format!(
            "direction must be one of 'any', 'upstream', or 'downstream'; got {other:?}",
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
) -> PyResult<BioOverlapOptions> {
    Ok(BioOverlapOptions {
        multiple: parse_overlap_mode(multiple)?,
        slack,
        contained_intervals_only,
        match_by: match_by.unwrap_or_default(),
        preserve_input_order,
        strand_behavior: parse_strand_behavior(strand_behavior)?,
        invert,
        ..BioOverlapOptions::default()
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
) -> PyResult<BioNearestOptions> {
    Ok(BioNearestOptions {
        strand_behavior: parse_strand_behavior(strand_behavior)?,
        match_by: match_by.unwrap_or_default(),
        suffix,
        exclude_overlaps,
        k,
        distance_column: dist_col,
        direction: parse_bio_nearest_direction(direction)?,
        preserve_input_order,
        ..BioNearestOptions::default()
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
fn overlap_range_pairs(
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
fn overlap_ranges(
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

#[pyfunction(signature = (left, right, *, strand_behavior="auto", match_by=None, suffix="_b".to_string(), exclude_overlaps=false, k=1, dist_col=Some("Distance".to_string()), direction="any", preserve_input_order=true))]
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
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .bio_nearest_ranges(
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
            )?,
        )
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

#[pyfunction(signature = (left, right, multiple="first", slack=0, *, strand_behavior="auto", contained_intervals_only=false, match_by=None, invert=false, preserve_input_order=true))]
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
) -> PyResult<PyDataFrame> {
    let out = left
        .0
        .bio_overlap_ranges(
            &right.0,
            bio_overlap_options(
                multiple,
                slack,
                contained_intervals_only,
                match_by,
                preserve_input_order,
                strand_behavior,
                invert,
            )?,
        )
        .map_err(to_py_err)?;
    Ok(PyDataFrame(out))
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
    m.add_function(wrap_pyfunction!(overlap_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_merge_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(merge_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_cluster_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_nearest_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_overlap_ranges, m)?)?;
    m.add_function(wrap_pyfunction!(bio_subtract_overlaps, m)?)?;
    m.add_function(wrap_pyfunction!(bio_has_valid_strand, m)?)?;
    m.add_function(wrap_pyfunction!(subtract_overlaps, m)?)?;
    Ok(())
}
