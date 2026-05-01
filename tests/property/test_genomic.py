import polars as pl
from hypothesis import given

from .genomic import GenomicColumns, genomic_dataframes, genomic_records


@given(genomic_records(max_position=1000, max_interval_width=100, max_size=25))
def test_genomic_records_are_valid_half_open_intervals(records) -> None:
    for record in records:
        assert record["chromosome"]
        assert 0 <= record["start"] < record["end"] <= 1000
        assert record["end"] - record["start"] <= 100
        assert record["strand"] in {"+", "-", "."}
        assert record["gene"].startswith("gene_")
        assert record["transcript_id"].startswith("tx_")


@given(genomic_dataframes(min_size=1, max_size=25))
def test_genomic_dataframes_have_expected_columns_and_types(df: pl.DataFrame) -> None:
    expected = ["chromosome", "start", "end", "strand", "gene", "transcript_id"]

    assert df.columns == expected
    assert df.schema == {
        "chromosome": pl.String,
        "start": pl.Int64,
        "end": pl.Int64,
        "strand": pl.String,
        "gene": pl.String,
        "transcript_id": pl.String,
    }
    assert (df["start"] < df["end"]).all()


@given(
    genomic_dataframes(
        min_size=1,
        max_size=10,
        chromosomes=("chr1", "chr2"),
        columns=GenomicColumns(
            chromosome="Chromosome",
            start="Start",
            end="End",
            strand="Strand",
            gene="Gene",
            transcript_id="TranscriptID",
        ),
    )
)
def test_genomic_dataframe_columns_can_match_polaranges_conventions(
    df: pl.DataFrame,
) -> None:
    assert df.columns == ["Chromosome", "Start", "End", "Strand", "Gene", "TranscriptID"]
    assert set(df["Chromosome"]) <= {"chr1", "chr2"}
    assert (df["Start"] < df["End"]).all()
