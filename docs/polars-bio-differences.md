# polars-bio differences

This file records known behavioral differences and oracle caveats that matter
for property tests comparing `polaranges_py` with `polars-bio`.

## `nearest` `k` semantics

Both libraries expose a `k` argument for nearest-neighbor queries, but the
semantics differ.

`polars-bio` treats `k` as a literal row limit per query interval. For example,
`k=1` returns at most one nearest right interval for each left interval, even
when several right intervals tie at the same distance.

`polaranges_py` forwards `k` to `polaranges`/`ruranges-core`, where `k` behaves
as the number of nearest distance levels. All right intervals whose distances
fall within those `k` nearest distance levels are returned. This means `k=1`
can return more than one row for a left interval when there are ties.

Property tests should account for this before comparing exact nearest outputs.
One option is to ask `polars-bio` for enough rows to expose the tied candidates
and then compare after filtering to the same distinct distance levels.

## `nearest` nulls zero-based single-base intervals

Observed with `polars-bio` 0.29.0.

When `datafusion.bio.coordinate_system_zero_based` is `true`, `pb.nearest` can
return a row with null right-side fields for valid single-base intervals, even
when the candidate right frame is non-empty and contains an exact match.

Minimal example:

```text
left:  chrom=chr1, Start=0, End=1
right: chrom=chr1, Start=0, End=1
```

Observed result:

```text
right_id_2 = null
distance = null
```

The nearest property tests patch only this null-row oracle artifact after the
`polars-bio` call, so other `polars-bio` mismatches can still surface.
