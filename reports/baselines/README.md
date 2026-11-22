# Selected official baseline results

These files export the 12 selected official benchmark series
for Chroma and PostgreSQL/pgvector.

## Contents

- `manifest.csv`: provenance for the 12 selected sessions
- `per_run.csv`: 240 per-filter observations from 60 runs
- `summary.csv`: 48 per-configuration/filter summaries

The exporter is `scripts/export_official_baselines.py`.

## Experimental scope

Datasets:
- Synthetic 100K, 128 dimensions
- AG News 100K, 384 dimensions
- Synthetic 1M, 128 dimensions

Each database was evaluated with 1 and 8 closed-loop clients.
There were five repetitions per configuration, 200 measured
queries per filter and run, and 20 warm-up queries per client
and filter.

The filters are 100%, 50%, 10%, and 1% selectivity.
The primary distance metric is L2 and recall uses exact
ground truth with k=10.

Baseline search configurations:
- pgvector: HNSW `ef_search=200`
- Chroma: HNSW `ef_search=100`, requested candidates=10

The `ef_search` field in Chroma runner CSV rows is a generic
CLI default and does not set the Chroma HNSW parameter.

## Provenance and exclusions

Only the session IDs in `manifest.csv` are selected.
The raw source records remain locally under
`results/official/repetitions/`.

Earlier replaced runs, interrupted runs and development
pilots are not included. They are not deleted.

The SHA-256 value in the manifest identifies the source
summary JSON used for each export.

## Measurement limitations

The experiments used a shared laptop, sequential benchmark
execution, and no explicit Docker CPU/RAM limits. This does
not guarantee complete isolation from background activity.

Container CPU percentages represent logical-core equivalents.
Memory is a container working-set approximation, not memory
attributable exclusively to the active dataset.

Resource monitoring sampled at 50 ms except the selected
Synthetic 100K pgvector 1-client series, which used 100 ms.
Short measurement windows can make resource statistics less
precise, even when their validation status is OK.

Each filter has 200 measured queries per run, so p99 is
exploratory. Confidence intervals and summaries describe the
five repetitions, not independent random dataset samples.

The configurations were not tuned to equal Recall@10.
Parameter sweeps and recall-matched comparisons are separate
experiments and are not represented in this baseline export.

The export is not a replacement for full raw logs and query
JSON files, which should be retained and backed up separately.
