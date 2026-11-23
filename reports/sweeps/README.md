# Official Synthetic 1M Recall-Latency sweeps

This directory contains four additional official
benchmark series, separate from the 12 baseline series.

## Contents

- manifest.csv: four selected sessions and provenance
- per_run.csv: 80 run/filter observations
- summary.csv: 16 configuration/filter summaries

Export with:

    python scripts/export_official_sweeps.py

## Experimental conditions

- Dataset: Synthetic 1M, 128 dimensions
- Distance metric: L2
- One closed-loop client
- Five repetitions per setting
- 200 measured queries per filter per run
- 20 warm-up queries per client per filter
- Filters: 100%, 50%, 10%, 1%
- Recall@10 against exact filtered ground truth

## Search configurations

PostgreSQL/pgvector:
- Baseline: HNSW ef_search=200
- Additional sweeps: ef_search=400 and 1000

Chroma:
- Baseline: HNSW ef_search=100, n_results=10
- Additional sweeps: HNSW ef_search=100,
  n_results=200 and 400, retaining the first 10 IDs

Chroma n_results is not equivalent to pgvector ef_search.

The generic CLI ef_search value appearing in Chroma
run CSV files does not configure the Chroma collection.
The actual Chroma HNSW configuration was verified
in the underlying run JSON files.

## Interpretation

These results measure the Recall-Latency trade-off
for the selected search strategies. Settings should
be compared at similar measured Recall@10 values
where possible, not by matching their numeric values.

Chroma's filtered-query latencies were already high
in the original baseline. The cause is not established.

Every official setting has five repetitions.
P99 is exploratory because each filter has only
200 measured queries per run.

CPU/RAM values are approximate container measurements.
The benchmark host was not completely isolated.

Raw sessions and detailed logs are retained locally
and require a separate backup.
