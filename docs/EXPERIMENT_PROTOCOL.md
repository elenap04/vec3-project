# Vec3 — Experimental Protocol

Updated: 2026-11-22

## 1. Research objective

Compare Chroma and PostgreSQL/pgvector as complete vector
database systems using common datasets and documented
experimental conditions.

Evaluation dimensions:
- Recall@10 against exact ground truth
- Latency: p50, p95 and exploratory p99
- Query throughput (QPS)
- Metadata filtering
- Concurrent clients
- CPU and RAM consumption
- Dataset scalability
- Ingestion and index construction
- Storage footprint
- Recall-Latency trade-offs
- Cold/warm and constrained-memory behavior

Completed and pending experiments are distinguished below.

## 2. Environment

Baseline experiments were performed on one Debian laptop.

Recorded environment:
- Debian 13
- Intel Core i7-8550U
- 8 logical CPUs
- Approximately 15 GiB RAM
- Local SSD
- Docker
- Chroma 1.5.9
- PostgreSQL 17
- pgvector 0.8.7

Both databases ran in Docker containers on the same host.

No explicit Docker CPU or RAM limits were configured.

Benchmark workloads were executed sequentially.
Unrelated demanding applications were avoided during
the selected official benchmark sessions.

The shared host was not completely isolated.

## 3. Datasets

| Dataset | Vectors | Dimensions |
|---|---:|---:|
| Synthetic 10K | 10,000 | 128 |
| Synthetic 100K | 100,000 | 128 |
| Synthetic 1M | 1,000,000 | 128 |
| AG News 100K | 100,000 | 384 |

Synthetic datasets use reproducible generation settings.

AG News source: fancyzhx/ag_news
Embedding model: BAAI/bge-small-en-v1.5

Synthetic 10K was used for preliminary experiments
and is not included in the selected official baselines.

Official baseline datasets:
- Synthetic 100K
- Synthetic 1M
- AG News 100K

## 4. Correctness and filters

Primary baseline distance metric: L2.

Each official dataset has 200 query vectors and exact
ground truth computed independently of both databases.

Recall@10 compares returned identifiers with the exact
top-10 neighbors under each filter condition.

Filter selectivities:
- 100%: unfiltered
- 50%
- 10%
- 1%

Filtering uses bucket metadata.

For AG News, filtering is not based on news categories.

Filtered ground truth is computed separately for each
eligible subset.

## 5. HNSW configuration

Common construction parameters:
- M / max_neighbors: 16
- ef_construction: 100
- Distance metric: L2

PostgreSQL/pgvector baseline:
- hnsw.ef_search = 200

Chroma baseline:
- HNSW ef_search = 100
- n_results = 10
- No additional candidate oversampling

The generic ef_search value printed by the common
runner is not applied to Chroma.

Chroma search configuration belongs to the collection.

These configurations do not necessarily have equal
Recall@10 and must not be presented as recall-matched.

## 6. Official baseline workloads

Completed configurations:
- 3 official datasets
- 2 database systems
- 1 and 8 concurrent clients
- 5 repetitions per configuration

Totals:
- 12 selected official series
- 60 validated official runs
- 240 per-run/filter observations

Each run used:
- 200 measured queries per filter
- 4 filter selectivities
- 20 warm-up queries per client and filter
- k = 10
- Closed-loop client model

With 8 clients, measured queries were distributed
across clients.

Warm-up queries were excluded from measured latency
and throughput.

## 7. Repetitions and statistics

Each selected configuration has five completed
and validated repetitions.

Recorded statistics include:
- Recall@10
- QPS
- Latency p50, p95 and p99
- Mean and median
- Standard deviation
- Estimated 95% confidence intervals
- Container CPU and RAM measurements

Confidence intervals describe observed variability
across five repeated runs.

Since each filter has only 200 measured queries per run,
p99 estimates are exploratory.

Development pilots are excluded from official results.

## 8. Resource monitoring

Resource monitoring uses Linux cgroups v2 statistics.

CPU percentages are expressed in logical-core
equivalents. Approximately 100% corresponds to one
fully utilized logical CPU.

Memory is an approximation of container working-set
usage, not memory uniquely attributable to one query
or dataset.

Resource measurements are collected per filter.

The selected Synthetic 100K PostgreSQL 1-client series
used a sampling interval of 100 ms.

Other selected baseline series used 50 ms.

Short measurement windows may contain relatively few
samples. A resource status of OK confirms the minimum
validation requirements, not perfect measurement
precision.

## 9. Official session selection

The final baseline export includes only 12 explicitly
selected and verified benchmark sessions.

Required checks:
- Correct database and dataset
- Correct client count
- Five completed runs
- Four filters per run
- Completed query runner logs
- Five valid resource runs per filter
- Correct source session identification

Superseded, interrupted and nonstandard executions
are excluded from the selected baseline results.

They are preserved locally rather than deleted.

Selected results and provenance:
- reports/baselines/manifest.csv
- reports/baselines/per_run.csv
- reports/baselines/summary.csv
- reports/baselines/README.md

Export script:
- scripts/export_official_baselines.py

Raw result JSON files and logs must be preserved
and backed up separately.

## 10. Additional experiments — not yet completed

The following remain part of the study:

1. Recall-Latency and ANN parameter sweeps.
   Test meaningful search settings for both systems
   and identify comparable Recall@10 levels where
   these are achievable.

2. Additional distance metrics.
   Evaluate cosine and inner product with compatible
   indexes and suitable ground truth.

3. Ingestion and indexing.
   Measure ingestion, index construction, readiness,
   storage footprint and resource consumption with
   clearly documented timing boundaries.

4. Cold/warm experiments.
   Define reproducible cache-state conditions.

5. Constrained-memory experiments.
   Apply documented, comparable memory limits and
   record performance changes or failures.

6. Additional scalability and resource analysis.
   Account for short measurement windows and
   shared-host limitations.

7. Statistical analysis, visualizations and reporting.

Completion of the 12 baseline series does not mean
that these additional experiments are complete.

## 11. Reporting principles

Final comparisons must:
- Preserve the actual database configurations.
- Show Recall@10 alongside latency and throughput.
- Distinguish measured results from hypotheses.
- Avoid unsupported causal explanations.
- Report variability and measurement limitations.
- Identify unattainable recall targets honestly.
- Separate pilots from official results.
- Preserve source session identifiers.
- Keep complete raw records for reproducibility.
