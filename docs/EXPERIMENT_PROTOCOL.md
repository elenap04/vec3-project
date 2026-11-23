# Vec3 — Experimental Protocol

Updated: 2026-11-23

## 1. Research objective

Compare Chroma and PostgreSQL/pgvector as vector
database systems under documented experimental
conditions.

The evaluation covers:
- Recall@10 against exact ground truth
- Query latency and throughput
- Metadata-filter selectivity
- Concurrent clients
- CPU and RAM consumption
- Dataset scalability
- ANN search-parameter trade-offs
- Ingestion and index construction
- Storage footprint
- Cold/warm execution
- Constrained-memory behavior

Completed experiments and remaining work are
distinguished throughout this document.

## 2. Experimental environment

The selected benchmarks ran on a single laptop.

Environment:
- Debian 13
- Intel Core i7-8550U
- Four physical cores
- Eight logical CPUs
- Approximately 15 GiB RAM
- Local SSD
- Docker and Docker Compose
- Chroma 1.5.9
- PostgreSQL 17
- pgvector 0.8.7

Both systems ran in containers on the same host.

No explicit Docker CPU or memory limits were applied
during the baseline or Recall-Latency experiments.

Experiments were executed sequentially.

The host was not completely isolated, so operating
system activity and other environmental effects
remain possible sources of measurement variability.

## 3. Datasets

| Dataset | Vectors | Dimensions |
|---|---:|---:|
| Synthetic 10K | 10,000 | 128 |
| Synthetic 100K | 100,000 | 128 |
| Synthetic 1M | 1,000,000 | 128 |
| AG News 100K | 100,000 | 384 |

AG News embeddings were produced with
BAAI/bge-small-en-v1.5.

Synthetic 10K was used for preliminary development
and correctness experiments.

The three official baseline datasets were:
- Synthetic 100K
- Synthetic 1M
- AG News 100K

The additional Recall-Latency sweeps were performed
on Synthetic 1M only.

## 4. Exact ground truth and filtering

The primary evaluation distance metric is L2.

Recall@10 measures the overlap between the returned
top-ten identifiers and exact top-ten nearest neighbors.

Official experiments use 200 measured queries per
filter and run.

Filter selectivities:
- 100%: no metadata filter
- 50%
- 10%
- 1%

Filtered queries select vectors using bucket metadata.

For AG News, bucket filtering is not equivalent to
filtering by news category.

Exact ground truth is computed separately for each
eligible filtered subset.

## 5. HNSW index construction

Common index construction settings:
- Maximum neighbors / M: 16
- ef_construction: 100
- Distance metric: L2

These settings were held fixed in the selected
baseline and Synthetic 1M Recall-Latency experiments.

The search-time configurations differ by database
and by experimental phase.

## 6. Initial official baseline campaign

The selected baseline campaign contains:
- Three datasets
- Two database systems
- One and eight concurrent closed-loop clients
- Four metadata-filter selectivities
- Five repetitions per configuration

Baseline totals:
- 12 selected official series
- 60 verified official runs
- 240 per-run/filter observations
- 48 configuration/filter summary rows

For each run:
- k = 10
- 200 measured queries per filter
- 20 warm-up queries per client and filter
- Four filter selectivities

Baseline PostgreSQL setting:
- HNSW ef_search = 200

Baseline Chroma setting:
- Collection HNSW ef_search = 100
- n_results = 10
- No additional candidate oversampling

The configurations were not selected to have
equal Recall@10.

Selected baseline exports:
- reports/baselines/manifest.csv
- reports/baselines/per_run.csv
- reports/baselines/summary.csv
- reports/baselines/README.md

Export script:
- scripts/export_official_baselines.py

## 7. Additional official Recall-Latency sweeps

Four additional official series were completed on
Synthetic 1M with one closed-loop client.

Each series used:
- Five repetitions
- Four filter selectivities
- 200 measured queries per filter and run
- 20 warm-up queries per client and filter
- L2 distance
- Exact Recall@10
- k = 10

PostgreSQL/pgvector:
- Baseline: HNSW ef_search = 200
- Additional series: ef_search = 400
- Additional series: ef_search = 1000

Chroma:
- Baseline: HNSW ef_search = 100, n_results = 10
- Additional series: HNSW ef_search = 100,
  n_results = 200
- Additional series: HNSW ef_search = 100,
  n_results = 400

For Chroma oversampling, the client requests
n_results candidates and retains only the first
ten identifiers for Recall@10 evaluation.

The entire Chroma request is included in timing.

The Chroma n_results parameter is not equivalent
to PostgreSQL's HNSW ef_search parameter.

These experiments compare the Recall-Latency
trade-offs of two different search strategies.

The four new series contain:
- 20 verified official runs
- 80 per-run/filter observations
- 16 configuration/filter summary rows

Selected sweep exports:
- reports/sweeps/manifest.csv
- reports/sweeps/per_run.csv
- reports/sweeps/summary.csv
- reports/sweeps/README.md

Export script:
- scripts/export_official_sweeps.py

## 8. Overall completed official experiments

Combining baselines and additional sweeps:

- 16 selected official series
- 80 verified official runs
- 320 per-run/filter observations
- 64 configuration/filter summary rows

These counts describe the selected experiments,
not all historical development pilots.

Raw experimental sessions remain local and are
excluded from the Git repository.

## 9. Chroma configuration diagnostics

The actual HNSW configuration was retrieved from
the Chroma collection.

For Synthetic 1M, the original collection used:
- HNSW ef_search = 100
- ef_construction = 100
- max_neighbors = 16
- L2 distance

A diagnostic confirmed that Chroma 1.5.9 accepted
a temporary change of ef_search from 100 to 200.

The configuration was restored to 100 and the
collection still contained 1,000,000 vectors.

Additional diagnostics tested configured ef_search
values of 100, 400 and 1000.

For the small set of queries examined, the returned
top-ten identifiers did not change across these
configured values.

This observation does not establish whether the
search algorithm used the updated setting internally.

Consequently, the selected official Chroma sweeps
used n_results oversampling while keeping the
collection HNSW ef_search fixed at 100.

Diagnostic queries showed that increasing n_results
to 400 could improve Recall@10 at the cost of
additional latency.

Diagnostic timings are not mixed with official
benchmark results.

The common runner's generic ef_search CLI default
does not configure Chroma's collection.

For Chroma, the actual HNSW collection configuration
was verified in the underlying run JSON files.

## 10. Experimental interpretation

The additional PostgreSQL series showed higher
Recall@10 with increased ef_search.

The additional Chroma oversampling series showed
higher Recall@10 with increased n_results.

The cost in latency depends on the filter condition.

In particular, Chroma already exhibited high
filtered-query latency in its original baseline.

The cause of that behavior has not been established.

Cross-system performance should be compared at
similar observed Recall@10 levels where possible.

Numerically matching ef_search and n_results
would not constitute a fair parameter match.

Comparisons should report accuracy, latency and
throughput together, including variability.

The official summaries retain per-setting and
per-filter measurements for this purpose.

## 11. Repetitions and statistics

Every selected official series contains five runs.

Reported statistics include:
- Mean and median
- Standard deviation
- Estimated 95% confidence intervals
- Recall@10
- QPS
- Latency p50, p95 and p99

Intervals summarize variability across five runs.

The fixed query workload can produce identical
Recall@10 values across repeated runs.

Because only 200 measured queries are executed per
filter and run, p99 estimates are exploratory.

Development pilots are excluded from official
aggregate results.

## 12. CPU and RAM monitoring

Resource measurements use Linux cgroups v2.

CPU percentages use logical-core equivalents.
Approximately 100% represents one fully utilized
logical CPU.

Memory is an approximation of container working-set
usage, not memory uniquely attributable to a query.

The selected Synthetic 100K PostgreSQL one-client
baseline used a resource sampling interval of 100 ms.

The other selected baseline series used 50 ms.

The additional selected sweeps used the same
monitoring framework.

Short query windows can produce relatively few
resource samples.

A resource status of OK indicates that the
configured validation criteria were met.
It does not guarantee perfect measurement precision.

## 13. Validation and provenance

The selected experiments were checked for:
- Correct database and dataset
- Correct client count
- Correct search configuration
- Five completed repetitions
- Four filter summaries
- Twenty per-run/filter rows per series
- Valid resource status for every run and filter
- Completed query-runner logs

The sweep exporter additionally verifies the
underlying run configuration records.

Export manifests retain source session identifiers
and SHA-256 hashes of source summary JSON files.

Superseded, interrupted and development runs are
not included in the selected official exports.

Raw results and logs should be backed up separately.

## 14. Remaining experiments

The baseline and Synthetic 1M Recall-Latency
campaigns are complete.

The overall project is not yet finished.

Remaining experimental and reporting tasks:
- Additional distance metrics
- Ingestion and index-construction timing
- Disk-footprint measurement
- Cold versus warm behavior
- Constrained-memory experiments
- Additional resource analysis
- Statistical analysis and visualization
- Final written report and conclusions

These tasks must not be represented as completed
until their evidence has been collected and checked.

## 15. Reporting principles

Final reporting should:
- Preserve actual configurations and provenance.
- Distinguish official data from pilot observations.
- Report Recall@10 together with latency and QPS.
- Discuss measurement variability.
- Avoid unsupported causal explanations.
- Explain the different search parameters.
- Identify unavailable recall-matched comparisons.
- Explain the limitations of the shared host.
- Preserve raw records for reproducibility.
