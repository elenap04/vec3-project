# Vec3 — Chroma vs PostgreSQL/pgvector

**Performance Evaluation of Vector Database Systems**

A comparative benchmarking project for the Information
Management Systems course, academic year 2026–2027.

## Research goal

Evaluate Chroma and PostgreSQL/pgvector using common
datasets and documented experimental conditions.

The study examines:
- Search accuracy (Recall@10)
- Query latency and throughput
- Metadata filtering and concurrency
- CPU and RAM consumption
- Ingestion and index construction
- Storage footprint
- ANN search parameters and Recall–Latency trade-offs

## Technologies

- Python
- Docker / Docker Compose
- Chroma 1.5.9
- PostgreSQL 17 with pgvector 0.8.7
- HNSW approximate nearest-neighbor indexing

## Datasets

| Dataset | Vectors | Dimensions |
|---|---:|---:|
| Synthetic 10K | 10,000 | 128 |
| Synthetic 100K | 100,000 | 128 |
| Synthetic 1M | 1,000,000 | 128 |
| AG News embeddings | 100,000 | 384 |

AG News embeddings use BAAI/bge-small-en-v1.5.

Synthetic 10K was used for preliminary experiments.
It is not included in the 12 selected official series.

Large datasets are not stored in this Git repository.

## Experimental progress

As of 2026-11-22, the initial official baseline
benchmark campaign is complete.

Completed:
- 3 official datasets
- 2 vector database systems
- 1 and 8 concurrent closed-loop clients
- 4 metadata-filter selectivities
- 5 repetitions per configuration
- 12 verified official series
- 60 verified runs
- 240 per-run/filter observations

The primary baseline distance metric is L2.

Baseline query configurations:
- PostgreSQL/pgvector: HNSW ef_search=200
- Chroma: collection HNSW ef_search=100, n_results=10

These configurations were not selected to provide
equal Recall@10.

## Official baseline results

Selected results are available under:

reports/baselines/

Files:
- manifest.csv: provenance of 12 selected sessions
- per_run.csv: 240 per-run/filter observations
- summary.csv: 48 aggregated results
- README.md: result documentation and limitations

The export script is:

scripts/export_official_baselines.py

To regenerate the export from locally preserved
raw sessions, run:

    python scripts/export_official_baselines.py

Raw results, logs, and datasets are retained locally
and excluded from Git. They require separate backups.

## Methodology

See docs/EXPERIMENT_PROTOCOL.md for detailed
experimental methodology, configurations, and
limitations.

Official benchmarks used:
- 200 measured queries per filter and run
- 20 warm-up queries per client and filter
- k=10
- Exact ground truth for Recall@10
- Sequential experiments on the same laptop

Important limitations:
- The shared host was not completely isolated.
- Baseline results are configuration-specific.
- Container memory is a working-set approximation.
- Short CPU/RAM sampling windows require caution.
- p99 latency estimates are exploratory.
- Recall-matched comparisons require parameter sweeps.

## Remaining experiments

The baseline campaign is complete, but the overall
study is not yet finished.

Remaining work:
- Recall–Latency and ANN parameter sweeps
- Additional distance metrics
- Ingestion, indexing, and disk-footprint measurements
- Cold versus warm behavior
- Constrained-memory experiments
- Statistical analysis and visualizations
- Final report and conclusions

## Reproducibility

The repository contains benchmark code, dataset
generation scripts, adapters, and experimental
configuration files.

Reproducing the complete benchmarks also requires
preparing datasets, loading both databases,
constructing indexes, and executing the documented
workloads.
