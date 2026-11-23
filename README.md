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
- Dataset scalability
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

Synthetic 10K was used for preliminary experiments
and is not included in the selected official results.

Large datasets are not stored in this Git repository.

## Experimental progress

As of 2026-11-23, the baseline campaign and the
additional Synthetic 1M Recall-Latency sweeps
are complete.

Completed:
- 12 selected baseline series
- 4 additional Recall-Latency sweep series
- 16 verified official series in total
- 80 verified official runs
- 320 per-run/filter observations
- 64 configuration/filter summaries

All selected official series have five repetitions.

## Official baseline results

Selected baseline results are available under:

reports/baselines/

Files:
- manifest.csv: provenance of 12 baseline sessions
- per_run.csv: 240 per-run/filter observations
- summary.csv: 48 aggregated results
- README.md: methodology and limitations

Baseline conditions:
- Three official datasets
- Two database systems
- 1 and 8 concurrent closed-loop clients
- Four metadata-filter selectivities
- Primary distance metric: L2
- Recall@10 against exact ground truth

Baseline search settings:
- PostgreSQL/pgvector: HNSW ef_search=200
- Chroma: HNSW ef_search=100, n_results=10

Regenerate the baseline export from raw sessions:

    python scripts/export_official_baselines.py

## Official Recall-Latency sweeps

Additional results are available under:

reports/sweeps/

Files:
- manifest.csv: provenance of four sweep sessions
- per_run.csv: 80 per-run/filter observations
- summary.csv: 16 aggregated results
- README.md: configurations and limitations

All four new series use:
- Synthetic 1M, 128 dimensions
- One closed-loop client
- Five repetitions
- 200 measured queries per filter per run
- 20 warm-up queries per client and filter
- Filter selectivities: 100%, 50%, 10%, 1%
- L2 distance and exact Recall@10

PostgreSQL/pgvector settings:
- Baseline ef_search=200
- Sweep ef_search=400
- Sweep ef_search=1000

Chroma settings:
- Baseline HNSW ef_search=100, n_results=10
- Sweep HNSW ef_search=100, n_results=200
- Sweep HNSW ef_search=100, n_results=400

The Chroma sweeps request additional results and
retain the first ten returned identifiers.

Chroma n_results and pgvector ef_search are
different parameters and must not be treated
as numerically equivalent.

Changing the configured Chroma ef_search during
diagnostic experiments did not change the returned
top-ten identifiers in the queries examined.
This observation does not establish the cause.

Regenerate the sweep export from raw sessions:

    python scripts/export_official_sweeps.py

## Methodology

See docs/EXPERIMENT_PROTOCOL.md for the experimental
protocol and limitations.

Important considerations:
- The host was not completely isolated.
- Results are configuration-specific.
- CPU and RAM measurements are approximations.
- Short monitoring windows require caution.
- P99 estimates are exploratory.
- Similar Recall@10 levels are needed for meaningful
  recall-matched performance comparisons.
- Recall-matched settings may not be available for
  every dataset and filter condition.

Raw results and logs remain local and are excluded
from Git. They require separate backups.

## Remaining work

The baseline and Synthetic 1M Recall-Latency campaigns
are complete, but the overall study is not finished.

Remaining areas include:
- Additional distance metrics
- Ingestion and index-construction measurements
- Disk-footprint measurements
- Cold versus warm behavior
- Constrained-memory experiments
- Further resource analysis
- Statistical analysis and visualizations
- Final report and conclusions

## Reproducibility

The repository contains benchmark code, dataset
generation scripts, adapters, experiment configuration,
and selected exported results.

Reproducing the full experiments also requires
preparing the datasets, loading the databases,
constructing the indexes, and running the documented
workloads under comparable conditions.
