"""Export four selected, verified Synthetic 1M recall-latency sweeps."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("results/official/repetitions")
OUTPUT = Path("reports/sweeps")

# Dataset, database, clients, parameter, value, session ID.
SESSIONS = [
    (
        "1m", "pgvector", 1, "ef_search", 400,
        "20261122T013949894595Z_d4ebab5d",
    ),
    (
        "1m", "pgvector", 1, "ef_search", 1000,
        "20261122T013021548193Z_a9879a49",
    ),
    (
        "1m", "chroma", 1, "n_results", 200,
        "20261123T015414794871Z_d2604575",
    ),
    (
        "1m", "chroma", 1, "n_results", 400,
        "20261122T233433032043Z_4d68844c",
    ),
]

FILTERS = {100, 50, 10, 1}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    assert rows, f"No records for {path}"

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = []
    all_runs = []
    all_summaries = []

    print("===== VERIFY OFFICIAL RECALL-LATENCY SWEEPS =====")

    for dataset, database, clients, parameter, value, session_id in SESSIONS:
        folder = ROOT / session_id

        summary_path = folder / "summary.json"
        raw = summary_path.read_bytes()
        summary = json.loads(raw)

        assert summary["mode"] == "official"
        assert summary["dataset_key"] == dataset
        assert summary["database"] == database
        assert int(summary["clients"]) == clients
        assert int(summary["repetitions"]) == 5
        assert int(summary["queries_per_run"]) == 200
        assert int(summary["warmup_queries_per_client_per_filter"]) == 20
        assert summary["metric"] == "L2"
        assert int(summary["k"]) == 10

        if database == "pgvector":
            assert int(summary["ef_search"]) == value
            search_configuration = f"pgvector HNSW ef_search={value}"
            actual_hnsw_ef = value
            candidates = None
        else:
            assert int(summary["chroma_candidates"]) == value
            assert summary["search_strategy"] == "chroma_oversampling_top10"
            search_configuration = (
                f"Chroma HNSW ef_search=100; n_results={value}; "
                "top 10 retained"
            )
            actual_hnsw_ef = 100
            candidates = value

        per_run = read_csv(folder / "per_run.csv")
        summaries = read_csv(folder / "summary.csv")

        assert len(per_run) == 20
        assert len(summaries) == 4

        assert {
            int(row["filter_percent"]) for row in summaries
        } == FILTERS

        assert Counter(
            int(row["filter_percent"]) for row in per_run
        ) == {f: 5 for f in FILTERS}

        assert {
            int(row["run"]) for row in per_run
        } == {1, 2, 3, 4, 5}

        assert len({
            (int(row["run"]), int(row["filter_percent"]))
            for row in per_run
        }) == 20

        for row in per_run:
            assert row["mode"] == "official"
            assert row["database"] == database
            assert row["dataset"] == dataset
            assert int(row["clients"]) == clients
            assert row["resource_status"] == "ok"
            assert int(row["resource_samples"]) >= 3

            if database == "pgvector":
                assert int(row["ef_search"]) == value
            else:
                assert int(row["chroma_candidates"]) == value

        for row in summaries:
            assert row["mode"] == "official"
            assert row["database"] == database
            assert row["dataset"] == dataset
            assert int(row["clients"]) == clients
            assert int(row["runs"]) == 5
            assert int(row["valid_resource_runs"]) == 5

            if database == "pgvector":
                assert int(row["ef_search"]) == value
            else:
                assert int(row["chroma_candidates"]) == value

        for f in FILTERS:
            assert summary["filters"][str(f)]["resources"]["valid_runs"] == 5

        for number in range(1, 6):
            run_path = folder / f"run_{number:02d}.json"
            log_path = folder / f"run_{number:02d}.log"

            run = json.loads(run_path.read_text(encoding="utf-8"))
            log = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            assert "QUERY RUNNER COMPLETED" in log
            assert run["mode"] == "official"
            assert run["database"] == database
            assert run["dataset_key"] == dataset
            assert int(run["clients"]) == clients

            if database == "pgvector":
                assert int(run["ef_search"]) == value
            else:
                assert int(run["chroma_candidates"]) == value
                assert (
                    run["chroma_hnsw_configuration"]["ef_search"]
                    == 100
                )

        manifest.append({
            "dataset": dataset,
            "database": database,
            "clients": clients,
            "parameter": parameter,
            "value": value,
            "actual_hnsw_ef_search": actual_hnsw_ef,
            "chroma_n_results": candidates,
            "search_configuration": search_configuration,
            "session_id": session_id,
            "runs": 5,
            "filters": "100,50,10,1",
            "queries_per_filter_per_run": 200,
            "warmup_per_client_per_filter": 20,
            "source_summary_sha256": hashlib.sha256(raw).hexdigest(),
        })

        all_runs.extend(
            {"session_id": session_id, **row}
            for row in per_run
        )

        all_summaries.extend(
            {"session_id": session_id, **row}
            for row in summaries
        )

        print(
            f"PASS | {database:8} | "
            f"{parameter}={value:<4} | "
            f"5 runs | 4 filters"
        )

    assert len(manifest) == 4
    assert len(all_runs) == 80
    assert len(all_summaries) == 16

    OUTPUT.mkdir(parents=True, exist_ok=True)

    write_csv(OUTPUT / "manifest.csv", manifest)
    write_csv(OUTPUT / "per_run.csv", all_runs)
    write_csv(OUTPUT / "summary.csv", all_summaries)

    documentation = """# Official Synthetic 1M Recall-Latency sweeps

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
"""

    (OUTPUT / "README.md").write_text(
        documentation,
        encoding="utf-8",
    )

    print("\n===== EXPORT SUMMARY =====")
    print("Selected new series:", len(manifest))
    print("Official runs:", len(manifest) * 5)
    print("Per-run/filter rows:", len(all_runs))
    print("Summary rows:", len(all_summaries))

    for path in sorted(OUTPUT.iterdir()):
        print(f"{path}: {path.stat().st_size} bytes")

    print("\nOFFICIAL SWEEP EXPORT — PASS")


if __name__ == "__main__":
    main()
