"""Export the 12 selected official baseline sessions.

Raw benchmark files remain unchanged in results/.
Only the explicitly selected, validated sessions are exported.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("results/official/repetitions")
OUTPUT = Path("reports/baselines")

SESSIONS = [
    ("100k", "pgvector", 1, "20261108T115342094350Z_ddec8543"),
    ("100k", "pgvector", 8, "20261108T120607065023Z_3099b6d5"),
    ("100k", "chroma", 1, "20261108T121018372262Z_b4da0dad"),
    ("100k", "chroma", 8, "20261108T203235885693Z_b87be9ed"),

    ("agnews100k", "pgvector", 1, "20261108T215401485103Z_e9840a3c"),
    ("agnews100k", "pgvector", 8, "20261108T215907773328Z_dcd24858"),
    ("agnews100k", "chroma", 1, "20261108T222215528968Z_14fa0131"),
    ("agnews100k", "chroma", 8, "20261109T123044181747Z_1e347c8b"),

    ("1m", "pgvector", 1, "20261120T103242595365Z_e1351248"),
    ("1m", "pgvector", 8, "20261120T103845201808Z_86ac5bf9"),
    ("1m", "chroma", 1, "20261121T145620563053Z_77e98037"),
    ("1m", "chroma", 8, "20261121T220759963377Z_1cdb8c26"),
]

FILTERS = {100, 50, 10, 1}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    assert rows, f"No rows for {path}"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    manifest = []
    all_per_run = []
    all_summary = []

    print("===== EXPORT SELECTED OFFICIAL BASELINES =====")

    # Validate every source before writing the exported files.
    for dataset, database, clients, session_id in SESSIONS:
        folder = ROOT / session_id
        summary_path = folder / "summary.json"

        raw_summary = summary_path.read_bytes()
        summary = json.loads(raw_summary)

        assert summary["mode"] == "official"
        assert summary["dataset_key"] == dataset
        assert summary["database"] == database
        assert int(summary["clients"]) == clients
        assert int(summary["repetitions"]) == 5

        per_run = read_csv(folder / "per_run.csv")
        summary_rows = read_csv(folder / "summary.csv")

        assert len(per_run) == 20
        assert len(summary_rows) == 4

        assert {
            int(row["filter_percent"]) for row in summary_rows
        } == FILTERS

        assert Counter(
            int(row["filter_percent"]) for row in per_run
        ) == {value: 5 for value in FILTERS}

        for row in per_run:
            assert row["database"] == database
            assert row["dataset"] == dataset
            assert int(row["clients"]) == clients
            assert row["mode"] == "official"
            assert row["resource_status"] == "ok"
            assert int(row["resource_samples"]) >= 3

        for row in summary_rows:
            assert int(row["valid_resource_runs"]) == 5

        for run_number in range(1, 6):
            log_path = folder / f"run_{run_number:02d}.log"
            log_text = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            assert "QUERY RUNNER COMPLETED" in log_text
            assert f"Clients: {clients}" in log_text

        for value in FILTERS:
            assert (
                summary["filters"][str(value)]["resources"]["valid_runs"]
                == 5
            )

        sampling_ms = (
            100
            if session_id == "20261108T115342094350Z_ddec8543"
            else 50
        )

        search_configuration = (
            "pgvector hnsw.ef_search=200"
            if database == "pgvector"
            else "Chroma HNSW ef_search=100; n_results=10"
        )

        manifest.append({
            "dataset": dataset,
            "database": database,
            "clients": clients,
            "session_id": session_id,
            "runs": 5,
            "filters": "100,50,10,1",
            "queries_per_filter_per_run": 200,
            "warmup_per_client_per_filter": 20,
            "search_configuration": search_configuration,
            "resource_sampling_ms": sampling_ms,
            "source_summary_sha256": hashlib.sha256(
                raw_summary
            ).hexdigest(),
        })

        for row in per_run:
            all_per_run.append({
                "session_id": session_id,
                **row,
            })

        for row in summary_rows:
            all_summary.append({
                "session_id": session_id,
                **row,
            })

        print(
            f"PASS | {dataset:10} | {database:8} | "
            f"clients={clients} | {session_id}"
        )

    assert len(manifest) == 12
    assert len(all_per_run) == 240
    assert len(all_summary) == 48

    OUTPUT.mkdir(parents=True, exist_ok=True)

    write_csv(OUTPUT / "manifest.csv", manifest)
    write_csv(OUTPUT / "per_run.csv", all_per_run)
    write_csv(OUTPUT / "summary.csv", all_summary)

    documentation = """# Selected official baseline results

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
"""

    (OUTPUT / "README.md").write_text(
        documentation,
        encoding="utf-8",
    )

    print()
    print("===== EXPORT COMPLETED =====")
    print("Selected sessions:", len(manifest))
    print("Per-run rows:", len(all_per_run))
    print("Summary rows:", len(all_summary))

    for path in sorted(OUTPUT.iterdir()):
        print(f"{path}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
