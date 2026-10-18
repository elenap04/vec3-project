
import argparse
import json
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path
from statistics import median, stdev


FILTERS = ("100", "50", "10", "1")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--db",
        required=True,
        choices=["chroma", "pgvector"],
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--ef-search",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--warmup-queries",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if args.runs < 2:
        parser.error("--runs must be at least 2")

    if args.ef_search < 1:
        parser.error("--ef-search must be positive")

    if args.warmup_queries < 0:
        parser.error("--warmup-queries cannot be negative")

    run_files = []
    run_results = []

    for number in range(1, args.runs + 1):
        print(
            f"\n===== RUN {number}/{args.runs} =====",
            flush=True,
        )

        command = [
            sys.executable,
            "-m",
            "src.run_queries",
            "--db",
            args.db,
            "--warmup-queries",
            str(args.warmup_queries),
        ]

        if args.db == "pgvector":
            command += [
                "--ef-search",
                str(args.ef_search),
            ]

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )

        print(completed.stdout, end="", flush=True)

        if completed.returncode != 0:
            print(completed.stderr, file=sys.stderr)
            raise SystemExit(
                f"Run {number} failed"
            )

        prefix = "Results saved to: "

        paths = [
            line[len(prefix):].strip()
            for line in completed.stdout.splitlines()
            if line.startswith(prefix)
        ]

        if len(paths) != 1:
            raise RuntimeError(
                "Could not identify run result file"
            )

        path = Path(paths[0])

        if not path.is_file():
            raise FileNotFoundError(path)

        result = json.loads(
            path.read_text(encoding="utf-8")
        )

        if result["database"] != args.db:
            raise ValueError("Unexpected database")

        run_files.append(str(path))
        run_results.append(result)

    summary = {
        "mode": "development_pilot",
        "database": args.db,
        "repetitions": args.runs,
        "ef_search": (
            args.ef_search
            if args.db == "pgvector"
            else None
        ),
        "warmup_queries": args.warmup_queries,
        "run_files": run_files,
        "filters": {},
    }

    print("\n===== REPETITION SUMMARY =====")

    for selectivity in FILTERS:
        metrics = [
            result["filters"][selectivity]
            for result in run_results
        ]

        qps = [item["qps"] for item in metrics]
        p95 = [
            item["latency_p95_ms"]
            for item in metrics
        ]
        recall = [
            item["mean_recall_at_10"]
            for item in metrics
        ]

        values = {
            "median_qps": median(qps),
            "std_qps": stdev(qps),
            "median_p95_ms": median(p95),
            "mean_recall_at_10": sum(recall) / len(recall),
        }

        summary["filters"][selectivity] = values

        print(
            f"Filter {selectivity:>3}% | "
            f"median QPS={values['median_qps']:.2f} | "
            f"std={values['std_qps']:.2f} | "
            f"median p95={values['median_p95_ms']:.2f} ms"
        )

    output_dir = Path("results/pilot/repetitions")
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    setting = (
        f"ef{args.ef_search}"
        if args.db == "pgvector"
        else "default"
    )

    output_path = output_dir / (
        f"summary_{args.db}_{setting}_{stamp}.json"
    )

    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\nSummary saved to: {output_path}")
    print("REPETITIONS COMPLETED")


if __name__ == "__main__":
    main()

