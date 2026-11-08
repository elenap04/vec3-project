"""Repeat query experiments with resumable execution and audited outputs.

The default remains development_pilot. The official mode is opt-in and
requires matching support in src.run_queries (added separately).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from uuid import uuid4

from src.run_queries import DATASETS


FILTERS = ("100", "50", "10", "1")
DATASET_KEYS = tuple(sorted(DATASETS))
MIN_RESOURCE_SAMPLES = 3
METRIC_FIELDS = {
    "qps": "qps",
    "p50_ms": "latency_p50_ms",
    "p95_ms": "latency_p95_ms",
    "p99_ms": "latency_p99_ms",
    "recall_at_10": "mean_recall_at_10",
}
# Student-t two-sided 95% quantiles, degrees of freedom 1 through 30.
T95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306,
    2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
    2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
    2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def numeric_summary(values):
    values = [float(value) for value in values]
    if not values:
        return {"median": None, "mean": None, "std": None,
                "ci95_low": None, "ci95_high": None}
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite statistic")
    average = float(mean(values))
    deviation = float(stdev(values)) if len(values) >= 2 else None
    if deviation is not None:
        df = len(values) - 1
        critical = T95[df - 1] if df <= 30 else 1.96
        half_width = critical * deviation / math.sqrt(len(values))
        ci_low, ci_high = average - half_width, average + half_width
    else:
        ci_low, ci_high = None, None
    return {
        "median": float(median(values)), "mean": average,
        "std": deviation, "ci95_low": ci_low, "ci95_high": ci_high,
    }


def summarize_resource_field(resources, field):
    per_run = []
    for resource in resources:
        if (
            resource.get("status") == "ok"
            and resource.get("sample_count", 0) >= MIN_RESOURCE_SAMPLES
            and resource.get(field) is not None
        ):
            per_run.append(float(resource[field]))
        else:
            per_run.append(None)
    valid_values = [v for v in per_run if v is not None]
    return {
        "per_run": per_run,
        "valid_run_count": len(valid_values),
        **numeric_summary(valid_values),
    }


def summarize_resources(metrics):
    resources = [item.get("resources") or {} for item in metrics]
    statuses = [resource.get("status", "missing") for resource in resources]
    sample_counts = [int(resource.get("sample_count", 0)) for resource in resources]
    valid_runs = sum(
        status == "ok" and samples >= MIN_RESOURCE_SAMPLES
        for status, samples in zip(statuses, sample_counts)
    )
    return {
        "status": (
            "all_runs_valid" if valid_runs == len(metrics)
            else "partial_or_insufficient"
        ),
        "valid_runs": valid_runs,
        "total_runs": len(metrics),
        "status_per_run": statuses,
        "sample_count_per_run": sample_counts,
        "peak_memory_mib": summarize_resource_field(resources, "peak_memory_mib"),
        "mean_cpu_percent": summarize_resource_field(resources, "mean_cpu_percent"),
        "peak_cpu_percent": summarize_resource_field(resources, "peak_cpu_percent"),
    }


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path, fieldnames, rows):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def config_from_args(args):
    return {
        "db": args.db,
        "dataset": args.dataset,
        "runs": args.runs,
        "ef_search": args.ef_search,
        "chroma_candidates": args.chroma_candidates,
        "warmup_queries": args.warmup_queries,
        "clients": args.clients,
        "mode": args.mode,
        "run_timeout_seconds": args.run_timeout_seconds,
    }


def validate_config(config):
    db = config["db"]
    dataset = config["dataset"]
    if db not in ("chroma", "pgvector") or dataset not in DATASETS:
        raise ValueError("Unknown database or dataset")
    if config["runs"] < 2:
        raise ValueError("At least two runs required")
    if config["mode"] not in ("development_pilot", "official"):
        raise ValueError("Invalid mode")
    if config["mode"] == "official" and config["runs"] < 5:
        raise ValueError("Official mode requires at least five runs")
    if config["ef_search"] < 1:
        raise ValueError("--ef-search must be positive")
    if config["clients"] not in (1, 8):
        raise ValueError("--clients must be 1 or 8")
    if config["warmup_queries"] < 0:
        raise ValueError("Warmup count cannot be negative")
    if config["run_timeout_seconds"] < 1:
        raise ValueError("Run timeout must be positive")
    if config["chroma_candidates"] < 10:
        raise ValueError("Chroma candidates must be at least 10")
    if db == "pgvector" and config["chroma_candidates"] != 10:
        raise ValueError("Chroma candidates apply only to Chroma")
    if db == "chroma":
        smallest_filter_count = DATASETS[dataset]["expected_vectors"] // 100
        if config["chroma_candidates"] > smallest_filter_count:
            raise ValueError(
                "Chroma candidate count exceeds the 1% filter capacity "
                f"({smallest_filter_count})"
            )


def child_command(config):
    command = [
        sys.executable, "-u", "-m", "src.run_queries",
        "--db", config["db"],
        "--dataset", config["dataset"],
        "--warmup-queries", str(config["warmup_queries"]),
        "--clients", str(config["clients"]),
        "--mode", config["mode"],
    ]
    if config["db"] == "pgvector":
        command.extend(["--ef-search", str(config["ef_search"])])
    else:
        command.extend(["--chroma-candidates", str(config["chroma_candidates"])])
    return command


def source_fingerprints():
    names = (
        "src/run_queries.py", "src/run_repetitions.py",
        "src/adapters/chroma_adapter.py", "src/adapters/pgvector_adapter.py",
        "src/resource_monitor.py",
    )
    return {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in names
    }


def validate_result(result, config, number, reference=None):
    expected = {
        "mode": config["mode"], "database": config["db"],
        "dataset_key": config["dataset"], "clients": config["clients"],
        "k": 10, "metric": "L2",
        "warmup_queries_per_client_per_filter": config["warmup_queries"],
        "ef_search": config["ef_search"] if config["db"] == "pgvector" else None,
        "chroma_candidates": (
            config["chroma_candidates"] if config["db"] == "chroma" else None
        ),
    }
    for field, value in expected.items():
        if result.get(field) != value:
            raise ValueError(
                f"Run {number}: invalid {field}: "
                f"{result.get(field)!r}, expected {value!r}"
            )
    if result.get("client_model") != "closed_loop":
        raise ValueError(f"Run {number}: unexpected client model")
    if not isinstance(result.get("queries"), int) or result["queries"] < 8:
        raise ValueError(f"Run {number}: invalid query count")
    if reference is not None:
        for field in (
            "dataset", "ground_truth_file", "target", "base_vectors",
            "dimensions", "queries", "k", "metric", "client_model",
            "warmup_queries_per_client_per_filter", "ef_search",
            "chroma_candidates", "search_strategy", "mode",
        ):
            if result.get(field) != reference.get(field):
                raise ValueError(f"Run {number}: inconsistent {field}")
    filters = result.get("filters")
    if not isinstance(filters, dict) or set(filters) != set(FILTERS):
        raise ValueError(f"Run {number}: incomplete filters")
    for selectivity in FILTERS:
        metrics = filters[selectivity]
        if not isinstance(metrics, dict):
            raise ValueError(f"Run {number}: invalid filter {selectivity}%")
        for field in METRIC_FIELDS.values():
            value = metrics.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Run {number}: invalid {field} in {selectivity}%")
        if not 0 <= metrics["mean_recall_at_10"] <= 1:
            raise ValueError(f"Run {number}: invalid Recall@10")
        if metrics["qps"] <= 0 or metrics["latency_p50_ms"] <= 0:
            raise ValueError(f"Run {number}: invalid performance metrics")
        if metrics.get("queries_with_less_than_k", 0):
            raise ValueError(f"Run {number}: short query results in {selectivity}%")


def terminate_child(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execute_run(config, number, folder):
    command = child_command(config)
    log_path = folder / f"run_{number:02d}.log"
    captured_paths = []
    prefix = "Results saved to: "
    print(f"\n===== RUN {number}/{config['runs']} =====", flush=True)
    print(f"Log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        ) as process:
            def forward():
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    if line.startswith(prefix):
                        captured_paths.append(line[len(prefix):].strip())
            reader = threading.Thread(target=forward, daemon=True)
            reader.start()
            try:
                return_code = process.wait(timeout=config["run_timeout_seconds"])
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                terminate_child(process)
                reader.join(timeout=10)
                raise
            reader.join(timeout=10)
    if return_code != 0:
        raise RuntimeError(f"Run {number} failed (exit code {return_code})")
    if len(captured_paths) != 1:
        raise RuntimeError(f"Run {number}: expected exactly one result file")
    path = Path(captured_paths[0])
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path), json.loads(path.read_text(encoding="utf-8"))


def build_summary(results, config):
    reference = results[0]
    summary = {
        "mode": config["mode"], "timestamp_utc": utc_now(),
        "database": config["db"], "dataset_key": config["dataset"],
        "dataset": reference["dataset"],
        "ground_truth_file": reference.get("ground_truth_file"),
        "target": reference["target"], "base_vectors": reference["base_vectors"],
        "dimensions": reference["dimensions"],
        "queries_per_run": reference["queries"], "k": reference["k"],
        "metric": reference["metric"], "clients": config["clients"],
        "client_model": "closed_loop", "repetitions": config["runs"],
        "ef_search": reference.get("ef_search"),
        "chroma_candidates": reference.get("chroma_candidates"),
        "search_strategy": reference.get("search_strategy"),
        "warmup_queries_per_client_per_filter": config["warmup_queries"],
        "filters": {}, "warnings": [],
    }
    if reference["queries"] < 1000:
        summary["warnings"].append(
            "Fewer than 1000 measured queries per filter per run; "
            "P99 is exploratory."
        )
    if config["runs"] < 5:
        summary["warnings"].append(
            "Fewer than five repetitions; not sufficient for the official protocol."
        )
    for selectivity in FILTERS:
        measurements = [result["filters"][selectivity] for result in results]
        values = {}
        for name, field in METRIC_FIELDS.items():
            per_run = [float(item[field]) for item in measurements]
            stats = numeric_summary(per_run)
            values[f"{name}_per_run"] = per_run
            for statistic, number in stats.items():
                values[f"{statistic}_{name}"] = number
        values["resources"] = summarize_resources(measurements)
        summary["filters"][selectivity] = values
        if values["resources"]["valid_runs"] != config["runs"]:
            summary["warnings"].append(
                f"Filter {selectivity}%: missing/insufficient resource samples."
            )
    summary["resource_measurements"] = (
        "Resources are aggregated only when status='ok' and at least "
        "three samples were collected; see per-run JSON for full details."
    )
    return summary


def write_csv_outputs(folder, results, summary, config):
    base = {
        "mode": config["mode"], "database": config["db"],
        "dataset": config["dataset"], "clients": config["clients"],
        "ef_search": config["ef_search"] if config["db"] == "pgvector" else "",
        "chroma_candidates": (
            config["chroma_candidates"] if config["db"] == "chroma" else ""
        ),
    }
    per_run = []
    for index, result in enumerate(results, start=1):
        for selectivity in FILTERS:
            metrics = result["filters"][selectivity]
            resources = metrics.get("resources") or {}
            per_run.append({
                **base, "run": index, "filter_percent": selectivity,
                "mean_recall_at_10": metrics["mean_recall_at_10"],
                "qps": metrics["qps"],
                "p50_ms": metrics["latency_p50_ms"],
                "p95_ms": metrics["latency_p95_ms"],
                "p99_ms": metrics["latency_p99_ms"],
                "resource_status": resources.get("status", "missing"),
                "resource_samples": resources.get("sample_count", 0),
                "peak_memory_mib": resources.get("peak_memory_mib"),
                "mean_cpu_percent": resources.get("mean_cpu_percent"),
            })
    atomic_csv(folder / "per_run.csv", list(per_run[0]), per_run)
    aggregated = []
    for selectivity in FILTERS:
        row = {**base, "filter_percent": selectivity,
               "runs": config["runs"]}
        values = summary["filters"][selectivity]
        for metric in METRIC_FIELDS:
            for statistic in ("mean", "median", "std", "ci95_low", "ci95_high"):
                row[f"{statistic}_{metric}"] = values[f"{statistic}_{metric}"]
        row["valid_resource_runs"] = values["resources"]["valid_runs"]
        aggregated.append(row)
    atomic_csv(folder / "summary.csv", list(aggregated[0]), aggregated)


def self_test():
    stats = numeric_summary([1, 2, 3, 4, 5])
    assert stats["mean"] == 3
    assert stats["std"] is not None
    assert stats["ci95_low"] < 3 < stats["ci95_high"]
    assert summarize_resources([{"resources": {"status": "missing"}}])["valid_runs"] == 0
    candidate_config = {
        "db": "chroma", "dataset": "1m", "runs": 5,
        "ef_search": 80, "chroma_candidates": 10000,
        "warmup_queries": 2, "clients": 1,
        "mode": "development_pilot", "run_timeout_seconds": 3600,
    }
    validate_config(candidate_config)
    assert "--chroma-candidates" in child_command(candidate_config)
    try:
        validate_config({**candidate_config, "chroma_candidates": 15000})
    except ValueError:
        pass
    else:
        raise AssertionError("Unsafe filter candidate count accepted")
    try:
        validate_config({**candidate_config, "mode": "official", "runs": 2})
    except ValueError:
        pass
    else:
        raise AssertionError("Under-repeated official run accepted")
    print("REPETITIONS SELF-TEST PASSED — NO QUERIES EXECUTED")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resumable repeated Chroma/pgvector query experiments"
    )
    parser.add_argument("--db", choices=["chroma", "pgvector"])
    parser.add_argument("--dataset", choices=DATASET_KEYS, default="10k")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--ef-search", type=int, default=80)
    parser.add_argument("--chroma-candidates", type=int, default=10)
    parser.add_argument("--clients", type=int, choices=[1, 8], default=1)
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument(
        "--mode", choices=["development_pilot", "official"],
        default="development_pilot",
    )
    parser.add_argument("--run-timeout-seconds", type=int, default=7200)
    parser.add_argument("--resume", type=Path, help="Resume an interrupted session")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands only; do not connect to databases")
    parser.add_argument("--self-test", action="store_true",
                        help="Run code checks without any database queries")
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.resume:
        folder = args.resume.resolve()
        manifest = json.loads((folder / "manifest.json").read_text("utf-8"))
        config = manifest["config"]
        validate_config(config)
        if source_fingerprints() != manifest["source_fingerprints"]:
            parser.error("Source code has changed since this session began")
    else:
        if args.db is None:
            parser.error("--db is required unless --resume is used")
        config = config_from_args(args)
        try:
            validate_config(config)
        except ValueError as exc:
            parser.error(str(exc))
        folder = None
    print("\n=== REPETITIONS CONFIGURATION ===", flush=True)
    print(json.dumps(config, indent=2), flush=True)
    print("Command:", " ".join(child_command(config)), flush=True)
    if args.dry_run:
        print("DRY RUN PASSED — NO DATABASE QUERIES EXECUTED")
        return
    if folder is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        session_id = f"{stamp}_{uuid4().hex[:8]}"
        folder = Path("results") / (
            "official" if config["mode"] == "official" else "pilot"
        ) / "repetitions" / session_id
        folder.mkdir(parents=True, exist_ok=False)
        manifest = {
            "session_id": session_id,
            "created_utc": utc_now(),
            "status": "running",
            "config": config,
            "source_fingerprints": source_fingerprints(),
            "completed_runs": [],
        }
        atomic_json(folder / "manifest.json", manifest)
    print(f"Session: {folder}", flush=True)
    results = []
    paths = []
    try:
        for number in range(1, config["runs"] + 1):
            local_path = folder / f"run_{number:02d}.json"
            if local_path.is_file():
                result = json.loads(local_path.read_text(encoding="utf-8"))
                validate_result(result, config, number, results[0] if results else None)
                print(f"Run {number}: validated existing result (resume)", flush=True)
            else:
                source_path, result = execute_run(config, number, folder)
                validate_result(result, config, number, results[0] if results else None)
                atomic_json(local_path, result)
                print(f"Run {number}/{config['runs']} VALIDATED", flush=True)
                print(f"Source result: {source_path}", flush=True)
            results.append(result)
            paths.append(str(local_path))
            manifest["status"] = "running"
            manifest["completed_runs"] = list(range(1, len(results) + 1))
            atomic_json(folder / "manifest.json", manifest)
        summary = build_summary(results, config)
        summary["run_files"] = paths
        summary["session_directory"] = str(folder)
        atomic_json(folder / "summary.json", summary)
        write_csv_outputs(folder, results, summary, config)
        manifest["status"] = "completed"
        manifest["completed_utc"] = utc_now()
        manifest["completed_runs"] = list(range(1, config["runs"] + 1))
        atomic_json(folder / "manifest.json", manifest)
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["last_error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(folder / "manifest.json", manifest)
        print(f"Session incomplete: {folder}", file=sys.stderr, flush=True)
        print(f"Resume with: python -m src.run_repetitions --resume {folder}",
              file=sys.stderr, flush=True)
        raise
    print("\n===== REPETITION SUMMARY =====")
    for selectivity in FILTERS:
        values = summary["filters"][selectivity]
        print(
            f"Filter {selectivity:>3}% | "
            f"Recall@10={values['mean_recall_at_10']:.4f} | "
            f"median QPS={values['median_qps']:.2f} | "
            f"median p95={values['median_p95_ms']:.2f} ms | "
            f"resource runs={values['resources']['valid_runs']}/{config['runs']}",
            flush=True,
        )
    for warning in summary["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Summary JSON: {folder / 'summary.json'}")
    print(f"Per-run CSV: {folder / 'per_run.csv'}")
    print(f"Summary CSV: {folder / 'summary.csv'}")
    print("REPETITIONS COMPLETED")


if __name__ == "__main__":
    main()
