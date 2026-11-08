import re
import subprocess

from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from threading import Event, Thread
from time import perf_counter


MIN_SAMPLES = 3
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.1
CGROUP_ROOT = Path("/sys/fs/cgroup")


def memory_to_mib(value):
    match = re.fullmatch(
        r"([\d.]+)\s*([A-Za-z]+)",
        value.strip(),
    )

    if not match:
        raise ValueError(
            f"Unknown memory value: {value}"
        )

    factors = {
        "B": 1 / (1024 * 1024),
        "KiB": 1 / 1024,
        "MiB": 1,
        "GiB": 1024,
        "TiB": 1024 * 1024,
        "kB": 1000 / (1024 * 1024),
        "MB": 1000**2 / (1024 * 1024),
        "GB": 1000**3 / (1024 * 1024),
    }

    unit = match.group(2)

    if unit not in factors:
        raise ValueError(
            f"Unsupported unit: {unit}"
        )

    return float(match.group(1)) * factors[unit]


def summarize_samples(samples, container_name):
    count = len(samples)

    result = {
        "container": container_name,
        "sample_count": count,
        "status": (
            "ok"
            if count >= MIN_SAMPLES
            else "insufficient_samples"
        ),
        "source": "cgroup_v2",
        "memory_accounting": (
            "memory.current minus "
            "memory.stat inactive_file"
        ),
        "cpu_definition": (
            "100 percent equals one fully "
            "utilized CPU core"
        ),
        "sampling_interval_seconds": (
            DEFAULT_SAMPLE_INTERVAL_SECONDS
        ),
    }

    if count == 0:
        return {
            **result,
            "mean_cpu_percent": None,
            "peak_cpu_percent": None,
            "peak_memory_mib": None,
            "last_memory_mib": None,
        }

    cpu = [
        sample["cpu_percent"]
        for sample in samples
    ]

    memory = [
        sample["memory_mib"]
        for sample in samples
    ]

    return {
        **result,
        "mean_cpu_percent": float(mean(cpu)),
        "peak_cpu_percent": float(max(cpu)),
        "peak_memory_mib": float(max(memory)),
        "last_memory_mib": float(memory[-1]),
    }


def read_stat_file(path):
    entries = {}

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        name, value = line.split()
        entries[name] = int(value)

    return entries


class ResourceMonitor:
    def __init__(
        self,
        container_name,
        sample_interval_seconds=DEFAULT_SAMPLE_INTERVAL_SECONDS,
    ):
        if not 0.05 <= sample_interval_seconds <= 5.0:
            raise ValueError(
                "Sampling interval must be "
                "between 0.05 and 5 seconds"
            )

        self.container_name = container_name
        self.sample_interval_seconds = float(
            sample_interval_seconds
        )

        self.samples = []
        self.error = None
        self.stop_event = Event()
        self.thread = None
        self.cgroup_path = None

    def _locate_cgroup(self):
        process = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Pid}}",
                self.container_name,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

        pid = int(process.stdout.strip())

        if pid <= 0:
            raise RuntimeError(
                "Container is not running: "
                f"{self.container_name}"
            )

        lines = Path(
            f"/proc/{pid}/cgroup"
        ).read_text(
            encoding="utf-8"
        ).splitlines()

        paths = [
            line.split("::", 1)[1]
            for line in lines
            if line.startswith("0::")
        ]

        if len(paths) != 1:
            raise RuntimeError(
                "Expected cgroup v2; "
                "container cgroup not found"
            )

        root = CGROUP_ROOT.resolve()

        path = (
            root / paths[0].lstrip("/")
        ).resolve()

        if not path.is_relative_to(root):
            raise RuntimeError(
                "Invalid container cgroup path"
            )

        for name in (
            "cpu.stat",
            "memory.current",
            "memory.stat",
        ):
            if not (path / name).is_file():
                raise RuntimeError(
                    "Missing cgroup v2 file: "
                    f"{path / name}"
                )

        return path

    def _snapshot(self):
        path = self.cgroup_path

        cpu = read_stat_file(
            path / "cpu.stat"
        )

        cpu_usage_usec = cpu["usage_usec"]

        memory_current = int(
            (path / "memory.current")
            .read_text(encoding="utf-8")
            .strip()
        )

        memory_stats = read_stat_file(
            path / "memory.stat"
        )

        inactive_file = memory_stats[
            "inactive_file"
        ]

        adjusted_memory_bytes = max(
            0,
            memory_current - inactive_file,
        )

        memory_mib = (
            adjusted_memory_bytes
            / (1024 * 1024)
        )

        return (
            perf_counter(),
            cpu_usage_usec,
            memory_mib,
        )

    def _collect(self):
        try:
            (
                previous_time,
                previous_usage,
                _,
            ) = self._snapshot()

            while not self.stop_event.wait(
                self.sample_interval_seconds
            ):
                (
                    sampled_at,
                    usage,
                    memory_mib,
                ) = self._snapshot()

                elapsed = (
                    sampled_at - previous_time
                )

                cpu_delta = (
                    usage - previous_usage
                )

                if elapsed <= 0 or cpu_delta < 0:
                    raise RuntimeError(
                        "Invalid cgroup CPU "
                        "measurement interval"
                    )

                cpu_percent = (
                    cpu_delta
                    / 1_000_000
                    / elapsed
                    * 100.0
                )

                self.samples.append({
                    "timestamp_utc": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                    "timestamp_monotonic": sampled_at,
                    "cpu_percent": cpu_percent,
                    "memory_mib": memory_mib,
                })

                previous_time = sampled_at
                previous_usage = usage

        except Exception as exc:
            self.error = exc

    def start(self):
        if self.thread is not None:
            raise RuntimeError(
                "Monitor already started"
            )

        self.cgroup_path = (
            self._locate_cgroup()
        )

        self.stop_event.clear()

        self.thread = Thread(
            target=self._collect,
            daemon=True,
        )

        self.thread.start()

    def _summary(self, samples):
        result = summarize_samples(
            samples,
            self.container_name,
        )

        result[
            "sampling_interval_seconds"
        ] = self.sample_interval_seconds

        return result

    def summarize_interval(
        self,
        started_at,
        ended_at,
    ):
        if ended_at < started_at:
            raise ValueError(
                "Interval end is before "
                "interval start"
            )

        interval_samples = [
            sample
            for sample in self.samples
            if (
                started_at
                <= sample["timestamp_monotonic"]
                <= ended_at
            )
        ]

        result = self._summary(
            interval_samples
        )

        result[
            "window_duration_seconds"
        ] = float(
            ended_at - started_at
        )

        result["scope"] = (
            "measured_workload_only"
        )

        return result

    def stop(self):
        if self.thread is None:
            raise RuntimeError(
                "Monitor not started"
            )

        self.stop_event.set()

        self.thread.join(timeout=5)

        if self.thread.is_alive():
            raise RuntimeError(
                "Monitor thread did not stop"
            )

        if self.error is not None:
            raise RuntimeError(
                "Resource monitoring failed: "
                f"{self.error}"
            )

        return self._summary(
            self.samples
        )
