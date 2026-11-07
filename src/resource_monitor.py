import json
import re
import subprocess

from datetime import datetime, timezone
from statistics import mean
from threading import Event, Thread
from time import perf_counter


MIN_SAMPLES = 3


def memory_to_mib(value):
    match = re.fullmatch(
        r"([\d.]+)\s*([A-Za-z]+)",
        value.strip(),
    )

    if not match:
        raise ValueError(
            f"Unknown memory value: {value}"
        )

    amount = float(match.group(1))
    unit = match.group(2)

    factors = {
        "B": 1 / (1024 * 1024),
        "KiB": 1 / 1024,
        "MiB": 1,
        "GiB": 1024,
        "TiB": 1024 * 1024,
        "kB": 1000 / (1024 * 1024),
        "MB": 1000 * 1000 / (1024 * 1024),
        "GB": 1000 * 1000 * 1000 / (1024 * 1024),
    }

    if unit not in factors:
        raise ValueError(
            f"Unsupported unit: {unit}"
        )

    return amount * factors[unit]


def summarize_samples(samples, container_name):
    count = len(samples)

    if count == 0:
        return {
            "container": container_name,
            "sample_count": 0,
            "status": "insufficient_samples",
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
        "container": container_name,
        "sample_count": count,
        "status": (
            "ok"
            if count >= MIN_SAMPLES
            else "insufficient_samples"
        ),
        "mean_cpu_percent": float(mean(cpu)),
        "peak_cpu_percent": float(max(cpu)),
        "peak_memory_mib": float(max(memory)),
        "last_memory_mib": float(memory[-1]),
    }


class ResourceMonitor:
    def __init__(self, container_name):
        self.container_name = container_name

        self.samples = []
        self.error = None

        self.stop_event = Event()

        self.thread = None
        self.process = None

    def _collect(self):
        try:
            for line in self.process.stdout:
                if self.stop_event.is_set():
                    break

                clean_line = re.sub(
                    r"\x1b\[[0-?]*[ -/]*[@-~]",
                    "",
                    line,
                ).strip()

                if not clean_line:
                    continue

                data = json.loads(clean_line)

                cpu_percent = float(
                    data["CPUPerc"]
                    .replace("%", "")
                    .strip()
                )

                memory_used = (
                    data["MemUsage"].split("/")[0]
                )

                # Same monotonic clock as perf_counter()
                # in the benchmark runner.
                sampled_at = perf_counter()

                self.samples.append({
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "timestamp_monotonic": sampled_at,
                    "cpu_percent": cpu_percent,
                    "memory_mib": memory_to_mib(
                        memory_used
                    ),
                })

            if not self.stop_event.is_set():
                self.error = RuntimeError(
                    "docker stats stopped unexpectedly"
                )

        except Exception as exc:
            self.error = exc

    def start(self):
        if self.thread is not None:
            raise RuntimeError(
                "Monitor already started"
            )

        self.process = subprocess.Popen(
            [
                "docker",
                "stats",
                "--format",
                "{{json .}}",
                self.container_name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self.thread = Thread(
            target=self._collect,
            daemon=True,
        )

        self.thread.start()

    def summarize_interval(
        self,
        started_at,
        ended_at,
    ):
        """
        Summarize samples collected inside a workload.

        started_at and ended_at must come from
        time.perf_counter().

        If fewer than 3 samples exist, the result is
        marked insufficient_samples.
        """

        if ended_at < started_at:
            raise ValueError(
                "Interval end is before interval start"
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

        result = summarize_samples(
            interval_samples,
            self.container_name,
        )

        result["window_duration_seconds"] = float(
            ended_at - started_at
        )

        result["scope"] = "measured_workload_only"

        return result

    def stop(self):
        if self.thread is None:
            raise RuntimeError(
                "Monitor not started"
            )

        self.stop_event.set()

        if self.process.poll() is None:
            self.process.terminate()

        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

        self.thread.join(timeout=5)

        if self.thread.is_alive():
            raise RuntimeError(
                "Monitor thread did not stop"
            )

        if self.error is not None:
            raise RuntimeError(
                f"Resource monitoring failed: {self.error}"
            )

        return summarize_samples(
            self.samples,
            self.container_name,
        )
