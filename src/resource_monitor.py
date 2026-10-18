import json
import re
import subprocess

from datetime import datetime, timezone
from statistics import mean
from threading import Event, Thread


def memory_to_mib(value):
    match = re.fullmatch(
        r"([\d.]+)\s*([A-Za-z]+)",
        value.strip(),
    )

    if not match:
        raise ValueError(f"Unknown memory value: {value}")

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
        raise ValueError(f"Unsupported unit: {unit}")

    return amount * factors[unit]


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
                    data["CPUPerc"].replace("%", "").strip()
                )

                memory_used = data["MemUsage"].split("/")[0]

                self.samples.append({
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_mib": memory_to_mib(memory_used),
                })

            if not self.stop_event.is_set():
                self.error = RuntimeError(
                    "docker stats stopped unexpectedly"
                )

        except Exception as exc:
            self.error = exc

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Monitor already started")

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

    def stop(self):
        if self.thread is None:
            raise RuntimeError("Monitor not started")

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
            raise RuntimeError("Monitor thread did not stop")

        if self.error is not None:
            raise RuntimeError(
                f"Resource monitoring failed: {self.error}"
            )

        if not self.samples:
            return {
                "container": self.container_name,
                "sample_count": 0,
                "status": "insufficient_samples",
                "mean_cpu_percent": None,
                "peak_cpu_percent": None,
                "peak_memory_mib": None,
                "last_memory_mib": None,
            }

        cpu = [s["cpu_percent"] for s in self.samples]
        memory = [s["memory_mib"] for s in self.samples]

        return {
            "container": self.container_name,
            "sample_count": len(self.samples),
            "status": (
                "ok"
                if len(self.samples) >= 3
                else "insufficient_samples"
            ),
            "mean_cpu_percent": float(mean(cpu)),
            "peak_cpu_percent": float(max(cpu)),
            "peak_memory_mib": float(max(memory)),
            "last_memory_mib": float(memory[-1]),
        }
