"""Cross-platform power sampling with explicit measurement provenance."""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from benchwolf.config import POWER_SAMPLE_INTERVAL_SECONDS


@dataclass
class PowerSample:
    timestamp: float
    watts: float


@dataclass
class PowerMeasurement:
    method: str = "unavailable"
    source: str = "unavailable"
    samples: list[PowerSample] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def avg_watts(self) -> float | None:
        if not self.samples:
            return None
        return sum(sample.watts for sample in self.samples) / len(self.samples)


class PowerMeter:
    """Sample power in a background thread while inference runs."""

    def __init__(self, sample_interval_s: float = POWER_SAMPLE_INTERVAL_SECONDS):
        self._method = self._detect_best_method()
        self._source = "estimated" if self._method in {"estimate", "battery"} else "measured"
        if self._method == "unavailable":
            self._source = "unavailable"
        self._interval = max(0.05, sample_interval_s)
        self._measurement: PowerMeasurement | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def method(self) -> str:
        return self._method

    @property
    def source(self) -> str:
        return self._source

    def _detect_best_method(self) -> str:
        if platform.system() == "Linux":
            energy = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")
            if energy.exists():
                return "rapl"
            hwmon = Path("/sys/class/hwmon")
            if hwmon.exists():
                for directory in hwmon.iterdir():
                    if (directory / "power1_input").exists():
                        return "hwmon"
        battery = psutil.sensors_battery()
        if battery is not None and not battery.power_plugged:
            return "battery"
        return "estimate"

    def start(self) -> None:
        self._stop_event.clear()
        self._measurement = PowerMeasurement(
            method=self._method,
            source=self._source,
            start_time=time.perf_counter(),
        )
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def _sampling_loop(self) -> None:
        while not self._stop_event.is_set():
            watts = self._read_power()
            if watts is not None and watts > 0 and self._measurement is not None:
                self._measurement.samples.append(PowerSample(time.perf_counter(), watts))
            self._stop_event.wait(self._interval)

    def stop(self) -> PowerMeasurement:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 4))
        if self._measurement is None:
            return PowerMeasurement()
        self._measurement.end_time = time.perf_counter()
        if not self._measurement.samples:
            self._measurement.source = "unavailable"
        return self._measurement

    def _read_power(self) -> float | None:
        if self._method == "rapl":
            return self._read_rapl()
        if self._method == "hwmon":
            return self._read_hwmon()
        if self._method == "battery":
            return self._read_battery()
        if self._method == "estimate":
            return self._estimate_power()
        return None

    def _read_rapl(self) -> float | None:
        base = Path("/sys/class/powercap/intel-rapl/intel-rapl:0")
        energy_file = base / "energy_uj"
        try:
            before = int(energy_file.read_text(encoding="utf-8").strip())
            started = time.perf_counter()
            time.sleep(0.1)
            after = int(energy_file.read_text(encoding="utf-8").strip())
            elapsed = time.perf_counter() - started
            if after < before:
                max_range = int((base / "max_energy_range_uj").read_text().strip())
                delta = max_range - before + after
            else:
                delta = after - before
            return delta / 1_000_000.0 / max(elapsed, 0.001)
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            return None

    def _read_hwmon(self) -> float | None:
        try:
            for directory in Path("/sys/class/hwmon").iterdir():
                power_file = directory / "power1_input"
                if power_file.exists():
                    return int(power_file.read_text().strip()) / 1_000_000.0
        except (FileNotFoundError, PermissionError, ValueError, OSError):
            return None
        return None

    def _read_battery(self) -> float | None:
        if platform.system() == "Linux":
            try:
                for battery in Path("/sys/class/power_supply").glob("BAT*"):
                    power_file = battery / "power_now"
                    if power_file.exists():
                        value = int(power_file.read_text().strip()) / 1_000_000.0
                        if value > 0:
                            return value
            except (PermissionError, ValueError, OSError):
                pass

        battery = psutil.sensors_battery()
        if battery is None or battery.power_plugged:
            return None
        if battery.secsleft and battery.secsleft > 0:
            remaining_wh = battery.percent / 100.0 * 55.0
            hours = battery.secsleft / 3600.0
            if hours > 0:
                value = remaining_wh / hours
                if 2.0 <= value <= 150.0:
                    return value
        return self._estimate_power()

    def _estimate_power(self) -> float | None:
        try:
            cpu_pct = psutil.cpu_percent(interval=0.1)
            arch = platform.machine().lower()
            if "arm" in arch or "aarch" in arch:
                tdp = 10.0
            elif psutil.sensors_battery() is not None:
                tdp = 25.0
            else:
                tdp = 65.0
            return round(tdp * (0.3 + 0.7 * cpu_pct / 100.0), 1)
        except Exception:
            return None
