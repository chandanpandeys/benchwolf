"""Power measurement for InferBox.

Tiered approach:
  Tier 1: Intel RAPL (pyRAPL) — most accurate on x86
  Tier 2: Linux hwmon sysfs — ARM devices
  Tier 3: Battery drain rate — laptops
  Tier 4: TDP estimation — fallback for any platform
"""

from __future__ import annotations

import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import psutil


@dataclass
class PowerSample:
    """A single power measurement sample."""
    timestamp: float
    watts: float
    method: str


@dataclass
class PowerMeasurement:
    """Collected power measurement data."""
    method: str = "unknown"
    samples: list[PowerSample] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def avg_watts(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.watts for s in self.samples) / len(self.samples)

    @property
    def max_watts(self) -> float:
        if not self.samples:
            return 0.0
        return max(s.watts for s in self.samples)


class PowerMeter:
    """Multi-tier power measurement."""

    def __init__(self):
        self._method = self._detect_best_method()
        self._measurement: Optional[PowerMeasurement] = None
        self._sampling = False

    def _detect_best_method(self) -> str:
        """Detect the best available power measurement method."""
        system = platform.system()

        # Tier 1: Intel RAPL
        if system == "Linux":
            rapl_path = Path("/sys/class/powercap/intel-rapl")
            if rapl_path.exists():
                return "rapl"

        # Tier 2: Linux hwmon (ARM)
        if system == "Linux":
            hwmon_path = Path("/sys/class/hwmon")
            if hwmon_path.exists():
                for hwmon_dir in hwmon_path.iterdir():
                    power_file = hwmon_dir / "power1_input"
                    if power_file.exists():
                        return "hwmon"

        # Tier 3: Battery
        battery = psutil.sensors_battery()
        if battery is not None and not battery.power_plugged:
            return "battery"

        # Tier 4: Estimation
        return "estimate"

    @property
    def method(self) -> str:
        return self._method

    def start(self) -> None:
        """Start power measurement."""
        self._measurement = PowerMeasurement(
            method=self._method,
            start_time=time.time(),
        )
        self._sampling = True

    def sample(self) -> Optional[float]:
        """Take a single power sample. Returns watts or None."""
        if not self._sampling or not self._measurement:
            return None

        watts = self._read_power()
        if watts is not None and watts > 0:
            sample = PowerSample(
                timestamp=time.time(),
                watts=watts,
                method=self._method,
            )
            self._measurement.samples.append(sample)
            return watts
        return None

    def stop(self) -> PowerMeasurement:
        """Stop measurement and return results."""
        self._sampling = False
        if self._measurement:
            self._measurement.end_time = time.time()
            return self._measurement
        return PowerMeasurement(method="none")

    def _read_power(self) -> Optional[float]:
        """Read current power consumption in watts."""
        if self._method == "rapl":
            return self._read_rapl()
        elif self._method == "hwmon":
            return self._read_hwmon()
        elif self._method == "battery":
            return self._read_battery()
        elif self._method == "estimate":
            return self._estimate_power()
        return None

    def _read_rapl(self) -> Optional[float]:
        """Read Intel RAPL power in watts."""
        try:
            rapl_base = Path("/sys/class/powercap/intel-rapl/intel-rapl:0")
            energy_file = rapl_base / "energy_uj"

            if not energy_file.exists():
                return None

            # Read energy, wait, read again to compute power
            energy_before = int(energy_file.read_text().strip())
            time.sleep(0.1)
            energy_after = int(energy_file.read_text().strip())

            # Handle counter wrap-around
            if energy_after < energy_before:
                max_range = int((rapl_base / "max_energy_range_uj").read_text().strip())
                energy_diff = (max_range - energy_before) + energy_after
            else:
                energy_diff = energy_after - energy_before

            watts = energy_diff / (0.1 * 1_000_000)  # µJ to W over 0.1s
            return watts
        except (FileNotFoundError, ValueError, PermissionError):
            return None

    def _read_hwmon(self) -> Optional[float]:
        """Read power from Linux hwmon sysfs."""
        try:
            hwmon_path = Path("/sys/class/hwmon")
            for hwmon_dir in hwmon_path.iterdir():
                power_file = hwmon_dir / "power1_input"
                if power_file.exists():
                    # power1_input is in microwatts
                    microwatts = int(power_file.read_text().strip())
                    return microwatts / 1_000_000
        except (FileNotFoundError, ValueError, PermissionError):
            pass
        return None

    def _read_battery(self) -> Optional[float]:
        """Estimate power from battery drain rate."""
        try:
            battery = psutil.sensors_battery()
            if battery is None or battery.power_plugged:
                return None

            # psutil gives secsleft — estimate watts from system TDP
            # This is a rough estimate
            pct1 = battery.percent
            time.sleep(0.5)
            battery2 = psutil.sensors_battery()
            if battery2 is None:
                return None
            pct2 = battery2.percent

            if pct1 == pct2:
                # Can't measure, use TDP estimate
                return self._estimate_power()

            # Typical laptop battery is ~50Wh
            battery_wh = 50.0
            drain_rate = (pct1 - pct2) / 0.5 * 3600  # %/hour
            watts = (drain_rate / 100) * battery_wh
            return max(watts, 1.0)  # At least 1W
        except Exception:
            return None

    def _estimate_power(self) -> Optional[float]:
        """Estimate power from CPU utilization and TDP."""
        try:
            cpu_pct = psutil.cpu_percent(interval=0.1)
            freq = psutil.cpu_freq()
            cores = psutil.cpu_count(logical=False) or 1

            # Rough TDP estimation
            # Mobile ARM: ~5-15W, Desktop x86: ~15-125W, Laptop: ~15-45W
            arch = platform.machine().lower()
            if "arm" in arch or "aarch" in arch:
                base_tdp = 10.0
            elif psutil.sensors_battery() is not None:
                base_tdp = 25.0  # Laptop
            else:
                base_tdp = 65.0  # Desktop

            # Scale by utilization
            estimated_watts = base_tdp * (cpu_pct / 100) * 0.7 + base_tdp * 0.3
            return round(estimated_watts, 1)
        except Exception:
            return None
