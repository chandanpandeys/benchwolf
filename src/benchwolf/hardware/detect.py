"""Best-effort cross-platform hardware detection for BenchWolf."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path

import psutil

from benchwolf.models import HardwareProfile


def _run(command: list[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _cpu_name() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
                if line.startswith("model name") or line.startswith("Hardware"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        value = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if value:
            return value
    elif system == "Windows":
        value = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if value:
            return value.splitlines()[0].strip()
    return platform.processor() or "Unknown CPU"


def _gpu_name() -> str | None:
    value = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"])
    if value:
        return value.splitlines()[0].strip()
    if platform.system() == "Darwin":
        value = _run(["system_profiler", "SPDisplaysDataType"])
        for line in value.splitlines():
            if "Chipset Model:" in line:
                return line.split(":", 1)[1].strip()
    if platform.system() == "Windows":
        value = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_VideoController).Name -join ','",
            ]
        )
        if value:
            return value.split(",")[0].strip()
    return None


def _npu_name(cpu_name: str) -> str | None:
    lower = cpu_name.lower()
    if "apple m" in lower or platform.system() == "Darwin":
        return "Apple Neural Engine"
    if "snapdragon" in lower or "x elite" in lower or "x plus" in lower:
        return "Qualcomm Hexagon NPU"
    if "ryzen ai" in lower:
        return "AMD Ryzen AI NPU"
    if "core ultra" in lower or "intel ultra" in lower:
        return "Intel AI Boost NPU"
    if Path("/sys/class/misc/npu").exists():
        return "Linux NPU device"
    return None


def _estimated_bandwidth(cpu_name: str, gpu_name: str | None, ram_gb: float) -> float | None:
    """Return a coarse peak-bandwidth estimate used only by preflight heuristics."""
    name = f"{cpu_name} {gpu_name or ''}".lower()
    apple = {
        "m4 ultra": 800.0,
        "m4 max": 546.0,
        "m4 pro": 273.0,
        "m4": 120.0,
        "m3 max": 400.0,
        "m3 pro": 150.0,
        "m3": 100.0,
        "m2 ultra": 800.0,
        "m2 max": 400.0,
        "m2 pro": 200.0,
        "m2": 100.0,
        "m1 ultra": 800.0,
        "m1 max": 400.0,
        "m1 pro": 200.0,
        "m1": 68.0,
    }
    for needle, bandwidth in apple.items():
        if needle in name:
            return bandwidth
    gpu_estimates = {
        "4090": 1008.0,
        "4080": 716.0,
        "4070": 504.0,
        "3090": 936.0,
        "3080": 760.0,
        "3060": 360.0,
    }
    for needle, bandwidth in gpu_estimates.items():
        if needle in name:
            return bandwidth
    if "ryzen ai max" in name:
        return 270.0
    if "snapdragon" in name or "x elite" in name:
        return 135.0
    if ram_gb >= 16:
        return 64.0
    if ram_gb >= 8:
        return 38.4
    return 25.0


def _storage_type() -> str | None:
    if platform.system() == "Linux":
        value = _run(["lsblk", "-d", "-o", "ROTA", "--noheadings"])
        values = {line.strip() for line in value.splitlines() if line.strip()}
        if "0" in values:
            return "SSD/NVMe"
        if "1" in values:
            return "HDD"
    return None


def detect_hardware() -> HardwareProfile:
    vm = psutil.virtual_memory()
    cpu = _cpu_name()
    gpu = _gpu_name()
    freq = psutil.cpu_freq()
    ram_gb = vm.total / (1024**3)
    return HardwareProfile(
        cpu_name=cpu,
        cpu_arch=platform.machine(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        cpu_freq_mhz=(freq.current if freq else 0.0) or 0.0,
        ram_total_gb=ram_gb,
        ram_available_gb=vm.available / (1024**3),
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        gpu_name=gpu,
        npu_name=_npu_name(cpu),
        storage_type=_storage_type(),
        memory_bandwidth_gb_s=_estimated_bandwidth(cpu, gpu, ram_gb),
    )
