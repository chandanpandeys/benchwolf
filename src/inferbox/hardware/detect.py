"""Hardware detection for InferBox."""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Optional

import psutil

from inferbox.models import HardwareProfile


def detect_hardware() -> HardwareProfile:
    """Detect and return comprehensive hardware information."""
    return HardwareProfile(
        cpu_name=_get_cpu_name(),
        cpu_arch=platform.machine(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        cpu_freq_mhz=_get_cpu_freq(),
        ram_total_gb=psutil.virtual_memory().total / (1024**3),
        ram_available_gb=psutil.virtual_memory().available / (1024**3),
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        gpu_name=_detect_gpu(),
        npu_name=_detect_npu(),
        storage_type=_detect_storage(),
    )


def _get_cpu_name() -> str:
    """Get CPU model name."""
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.strip().startswith("model name"):
                        return line.split(":")[1].strip()
                    # ARM devices use "Hardware" or "Model"
                    if line.strip().startswith("Hardware"):
                        return line.split(":")[1].strip()
        except (FileNotFoundError, PermissionError):
            pass

    elif system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name", "/value"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split("\n"):
                if line.startswith("Name="):
                    return line.split("=", 1)[1].strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    elif system == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return platform.processor() or "Unknown CPU"


def _get_cpu_freq() -> float:
    """Get CPU frequency in MHz."""
    freq = psutil.cpu_freq()
    if freq:
        return freq.current or freq.max or 0.0
    return 0.0


def _detect_gpu() -> Optional[str]:
    """Detect GPU if available."""
    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Try ROCm for AMD
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.split("\n"):
                if "GPU" in line:
                    return line.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return None


def _detect_npu() -> Optional[str]:
    """Detect NPU if available (ARM devices)."""
    system = platform.system()
    if system != "Linux":
        return None

    # Check for Rockchip RKNN NPU
    try:
        from pathlib import Path
        rknn_path = Path("/sys/class/misc/npu")
        if rknn_path.exists():
            return "Rockchip RKNN NPU"
    except Exception:
        pass

    # Check for Qualcomm NPU
    try:
        result = subprocess.run(
            ["cat", "/sys/devices/platform/soc/*/subsys_name"],
            capture_output=True, text=True, timeout=5, shell=True
        )
        if "npu" in result.stdout.lower():
            return "Qualcomm Hexagon NPU"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return None


def _detect_storage() -> Optional[str]:
    """Detect storage type."""
    system = platform.system()

    if system == "Linux":
        try:
            result = subprocess.run(
                ["lsblk", "-d", "-o", "NAME,ROTA", "--noheadings"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rota = parts[1]
                        if rota == "0":
                            return "SSD/NVMe"
                        elif rota == "1":
                            return "HDD"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Check for SD card (common on RPi)
        try:
            from pathlib import Path
            if Path("/sys/block/mmcblk0").exists():
                return "SD Card"
        except Exception:
            pass

    elif system == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-Command",
                 "Get-PhysicalDisk | Select-Object MediaType | Format-List"],
                capture_output=True, text=True, timeout=10
            )
            if "SSD" in result.stdout:
                return "SSD"
            elif "NVMe" in result.stdout:
                return "NVMe"
            elif "HDD" in result.stdout:
                return "HDD"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return None
