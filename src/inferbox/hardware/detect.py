"""Hardware detection for InferBox."""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from typing import Optional

import psutil

from inferbox.models import HardwareProfile


def detect_hardware() -> HardwareProfile:
    """Detect and return comprehensive hardware information."""
    cpu_name = _get_cpu_name()
    cpu_arch = platform.machine()
    ram_total_gb = psutil.virtual_memory().total / (1024**3)
    gpu_name = _detect_gpu()
    npu_name = _detect_npu(cpu_name, gpu_name)
    bandwidth_gb_s = _estimate_memory_bandwidth_gb_s(cpu_name, gpu_name, cpu_arch, ram_total_gb)

    return HardwareProfile(
        cpu_name=cpu_name,
        cpu_arch=cpu_arch,
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        cpu_freq_mhz=_get_cpu_freq(),
        ram_total_gb=ram_total_gb,
        ram_available_gb=psutil.virtual_memory().available / (1024**3),
        os_name=platform.system(),
        os_version=platform.release(),
        python_version=platform.python_version(),
        gpu_name=gpu_name,
        npu_name=npu_name,
        storage_type=_detect_storage(),
        memory_bandwidth_gb_s=bandwidth_gb_s,
    )


def _get_cpu_name() -> str:
    """Get CPU model name."""
    system = platform.system()

    if system == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.strip().startswith("model name"):
                        return line.split(":", 1)[1].strip()
                    # ARM devices use "Hardware" or "Model"
                    if line.strip().startswith("Hardware") or line.strip().startswith("Model"):
                        return line.split(":", 1)[1].strip()
        except (FileNotFoundError, PermissionError):
            pass

    elif system == "Windows":
        # 1. Try PowerShell Get-CimInstance (modern Windows 10/11)
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")[0].strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # 2. Fallback to WMIC if available
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
            if result.returncode == 0 and result.stdout.strip():
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
    # 1. Try nvidia-smi (NVIDIA GPUs)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0].strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # 2. Try ROCm for AMD dGPUs
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

    # 3. Try Apple Silicon GPU (macOS)
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                if "Chipset Model:" in line:
                    return line.split(":", 1)[1].strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    # 4. Windows DirectX / WMI Display adapter check (e.g. Intel Arc, AMD Radeon 890M)
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ','"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpus = [g.strip() for g in result.stdout.strip().split(",") if g.strip()]
                # Filter out generic/virtual adapters
                for g in gpus:
                    if any(vendor in g.lower() for vendor in ["nvidia", "amd", "radeon", "intel", "geforce", "rtx", "arc"]):
                        return g
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    return None


def _detect_npu(cpu_name: str, gpu_name: Optional[str]) -> Optional[str]:
    """Detect dedicated NPU / AI accelerator."""
    cpu_lower = cpu_name.lower()
    system = platform.system()

    # Apple Neural Engine
    if system == "Darwin" or "apple m" in cpu_lower:
        return "Apple Neural Engine (16-core)"

    # Qualcomm Snapdragon X (Hexagon NPU 45 TOPS)
    if "snapdragon" in cpu_lower or "x elite" in cpu_lower or "x plus" in cpu_lower:
        return "Qualcomm Hexagon NPU (45 TOPS)"

    # AMD Ryzen AI (XDNA / XDNA 2)
    if any(k in cpu_lower for k in ["ryzen ai", "ai 9", "ai 7", "ai max", "strix", "8945", "7940"]):
        if "max" in cpu_lower or "395" in cpu_lower or "390" in cpu_lower:
            return "AMD XDNA 2 NPU (50+ TOPS)"
        return "AMD XDNA 2 NPU (50-55 TOPS)"

    # Intel AI Boost / NPU (Lunar Lake, Meteor Lake, Arrow Lake)
    if "ultra" in cpu_lower:
        if "2" in cpu_lower and ("v" in cpu_lower or "2" in cpu_lower):
            return "Intel NPU 4 (48 TOPS - Lunar Lake)"
        return "Intel AI Boost NPU (~13-48 TOPS)"

    # Linux check for Rockchip RKNN NPU
    if system == "Linux":
        try:
            from pathlib import Path
            if Path("/sys/class/misc/npu").exists():
                return "Rockchip RKNN NPU"
        except Exception:
            pass

    return None


def _estimate_memory_bandwidth_gb_s(
    cpu_name: str,
    gpu_name: Optional[str],
    cpu_arch: str,
    ram_total_gb: float,
) -> Optional[float]:
    """Estimate peak memory bandwidth in GB/s."""
    name_str = f"{cpu_name} {gpu_name or ''}".lower()

    # Apple Silicon Unified Memory
    if "m4 ultra" in name_str:
        return 800.0
    elif "m4 max" in name_str:
        return 546.0
    elif "m4 pro" in name_str:
        return 273.0
    elif "m4" in name_str:
        return 120.0
    elif "m3 max" in name_str:
        return 400.0
    elif "m3 pro" in name_str:
        return 150.0
    elif "m3" in name_str:
        return 100.0
    elif "m2 ultra" in name_str:
        return 800.0
    elif "m2 max" in name_str:
        return 400.0
    elif "m2 pro" in name_str:
        return 200.0
    elif "m2" in name_str:
        return 100.0
    elif "m1 ultra" in name_str:
        return 800.0
    elif "m1 max" in name_str:
        return 400.0
    elif "m1 pro" in name_str:
        return 200.0
    elif "m1" in name_str:
        return 68.0

    # AMD Strix Halo (Ryzen AI Max 300) - 256-bit LPDDR5X
    if "ai max" in name_str or "ryzen ai max" in name_str:
        return 270.0

    # AMD Strix Point (Ryzen AI 300) - 128-bit LPDDR5X
    if "ryzen ai 9" in name_str or "ryzen ai 7" in name_str:
        return 136.0

    # Intel Lunar Lake (Core Ultra 200V) - 128-bit on-package LPDDR5X-8533
    if "ultra" in name_str and ("2" in name_str and "v" in name_str):
        return 136.0

    # Qualcomm Snapdragon X Elite
    if "snapdragon" in name_str or "x elite" in name_str or "x plus" in name_str:
        return 135.0

    # Discrete NVIDIA GPU bandwidth
    if "4090" in name_str:
        return 1008.0
    elif "4080" in name_str:
        return 716.0
    elif "4070" in name_str:
        return 504.0
    elif "3090" in name_str:
        return 936.0
    elif "3080" in name_str:
        return 760.0
    elif "3060" in name_str:
        return 360.0

    # Standard DDR5 / DDR4 dual-channel desktop/laptop estimation
    if ram_total_gb >= 16.0:
        return 64.0  # Typical DDR5 Dual Channel (approx ~64 GB/s)
    elif ram_total_gb >= 8.0:
        return 38.4  # Typical DDR4 Dual Channel (approx ~38 GB/s)

    return 25.0


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
                ["powershell", "-NoProfile", "-Command",
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
