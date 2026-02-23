"""Tests for hardware detection module."""

import platform

from tinybench.hardware.detect import detect_hardware
from tinybench.models import HardwareProfile


def test_detect_hardware_returns_profile():
    """detect_hardware should return a HardwareProfile."""
    hw = detect_hardware()
    assert isinstance(hw, HardwareProfile)


def test_cpu_name_not_empty():
    """CPU name should never be empty."""
    hw = detect_hardware()
    assert hw.cpu_name != ""
    assert hw.cpu_name != "Unknown" or platform.system() not in ("Windows", "Linux", "Darwin")


def test_cpu_cores_positive():
    """Core counts must be positive integers."""
    hw = detect_hardware()
    assert hw.cpu_cores_physical > 0
    assert hw.cpu_cores_logical > 0
    assert hw.cpu_cores_logical >= hw.cpu_cores_physical


def test_ram_detected():
    """RAM values should be positive."""
    hw = detect_hardware()
    assert hw.ram_total_gb > 0
    assert hw.ram_available_gb > 0
    assert hw.ram_available_gb <= hw.ram_total_gb


def test_os_info():
    """OS name should be detected."""
    hw = detect_hardware()
    assert hw.os_name in ("Windows", "Linux", "Darwin")


def test_fingerprint_consistent():
    """Same hardware should produce the same fingerprint."""
    hw1 = detect_hardware()
    hw2 = detect_hardware()
    assert hw1.fingerprint == hw2.fingerprint
    assert len(hw1.fingerprint) == 12


def test_summary_format():
    """Summary should be a pipe-separated string."""
    hw = detect_hardware()
    assert "|" in hw.summary
    assert "RAM" in hw.summary
