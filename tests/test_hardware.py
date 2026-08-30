"""Tests for hardware detection module."""

import platform
import pytest

from inferbox.hardware.detect import detect_hardware
from inferbox.models import HardwareProfile


@pytest.fixture(scope="module")
def hw_profile():
    """Cache hardware profile across tests in module for fast execution."""
    return detect_hardware()


def test_detect_hardware_returns_profile(hw_profile):
    """detect_hardware should return a HardwareProfile."""
    assert isinstance(hw_profile, HardwareProfile)


def test_cpu_name_not_empty(hw_profile):
    """CPU name should never be empty."""
    assert hw_profile.cpu_name != ""
    assert hw_profile.cpu_name != "Unknown" or platform.system() not in ("Windows", "Linux", "Darwin")


def test_cpu_cores_positive(hw_profile):
    """Core counts must be positive integers."""
    assert hw_profile.cpu_cores_physical > 0
    assert hw_profile.cpu_cores_logical > 0
    assert hw_profile.cpu_cores_logical >= hw_profile.cpu_cores_physical


def test_ram_detected(hw_profile):
    """RAM values should be positive."""
    assert hw_profile.ram_total_gb > 0
    assert hw_profile.ram_available_gb > 0
    assert hw_profile.ram_available_gb <= hw_profile.ram_total_gb


def test_os_info(hw_profile):
    """OS name should be detected."""
    assert hw_profile.os_name in ("Windows", "Linux", "Darwin")


def test_fingerprint_consistent(hw_profile):
    """Same hardware should produce the same fingerprint."""
    hw2 = detect_hardware()
    assert hw_profile.fingerprint == hw2.fingerprint
    assert len(hw_profile.fingerprint) == 12


def test_summary_format(hw_profile):
    """Summary should be a pipe-separated string."""
    assert "|" in hw_profile.summary
    assert "RAM" in hw_profile.summary
