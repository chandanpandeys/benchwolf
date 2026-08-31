from benchwolf.models import BenchmarkResult, HardwareProfile


def test_result_roundtrip():
    result = BenchmarkResult(model_name="demo", backend="fake", hardware=HardwareProfile())
    restored = BenchmarkResult.from_json(result.to_json())
    assert restored.model_name == "demo"
    assert restored.benchwolf_version == result.benchwolf_version


def test_hardware_fingerprint_is_stable():
    hardware = HardwareProfile(
        cpu_name="CPU",
        cpu_arch="x86_64",
        cpu_cores_physical=4,
        ram_total_gb=16,
    )
    assert hardware.fingerprint == hardware.fingerprint
    assert len(hardware.fingerprint) == 12
