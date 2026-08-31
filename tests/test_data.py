from importlib.resources import files

from benchwolf.data.model_db import MODEL_DATABASE, get_model_spec


def test_bundled_datasets_exist():
    data = files("benchwolf.data")
    assert (data / "mmlu_mini.json").is_file()
    assert (data / "humaneval_mini.json").is_file()


def test_model_database_is_populated():
    assert len(MODEL_DATABASE) >= 35
    assert get_model_spec("qwen2.5:3b") is not None
