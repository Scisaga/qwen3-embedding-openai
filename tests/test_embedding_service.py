import pytest

import embedding_service


def test_prepare_backend_payload_for_single_document():
    payload = embedding_service.prepare_backend_payload(
        {
            "input": "The capital of China is Beijing.",
            "dimensions": 512,
        }
    )

    assert payload["input"] == "The capital of China is Beijing."
    assert payload["dimensions"] == 512
    assert payload["model"] == embedding_service.get_current_model_id()


def test_prepare_backend_payload_for_query_uses_default_instruction(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "default_query_instruction", "default task")

    payload = embedding_service.prepare_backend_payload(
        {
            "input": ["What is the capital of China?"],
            "input_type": "query",
        }
    )

    assert payload["input"] == ["Instruct: default task\nQuery:What is the capital of China?"]


def test_prepare_backend_payload_for_query_uses_custom_instruction():
    payload = embedding_service.prepare_backend_payload(
        {
            "input": "Explain gravity",
            "input_type": "query",
            "instruction": "Answer with retrieval-friendly vectors",
        }
    )

    assert payload["input"] == "Instruct: Answer with retrieval-friendly vectors\nQuery:Explain gravity"


def test_prepare_backend_payload_rejects_invalid_dimensions():
    with pytest.raises(embedding_service.InputValidationError):
        embedding_service.prepare_backend_payload({"input": "hello", "dimensions": 31})


def test_prepare_backend_payload_rejects_invalid_input_type():
    with pytest.raises(embedding_service.InputValidationError):
        embedding_service.prepare_backend_payload({"input": "hello", "input_type": "mixed"})


def test_build_backend_env_strips_vllm_port(monkeypatch):
    monkeypatch.setenv("VLLM_PORT", "8001")

    backend_env = embedding_service._build_backend_env()

    assert "VLLM_PORT" not in backend_env


def test_validate_backend_settings_rejects_batched_tokens_smaller_than_max_model_len(monkeypatch):
    monkeypatch.setattr(embedding_service._settings, "max_model_len", 4096)
    monkeypatch.setattr(
        embedding_service._settings,
        "extra_args",
        "--enforce-eager --max-num-batched-tokens 1024",
    )

    with pytest.raises(embedding_service.BackendUnavailableError):
        embedding_service._validate_backend_settings()
