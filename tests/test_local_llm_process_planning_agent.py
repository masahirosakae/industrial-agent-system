import pytest

from src.harness.schema import AgentInput, Source, Metadata
from src.harness.evaluation import evaluate_agent_output
from src.llm.base import LLMProvider, LLMResponse
from src.llm.fugu_provider import FuguProvider
from src.llm.ollama_provider import OllamaProvider
from src.agents.local_llm_process_planning_agent.local_llm_process_planning_agent import (
    LocalLLMProcessPlanningAgent,
)


class StaticProvider(LLMProvider):
    def __init__(self, text: str):
        self.text = text
        self.prompts = []

    def generate(self, prompt: str) -> LLMResponse:
        self.prompts.append(prompt)
        return LLMResponse(text=self.text, model="test-model", provider="test")


def create_input(file_path: str, task_id: str = "llm_test_001") -> AgentInput:
    return AgentInput(
        task_id=task_id,
        input_type="specification",
        source=Source(file_path=file_path, file_type="text"),
        metadata=Metadata(
            part_name="Test Plate",
            drawing_type="plate",
            created_at="2026-01-01",
        ),
    )


def test_local_llm_agent_detects_drilling_with_mock():
    agent_input = create_input("穴_4ヶ所")
    provider = StaticProvider(
        '{"manufacturing_processes":[{"process_type":"drilling","quantity":4}],"findings":[]}'
    )
    agent = LocalLLMProcessPlanningAgent(provider=provider)

    output = agent.run(agent_input)
    evaluation = evaluate_agent_output(output)

    assert output.status == "success"
    assert output.result is not None
    assert len(output.result.manufacturing_processes) == 1
    assert output.result.manufacturing_processes[0].process_type == "drilling"
    assert output.result.manufacturing_processes[0].quantity == 4
    assert evaluation.metrics["process_validity"] == 1.0
    assert len(provider.prompts) == 1


def test_local_llm_agent_detects_milling_with_mock():
    agent_input = create_input("フライス_面加工_外形")
    provider = StaticProvider(
        '{"manufacturing_processes":[{"process_type":"milling","quantity":1}],"findings":[]}'
    )
    agent = LocalLLMProcessPlanningAgent(provider=provider)

    output = agent.run(agent_input)
    evaluation = evaluate_agent_output(output)

    assert output.status == "success"
    assert output.result is not None
    assert len(output.result.manufacturing_processes) == 1
    assert output.result.manufacturing_processes[0].process_type == "milling"
    assert evaluation.metrics["process_validity"] == 1.0


def test_local_llm_agent_handles_unknown_input_with_mock():
    agent_input = create_input("unknown_input_no_manufacturing_feature")
    provider = StaticProvider('{"manufacturing_processes":[],"findings":[]}')
    agent = LocalLLMProcessPlanningAgent(provider=provider)

    output = agent.run(agent_input)

    assert output.status == "success"
    assert output.result is not None
    assert output.result.manufacturing_processes == []


def test_local_llm_agent_defaults_to_ollama_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    agent = LocalLLMProcessPlanningAgent()

    assert isinstance(agent.provider, OllamaProvider)


def test_local_llm_agent_accepts_fugu_provider_name(monkeypatch):
    monkeypatch.setenv("FUGU_API_KEY", "test-key")
    monkeypatch.setenv("FUGU_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUGU_MODEL", "test-fugu")

    agent = LocalLLMProcessPlanningAgent(provider_name="fugu")

    assert isinstance(agent.provider, FuguProvider)


def test_local_llm_agent_preserves_fallback_after_provider_response():
    provider = StaticProvider('{"manufacturing_processes":[],"findings":[]}')
    agent = LocalLLMProcessPlanningAgent(provider=provider)

    output = agent.run(create_input("M8_4"))

    assert output.result is not None
    assert len(output.result.manufacturing_processes) == 1
    assert output.result.manufacturing_processes[0].process_type == "drilling"


@pytest.mark.parametrize(
    ("file_path", "process_type", "checkpoint_type"),
    [
        ("タップ_M8_ねじ加工.txt", "tapping", "thread_gauge_check"),
        ("リーマ仕上げ_穴径公差.txt", "reaming", "hole_diameter_precision"),
        ("旋削_外径加工_シャフト.txt", "turning", "outer_diameter_check"),
    ],
)
def test_local_llm_agent_fallback_supports_extended_processes(
    file_path, process_type, checkpoint_type
):
    provider = StaticProvider('{"manufacturing_processes":[],"findings":[]}')
    agent = LocalLLMProcessPlanningAgent(provider=provider)

    output = agent.run(create_input(file_path))

    assert output.result is not None
    process = output.result.manufacturing_processes[0]
    assert process.process_type == process_type
    assert process.quality_checkpoints[0].checkpoint_type == checkpoint_type
    assert process.needs_review is True
    assert evaluate_agent_output(output).metrics["process_validity"] == 1.0


def test_local_llm_agent_fallback_supports_surface_grinding_after_parse_error():
    provider = StaticProvider("not valid JSON")
    agent = LocalLLMProcessPlanningAgent(provider=provider)

    output = agent.run(create_input("平面研削_表面粗さ_Ra0.8.txt"))

    assert output.result is not None
    process = output.result.manufacturing_processes[0]
    assert process.process_type == "surface_grinding"
    assert process.quality_checkpoints[0].checkpoint_type == "flatness_check"
    assert process.needs_review is True
    assert output.result.findings[0]["category"] == "fallback_after_parse_error"


def test_local_llm_agent_fallback_supports_mixed_process_input():
    provider = StaticProvider('{"manufacturing_processes":[],"findings":[]}')
    agent = LocalLLMProcessPlanningAgent(provider=provider)

    output = agent.run(create_input("穴加工_タップ_フライス_表面仕上げ.txt"))

    assert output.result is not None
    process_types = {
        process.process_type for process in output.result.manufacturing_processes
    }
    assert {"drilling", "tapping", "milling"} <= process_types


def test_local_llm_agent_parser_accepts_extended_process_type():
    provider = StaticProvider(
        '{"manufacturing_processes":[{"process_type":"turning"}],"findings":[]}'
    )
    agent = LocalLLMProcessPlanningAgent(provider=provider)

    output = agent.run(create_input("unknown_input.txt"))

    assert output.result is not None
    process = output.result.manufacturing_processes[0]
    assert process.process_type == "turning"
    assert process.quality_checkpoints[0].checkpoint_type == "outer_diameter_check"
