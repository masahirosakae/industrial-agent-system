from src.harness.schema import AgentInput, Source, Metadata
from src.harness.evaluation import evaluate_agent_output
from src.agents.local_llm_process_planning_agent.local_llm_process_planning_agent import (
    LocalLLMProcessPlanningAgent,
)


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


def test_local_llm_agent_detects_drilling_with_mock(monkeypatch):
    agent_input = create_input("穴_4ヶ所")
    agent = LocalLLMProcessPlanningAgent()

    monkeypatch.setattr(
        agent,
        "_call_ollama",
        lambda prompt: '{"manufacturing_processes":[{"process_type":"drilling","quantity":4}],"findings":[]}',
    )

    output = agent.run(agent_input)
    evaluation = evaluate_agent_output(output)

    assert output.status == "success"
    assert output.result is not None
    assert len(output.result.manufacturing_processes) == 1
    assert output.result.manufacturing_processes[0].process_type == "drilling"
    assert output.result.manufacturing_processes[0].quantity == 4
    assert evaluation.metrics["process_validity"] == 1.0


def test_local_llm_agent_detects_milling_with_mock(monkeypatch):
    agent_input = create_input("フライス_面加工_外形")
    agent = LocalLLMProcessPlanningAgent()

    monkeypatch.setattr(
        agent,
        "_call_ollama",
        lambda prompt: '{"manufacturing_processes":[{"process_type":"milling","quantity":1}],"findings":[]}',
    )

    output = agent.run(agent_input)
    evaluation = evaluate_agent_output(output)

    assert output.status == "success"
    assert output.result is not None
    assert len(output.result.manufacturing_processes) == 1
    assert output.result.manufacturing_processes[0].process_type == "milling"
    assert evaluation.metrics["process_validity"] == 1.0


def test_local_llm_agent_handles_unknown_input_with_mock(monkeypatch):
    agent_input = create_input("unknown_input_no_manufacturing_feature")
    agent = LocalLLMProcessPlanningAgent()

    monkeypatch.setattr(
        agent,
        "_call_ollama",
        lambda prompt: '{"manufacturing_processes":[],"findings":[]}',
    )

    output = agent.run(agent_input)

    assert output.status == "success"
    assert output.result is not None
    assert output.result.manufacturing_processes == []
