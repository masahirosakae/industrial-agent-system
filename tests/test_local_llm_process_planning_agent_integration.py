import pytest

from src.harness.schema import AgentInput, Source, Metadata
from src.agents.local_llm_process_planning_agent.local_llm_process_planning_agent import (
    LocalLLMProcessPlanningAgent,
)


def create_input(file_path: str, task_id: str = "llm_integration_001") -> AgentInput:
    return AgentInput(
        task_id=task_id,
        input_type="specification",
        source=Source(file_path=file_path, file_type="text"),
        metadata=Metadata(part_name="Test Plate", drawing_type="plate", created_at="2026-01-01"),
    )


def _require_local_llm(agent: LocalLLMProcessPlanningAgent) -> None:
    try:
        agent._call_ollama("ping")
    except Exception as exc:  # pragma: no cover - environment-dependent gate
        pytest.skip(f"Local LLM endpoint unavailable: {exc}")


@pytest.mark.integration
def test_local_llm_agent_live_call_smoke():
    agent = LocalLLMProcessPlanningAgent()
    _require_local_llm(agent)

    output = agent.run(create_input("穴_4ヶ所"))

    assert output.status in ["success", "needs_review", "failure"]
