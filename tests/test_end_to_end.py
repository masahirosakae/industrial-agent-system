from src.harness.schema import (
    AgentInput,
    Source,
    Metadata,
)
from src.harness.evaluation import evaluate_agent_output
from src.agents.process_planning_agent.process_planning_agent import RuleBasedProcessPlanningAgent


def create_sample_input() -> AgentInput:
    return AgentInput(
        task_id="e2e_001",
        input_type="specification",
        source=Source(
            file_path="dummy_spec_穴_4ヶ所.txt",
            file_type="text",
        ),
        metadata=Metadata(
            part_name="Test Plate",
            drawing_type="plate",
            created_at="2026-01-01",
        ),
    )


def test_end_to_end_basic():
    agent_input = create_sample_input()

    agent = RuleBasedProcessPlanningAgent()
    agent_output = agent.run(agent_input)

    evaluation = evaluate_agent_output(agent_output)

    print("\n=== Agent Output ===")
    print(agent_output)

    print("\n=== Evaluation ===")
    print(evaluation)

    assert agent_output is not None
    assert evaluation is not None
    assert "overall" in evaluation.metrics


if __name__ == "__main__":
    test_end_to_end_basic()