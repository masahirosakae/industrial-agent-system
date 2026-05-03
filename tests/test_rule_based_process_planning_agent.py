from src.harness.schema import AgentInput, Source, Metadata
from src.harness.evaluation import evaluate_agent_output
from src.agents.process_planning_agent.process_planning_agent import (
    RuleBasedProcessPlanningAgent,
)


def create_input(file_path: str, part_name: str = "Test Part") -> AgentInput:
    return AgentInput(
        task_id="rule_based_test_001",
        input_type="specification",
        source=Source(
            file_path=file_path,
            file_type="text",
        ),
        metadata=Metadata(
            part_name=part_name,
            drawing_type="test",
            created_at="2026-01-01",
        ),
    )


def test_rule_based_agent_detects_drilling():
    agent_input = create_input("dummy_spec_穴_4ヶ所.txt")

    agent = RuleBasedProcessPlanningAgent()
    output = agent.run(agent_input)
    evaluation = evaluate_agent_output(output)

    assert output.status == "success"
    assert output.result is not None
    assert output.result.manufacturing_processes[0].process_type == "drilling"
    assert output.result.manufacturing_processes[0].quantity == 4
    assert evaluation.passed is True
    assert evaluation.metrics["overall"] == 1.0


def test_rule_based_agent_detects_milling():
    agent_input = create_input("dummy_spec_フライス_面加工.txt")

    agent = RuleBasedProcessPlanningAgent()
    output = agent.run(agent_input)
    evaluation = evaluate_agent_output(output)

    assert output.status == "success"
    assert output.result is not None
    assert output.result.manufacturing_processes[0].process_type == "milling"
    assert evaluation.metrics["process_validity"] == 1.0


def test_rule_based_agent_returns_needs_review_for_unknown_input():
    agent_input = create_input("dummy_spec_unknown.txt")

    agent = RuleBasedProcessPlanningAgent()
    output = agent.run(agent_input)

    assert output.status == "needs_review"
    assert output.result is not None
    assert output.result.manufacturing_processes == []
    assert len(output.result.findings) >= 1
