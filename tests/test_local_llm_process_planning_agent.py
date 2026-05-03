from src.harness.schema import AgentInput, Source, Metadata
from src.harness.evaluation import evaluate_agent_output
from src.agents.local_llm_process_planning_agent.local_llm_process_planning_agent import (
    LocalLLMProcessPlanningAgent,
)


def create_input(file_path: str, task_id: str = "llm_test_001") -> AgentInput:
    return AgentInput(
        task_id=task_id,
        input_type="specification",
        source=Source(
            file_path=file_path,
            file_type="text",
        ),
        metadata=Metadata(
            part_name="Test Plate",
            drawing_type="plate",
            created_at="2026-01-01",
        ),
    )


def test_local_llm_agent_detects_drilling():
    agent_input = create_input("穴_4ヶ所")
    agent = LocalLLMProcessPlanningAgent()

    output = agent.run(agent_input)
    evaluation = evaluate_agent_output(output)

    print("\n=== Drilling Output ===")
    print(output)
    print("\n=== Drilling Evaluation ===")
    print(evaluation)

    assert output.status == "success"
    assert output.result is not None
    assert len(output.result.manufacturing_processes) >= 1
    assert output.result.manufacturing_processes[0].process_type == "drilling"
    assert evaluation.metrics["process_validity"] == 1.0


def test_local_llm_agent_detects_milling():
    agent_input = create_input("フライス_面加工_外形")
    agent = LocalLLMProcessPlanningAgent()

    output = agent.run(agent_input)
    evaluation = evaluate_agent_output(output)

    print("\n=== Milling Output ===")
    print(output)
    print("\n=== Milling Evaluation ===")
    print(evaluation)

    assert output.status == "success"
    assert output.result is not None
    assert len(output.result.manufacturing_processes) >= 1
    assert evaluation.metrics["process_validity"] == 1.0


def test_local_llm_agent_handles_unknown_input():
    agent_input = create_input("unknown_input_no_manufacturing_feature")
    agent = LocalLLMProcessPlanningAgent()

    output = agent.run(agent_input)

    print("\n=== Unknown Output ===")
    print(output)

    assert output is not None
    assert output.status in ["success", "needs_review", "failure"]