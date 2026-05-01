from src.harness.evaluation import evaluate_agent_output
from src.harness.schema import (
    AgentOutput,
    PlanningResult,
    ManufacturingProcess,
)


def test_evaluation_detects_invalid_planning_output():
    processes = [
        ManufacturingProcess(
            process_name="",
            description="Process is unclear",
            target_feature="unknown",
            quantity="unknown",
            basis=""
        ),
        ManufacturingProcess(
            process_name="machining",
            description="Machine feature",
            target_feature="hole",
            quantity=0,
            basis=""
        ),
    ]

    planning_result = PlanningResult(
        manufacturing_processes=processes,
        quality_checkpoints=[],
    )

    agent_output = AgentOutput(
        task_id="failure_001",
        agent_name="mock_agent",
        status="success",
        result=planning_result,
        confidence=0.9,
        errors=[],
    )

    evaluation = evaluate_agent_output(agent_output)

    print(evaluation)

    assert evaluation.metrics["basis_validity"] < 1.0
    assert evaluation.metrics["quantity_validity"] < 1.0
    assert evaluation.metrics["quality_checkpoint_consistency"] < 1.0
    assert evaluation.metrics["process_validity"] < 1.0
    assert len(evaluation.findings) > 0

if __name__ == "__main__":
  test_evaluation_detects_invalid_planning_output()