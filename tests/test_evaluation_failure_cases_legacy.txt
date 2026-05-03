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


from src.harness.evaluation import evaluate_agent_output
from src.harness.schema import AgentOutput


def test_failure_result_none():
    agent_output = AgentOutput(
        task_id="failure_result_none",
        agent_name="mock_agent",
        status="failure",
        result=None,
        confidence=0.0,
        errors=["result is missing"],
    )

    evaluation = evaluate_agent_output(agent_output)

    print(evaluation)

    assert evaluation.passed is False
    assert evaluation.metrics["has_result"] is False


def test_failure_low_confidence():
    agent_output = AgentOutput(
        task_id="failure_low_confidence",
        agent_name="mock_agent",
        status="success",
        result=None,
        confidence=0.3,
        errors=[],
    )

    evaluation = evaluate_agent_output(agent_output)

    print(evaluation)

    assert evaluation.passed is False
    assert evaluation.metrics["confidence"] < 0.7


def test_failure_needs_review():
    agent_output = AgentOutput(
        task_id="failure_needs_review",
        agent_name="mock_agent",
        status="needs_review",
        result=None,
        confidence=0.5,
        errors=[],
    )

    evaluation = evaluate_agent_output(agent_output)

    print(evaluation)

    assert evaluation.passed is False
    assert evaluation.metrics["schema_valid"] is True


def test_failure_success_with_errors():
    agent_output = AgentOutput(
        task_id="failure_success_with_errors",
        agent_name="mock_agent",
        status="success",
        result=None,
        confidence=0.9,
        errors=["unexpected parsing error"],
    )

    evaluation = evaluate_agent_output(agent_output)

    print(evaluation)

    assert evaluation.passed is False
    assert evaluation.metrics["error_count"] > 0




if __name__ == "__main__":
  test_evaluation_detects_invalid_planning_output()
  test_failure_result_none()
  test_failure_low_confidence()
  test_failure_needs_review()
  test_failure_success_with_errors()