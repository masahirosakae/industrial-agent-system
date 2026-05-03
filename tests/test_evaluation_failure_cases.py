from src.harness.evaluation import evaluate_agent_output
from src.harness.schema import (
    AgentOutput,
    PlanningResult,
    ManufacturingProcess,
    QualityCheckpoint,
)


def make_valid_agent_output() -> AgentOutput:
    return AgentOutput(
        task_id="valid_001",
        agent_name="mock_agent",
        status="success",
        result=PlanningResult(
            manufacturing_processes=[
                ManufacturingProcess(
                    process_id="P001",
                    process_type="drilling",
                    process_name="drilling",
                    description="Drill holes based on drawing instruction.",
                    target_feature="4x M8 holes",
                    quantity=4,
                    basis=["4x M8"],
                    quality_checkpoints=[
                        QualityCheckpoint(
                            checkpoint_type="hole_diameter",
                            description="Check hole diameter.",
                            inspection_method="caliper",
                            basis=["M8"],
                        )
                    ],
                    confidence=0.9,
                    needs_review=False,
                )
            ],
            findings=[],
        ),
        confidence=0.9,
        errors=[],
        notes=[],
    )


def test_result_none_should_fail():
    output = make_valid_agent_output()
    output.status = "failure"
    output.result = None
    output.confidence = 0.0
    output.errors = ["result is missing"]

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["has_result"] is False
    assert evaluation.metrics["error_count"] > 0


def test_no_processes_should_fail():
    output = make_valid_agent_output()
    assert output.result is not None

    output.result.manufacturing_processes = []

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["has_processes"] is False
    assert evaluation.metrics["process_validity"] == 0.0


def test_missing_basis_should_need_review():
    output = make_valid_agent_output()
    assert output.result is not None

    output.result.manufacturing_processes[0].basis = []

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["basis_validity"] < 1.0
    assert evaluation.metrics["warning_count"] > 0


def test_missing_quantity_should_need_review():
    output = make_valid_agent_output()
    assert output.result is not None

    output.result.manufacturing_processes[0].quantity = None

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["quantity_validity"] < 1.0
    assert evaluation.metrics["warning_count"] > 0


def test_invalid_quantity_zero_should_need_review():
    output = make_valid_agent_output()
    assert output.result is not None

    output.result.manufacturing_processes[0].quantity = 0

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["quantity_validity"] < 1.0
    assert evaluation.metrics["warning_count"] > 0


def test_invalid_quantity_negative_should_need_review():
    output = make_valid_agent_output()
    assert output.result is not None

    output.result.manufacturing_processes[0].quantity = -1

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["quantity_validity"] < 1.0
    assert evaluation.metrics["warning_count"] > 0


def test_invalid_process_type_should_need_review():
    output = make_valid_agent_output()
    assert output.result is not None

    output.result.manufacturing_processes[0].process_type = "invalid_type"

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["process_validity"] < 1.0
    assert evaluation.metrics["warning_count"] > 0


def test_missing_quality_checkpoint_should_need_review():
    output = make_valid_agent_output()
    assert output.result is not None

    output.result.manufacturing_processes[0].quality_checkpoints = []

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["quality_checkpoint_consistency"] < 1.0
    assert evaluation.metrics["warning_count"] > 0


def test_low_confidence_should_need_review():
    output = make_valid_agent_output()
    output.confidence = 0.3

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["confidence"] < 0.7


def test_success_status_with_errors_should_fail():
    output = make_valid_agent_output()
    output.errors = ["unexpected parsing error"]

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["error_count"] > 0


def test_unknown_process_type_should_need_review():
    output = make_valid_agent_output()
    assert output.result is not None

    output.result.manufacturing_processes[0].process_type = "unknown"

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["process_validity"] == 1.0
    assert evaluation.metrics["warning_count"] > 0


def test_checkpoint_mismatch_should_need_review():
    output = make_valid_agent_output()
    assert output.result is not None

    process = output.result.manufacturing_processes[0]
    process.process_type = "drilling"
    process.quality_checkpoints[0].checkpoint_type = "thread_check"
    process.quality_checkpoints[0].description = "Check thread quality."
    process.quality_checkpoints[0].inspection_method = "thread gauge"
    process.quality_checkpoints[0].basis = ["M8 thread"]

    evaluation = evaluate_agent_output(output)

    assert evaluation.passed is False
    assert evaluation.metrics["quality_checkpoint_consistency"] < 1.0
    assert evaluation.metrics["warning_count"] > 0


def run_test(test_func):
    try:
        test_func()
        print(f"[PASS] {test_func.__name__}")
    except AssertionError:
        print(f"[FAIL] {test_func.__name__}")
    except Exception as e:
        print(f"[ERROR] {test_func.__name__}: {e}")


if __name__ == "__main__":
    tests = [
        test_missing_basis_should_need_review,
        test_missing_quantity_should_need_review,
        test_invalid_process_type_should_need_review,
        test_missing_quality_checkpoint_should_need_review,
        test_result_none_should_fail,
        test_no_processes_should_fail,
        test_invalid_quantity_zero_should_need_review,
        test_invalid_quantity_negative_should_need_review,
        test_low_confidence_should_need_review,
        test_success_status_with_errors_should_fail,
        test_unknown_process_type_should_need_review,
        test_checkpoint_mismatch_should_need_review,
    ]

    for test in tests:
        run_test(test)
