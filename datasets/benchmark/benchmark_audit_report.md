# Industrial Agent Benchmark Dataset Audit Report

## Dataset v1 Summary

- total_cases: 100
- difficulty_distribution:
  - easy: 20
  - medium: 40
  - hard: 20
  - adversarial: 20
- domain_distribution:
  - screw_tightening: 10
  - soldering: 10
  - press_fit: 10
  - leak_testing: 10
  - dimensional_inspection: 10
  - visual_inspection: 10
  - noise_vibration: 10
  - assembly_process: 10
  - calibration_measurement: 10
  - production_control: 10
- overall_score: 0.65
- realism_score: 0.88
- diversity_score: 0.45
- difficulty_score: 0.70
- workflow_coverage_score: 0.55
- model_comparison_score: 0.60

## Dataset v1 Key Issues

- duplicate templates
  - Several easy cases followed the same recent maintenance / setup error / correction pattern.
  - Several adversarial cases reused the same missing-traceability and contradictory-records pattern.
- weak cases
  - Some cases leaked the answer directly in `observed_facts`.
  - Some cases were too trivial to differentiate weak and strong models.
- difficulty mismatch
  - Some medium cases were closer to hard due to interacting causes and missing evidence.
  - Some adversarial cases were closer to standard measurement-method review than true adversarial reasoning.
- static countermeasure categories
  - `expected_countermeasure_categories` was effectively constant across all cases: `containment`, `verification`, and `permanent_action`.
  - This weakened the evaluation signal for CountermeasurePlanningAgent.
- insufficient workflow coverage
  - QEA escalation behavior was tested primarily through missing evidence, not through subtle false RCA acceptance.
  - DOE, MSA, PLC/software interaction, and supplier 8D reasoning were underrepresented.

## Dataset v1.1 Summary

- total_cases: 20
- difficulty_distribution:
  - easy: 4
  - medium: 7
  - hard: 6
  - adversarial: 3
- domain_distribution:
  - screw_tightening: 2
  - visual_inspection: 1
  - production_control: 3
  - leak_testing: 3
  - dimensional_inspection: 2
  - soldering: 1
  - press_fit: 1
  - noise_vibration: 1
  - assembly_process: 3
  - calibration_measurement: 3
- overall_score: 0.74
- realism_score: 0.82
- diversity_score: 0.62
- difficulty_score: 0.70
- workflow_coverage_score: 0.68
- model_comparison_score: 0.80

## Dataset v1.1 Key Issues

- Small sample size limits statistical confidence for model comparison.
- Domain coverage is uneven; several domains have only one case.
- Expected results are heavily skewed toward `needs_review`.
- Some cases still contain mild answer leakage.
- Adversarial diversity is improved but still limited to a small number of archetypes.

## Improvements from v1 to v1.1

- more diverse countermeasure categories
- stronger MSA / DOE / PLC / supplier-quality cases
- improved model comparison value
- stronger adversarial reasoning cases
- better coverage of physical impossibility, statistical confounding, measurement-resolution limits, and supplier-claim rejection
- stronger cases for QualityEvaluationAgent review escalation behavior

## Remaining Issues

- small sample size
- uneven domain coverage
- accepted / needs_review imbalance
- some answer leakage
- adversarial diversity still limited
- Phase1 selection cannot exactly match the ideal Easy / Medium / Hard ratio because v1.1 has only 20 cases and includes adversarial cases
- more field-return, long-latency, supplier traceability, cosmetic acceptance, and human-factors cases should be added in Phase2

## Conclusion

Dataset v1 is broad and balanced by domain and difficulty, but its repeated templates and static countermeasure expectations reduce benchmark discrimination. Dataset v1.1 is smaller but more valuable for Phase1 model comparison because it includes stronger measurement-system, DOE, PLC/software, supplier-quality, physical-reasoning, and adversarial cases. For Phase1 Quality Workflow evaluation, v1.1 is suitable as a fixed 20-case evaluation set, with the known limitation that results should be interpreted as directional rather than statistically comprehensive.
