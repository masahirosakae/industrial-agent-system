# Phase1 Evaluation Dataset v1 Selection Report

## Purpose

Phase1 Quality Workflow の評価実験に使用する固定評価セット。

## Source Dataset

- Source: Industrial Agent Benchmark Dataset v1.1
- Total selected cases: 20

## Selection Policy

- Fugu / GPT-5.5 / Claude 比較に使う
- QIA / RCA / CMP / QEA の全段を評価する
- manufacturing reasoning を評価する
- evidence awareness を評価する
- needs_review 判断を評価する
- v1.1 は20ケースのみのため、Phase1 Evaluation Dataset v1 は v1.1 全ケースをそのまま使用する

## Selected Dataset Summary

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
- expected_workflow_result_distribution:
  - accepted: 1
  - needs_review: 19

## Selected Cases

| case_id | difficulty | domain | expected_workflow_result | reason |
|---|---|---|---|---|
| QW-V2-001 | easy | screw_tightening | needs_review | Baseline fastening setup case with containment-scope and safety-critical review risk. |
| QW-V2-002 | easy | visual_inspection | accepted | Baseline visual defect case with a clear mechanical damage mechanism. |
| QW-V2-003 | easy | production_control | needs_review | Material-control and mixed-WIP containment case. |
| QW-V2-004 | easy | leak_testing | needs_review | False-reject leak-test case requiring retest-capacity review. |
| QW-V2-005 | medium | dimensional_inspection | needs_review | SPC trend and reaction-plan failure with quantified Cpk degradation. |
| QW-V2-006 | medium | soldering | needs_review | Supplier material boundary condition and IPC Class 3 soldering risk. |
| QW-V2-007 | medium | press_fit | needs_review | Thermal expansion and lab-versus-floor measurement reasoning. |
| QW-V2-008 | medium | noise_vibration | needs_review | System-level NVH reasoning involving resonance and supplier material change. |
| QW-V2-009 | medium | assembly_process | needs_review | Facility/process interaction affecting conformal coating cure. |
| QW-V2-010 | medium | calibration_measurement | needs_review | MSA case where gauge variation dominates apparent process capability. |
| QW-V2-011 | hard | assembly_process | needs_review | PLC timing, mechanical bounce, and unsafe vision fail-open logic. |
| QW-V2-012 | hard | leak_testing | needs_review | DOE interaction effect and supplier material qualification gap. |
| QW-V2-013 | hard | calibration_measurement | needs_review | ISO 17025 traceability-chain break with aerospace safety implications. |
| QW-V2-014 | hard | dimensional_inspection | needs_review | Additive manufacturing residual-stress and equipment gas-flow case. |
| QW-V2-015 | hard | production_control | needs_review | Supplier 8D rejection case requiring physical process reasoning. |
| QW-V2-016 | adversarial | calibration_measurement | needs_review | Measurement-resolution fallacy and over-adjustment risk. |
| QW-V2-017 | adversarial | leak_testing | needs_review | Spurious-correlation trap requiring causal-mechanism reasoning. |
| QW-V2-018 | adversarial | production_control | needs_review | Metallurgical impossibility in supplier deviation justification. |
| QW-V2-019 | hard | assembly_process | needs_review | Robotics timing and fluid-lag interaction in dispensing. |
| QW-V2-020 | medium | screw_tightening | needs_review | Tightening sequence versus clamp-load/gasket behavior. |

## Known Limitations

- Easy / Medium / Hard の理想比率とは完全一致していない
- needs_review が多い
- accepted case が少ない
- domain coverage に偏りがある
- Phase2で adversarial cases を拡張予定

## Usage

```bash
python scripts/run_quality_workflow.py \
  --provider fugu \
  --trace datasets/results/fugu/quality_workflow_runs.jsonl
```

## Conclusion

Phase1 Evaluation Dataset v1 は、Industrial Agent Benchmark Dataset v1.1 の20ケースを固定評価セットとしてそのまま採用する。小規模かつ `needs_review` 偏重ではあるが、QIA / RCA / CMP / QEA の基本性能、manufacturing reasoning、evidence awareness、review escalation を比較するための初期評価セットとして利用可能である。
