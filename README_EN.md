# Industrial Agent System

A provider-agnostic, evaluation-first multi-agent system for industrial AI and manufacturing workflows.

![status](https://img.shields.io/badge/status-PoC-orange)
![phase](https://img.shields.io/badge/Phase1%20Quality%20Workflow-Completed-brightgreen)
![tests](https://img.shields.io/badge/tests-248%20passed-brightgreen)

> Japanese README: [README.md](./README.md)

---

## 1. Project Overview

**Industrial Agent System** is a research-oriented project for building reliable AI-assisted workflows in manufacturing.

The project focuses on:

- **Industrial AI** for manufacturing quality and process workflows
- **Manufacturing Workflow** decomposition into specialized agents
- **Multi-Agent System** architecture with explicit hand-offs between agents
- **Evaluation First** design rather than direct trust in LLM output
- **Human Review Gate** for uncertain, malformed, or risky outputs
- **Workflow Orchestration** that safely stops when any step requires review

The core idea is not to build a single autonomous model, but to build an industrial workflow system where every probabilistic LLM output is constrained, validated, evaluated, and gated before it can influence the next step.

---

## 2. Current Status

```text
Phase1 Quality Workflow Completed
```

Implemented and validated:

- Provider Abstraction
- Sakana Fugu Provider
- Ollama Provider
- Strict whole-response JSON parsing
- Schema validation
- Policy validation
- `needs_review` fallback
- Human review gate
- Quality workflow orchestration
- Integration tests

Current pytest result:

```text
248 passed
1 deselected
1 warning
```

---

## 3. Implemented Agents

### QualityIssueAnalysisAgent

Analyzes a manufacturing quality issue case and produces structured quality-issue analysis:

- issue summary
- suspected causes
- containment actions
- investigation plan
- additional data needed
- unresolved risk
- verification points

### RootCauseAnalysisAgent

Consumes `QualityIssueAnalysisAgent` output and performs structured root-cause analysis:

- 5Why analysis
- FTA-style decomposition
- fact / assumption / hypothesis separation
- missing evidence extraction
- confidence assignment per cause hypothesis

### CountermeasurePlanningAgent

Consumes `RootCauseAnalysisAgent` output and plans quality countermeasures:

- containment actions
- permanent actions
- verification plan
- risk assessment
- priority recommendation
- review reasons for provisional or evidence-limited plans

It uses RCA evidence signals such as:

- `hypotheses[].confidence`
- `five_why[].evidence_status`
- top-level and hypothesis-level `missing_evidence`

### QualityEvaluationAgent

Consumes `RootCauseAnalysisAgent` and `CountermeasurePlanningAgent` outputs and acts as the final quality gate:

- evidence sufficiency evaluation
- hallucination-risk detection
- RCA/CMP consistency check
- countermeasure quality assessment
- verification quality assessment
- approval readiness scoring
- final judgement: `accepted` or `needs_review`

---

## 4. Architecture Principles

### Evaluation First

LLM output is never treated as ground truth.

```text
LLM Output
↓
JSON Validation
↓
Schema Validation
↓
Policy Validation
↓
Human Review
```

Each agent treats LLM generation as an input candidate that must pass deterministic validation before it can become a trusted result.

### Provider Abstraction

Agents depend on the abstract `LLMProvider` interface, not on concrete model SDKs.

Supported providers:

- **Sakana Fugu** via `FuguProvider`
- **Ollama** via `OllamaProvider`

This allows the same workflow to be executed with local models or remote API providers without changing agent logic.

### Strict Structured Outputs

All Phase1 quality agents use strict structured output policy:

- whole-response JSON parse using `json.loads(text)`
- no Markdown
- no code fences
- no prefix/suffix prose
- no substring extraction from first `{` to last `}`
- schema validation before dataclass conversion
- policy validation after schema validation where needed
- `needs_review` fallback instead of uncaught failure

### Human Review Gate

Every agent returns an explicit status. Workflow progression is gated by that status.

```text
status == success
↓
next step

status != success
↓
stop workflow
↓
human review
```

`needs_review` outputs are never passed to downstream agents.

---

## 5. Phase1 Quality Workflow

The Phase1 Quality Workflow is now implemented as an orchestrated sequence:

```text
QualityIssueAnalysisAgent  (QIA)
↓
RootCauseAnalysisAgent     (RCA)
↓
CountermeasurePlanningAgent (CMP)
↓
QualityEvaluationAgent     (QEA)
```

Short form:

```text
QIA
↓
RCA
↓
CMP
↓
QEA
```

Workflow module:

```text
src/workflows/quality_workflow.py
```

Main orchestration API:

```python
run_quality_workflow(...)
```

CLI runner:

```text
scripts/run_quality_workflow.py
```

---

## 6. Workflow Result States

### `success`

All four agents completed successfully, and `QualityEvaluationAgent` returned:

```text
overall_judgement == accepted
```

The workflow result is:

```text
status = success
final_judgement = accepted
stopped_at = None
```

### `needs_review`

Any agent returned `needs_review`, or the final evaluation did not accept the result.

The workflow stops immediately at the first non-success step:

```text
status = needs_review
final_judgement = needs_review
stopped_at = <agent_name>
```

Unexpected orchestration exceptions are also converted into a safe-side `needs_review` workflow result with exception details in `review_reasons`.

---

## 7. Running Examples

### Run Phase1 Quality Workflow with Sakana Fugu

Set Fugu environment variables first:

```bash
export FUGU_API_KEY="<your-key>"
export FUGU_BASE_URL="https://<your-fugu-endpoint>/v1"
export FUGU_MODEL="<your-fugu-model>"
```

Then run:

```bash
python scripts/run_quality_workflow.py --provider fugu
```

Save the full JSON workflow result:

```bash
python scripts/run_quality_workflow.py \
  --provider fugu \
  --output outputs/result.json
```

### Run with local Ollama

```bash
python scripts/run_quality_workflow.py \
  --provider ollama \
  --model qwen2.5:1.5b \
  --output outputs/quality_workflow_ollama.json
```

CLI options:

```text
--provider {ollama,fugu}
--model MODEL
--output OUTPUT
```

---

## 8. Sample Manufacturing Case

The workflow runner includes a fixed sample manufacturing quality case:

```text
Case ID: QW-001
Process: screw tightening
Product/Part: aluminum bracket assembly
Issue: intermittent screw loosening detected after vibration test

Observed facts:
- Loosening observed in 3 out of 20 samples after vibration test
- Tightening torque target is 1.15 N.m
- Defects concentrated in the night shift lot
- No confirmed tool calibration record for the affected lot

Known constraints:
- Tightening cycle time must remain under 3 seconds per fastener
- Cannot change screw specification in this build

Available data:
- Torque trace per fastener
- Vibration test pass/fail log
- Shift roster
```

---

## 9. Project Goals

This repository is intended as a foundation for:

- Industrial AI workflow research
- Manufacturing quality workflow support
- Agent evaluation framework design
- Sakana Fugu evaluation platform experimentation
- Reproducible human-in-the-loop AI workflow design

The project is a PoC/research system, not a production manufacturing-control system.

---

## 10. Repository Structure

Key paths in the current implementation:

```text
src/
  agents/
    quality_issue_analysis_agent/
    root_cause_analysis_agent/
    countermeasure_planning_agent/
    quality_evaluation_agent/
    local_llm_process_planning_agent/
    process_planning_agent/
  llm/
    base.py
    factory.py
    fugu_provider.py
    ollama_provider.py
  workflows/
    quality_workflow.py
  harness/

scripts/
  run_quality_workflow.py
  run_quality_issue_case.py
  run_quality_issue_comparison.py
  run_planning_case.py
  run_model_comparison.py

tests/
  test_quality_workflow.py
  test_quality_issue_analysis_agent.py
  test_root_cause_analysis_agent.py
  test_countermeasure_planning_agent.py
  test_quality_evaluation_agent.py
```

---

## 11. Testing

Run the full test suite:

```bash
python -m pytest
```

Current result:

```text
248 passed
1 deselected
1 warning
```

Integration tests cover:

- agent success paths
- strict JSON parse failures
- schema validation failures
- policy validation / override behavior
- upstream trust gates
- workflow early-stop behavior
- JSON-serializable workflow result output

---

## 12. Roadmap

### Completed

- Phase1 Quality Workflow
- Provider Abstraction
- Sakana Fugu Provider
- Ollama Provider
- Strict JSON Validation
- Schema Validation
- Human Review Gate
- Workflow Orchestration
- Integration Tests

### Planned

- Benchmark Cases
- Fugu / GPT / Claude Comparison
- Workflow Analytics
- Additional Manufacturing Agents
- Broader manufacturing workflow orchestration beyond quality
- Better trace logging and run-history analysis
- Stable IDs for cross-agent references such as `hypothesis_id`

---

## 13. Disclaimer

- This repository is a research / PoC project.
- No real industrial production data is included.
- LLM-generated outputs may contain errors and must be reviewed by humans before use.
- The workflow is designed for decision support and evaluation, not autonomous production control.

---

## Final Note

Industrial AI is not only about stronger models.

It is about:

```text
structured workflows
+ constrained generation
+ deterministic validation
+ human review gates
+ continuous evaluation
```

This repository implements that foundation for manufacturing quality workflows.


