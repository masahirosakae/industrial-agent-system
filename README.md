# Industrial Agent System 日本語README

Industrial Agent System は、製造業務を対象とした **評価ファースト / Human Review Gate 前提** の Multi-Agent System 研究プロジェクトです。

![status](https://img.shields.io/badge/status-PoC-orange)
![phase](https://img.shields.io/badge/Phase1%20Quality%20Workflow-Completed-brightgreen)
![tests](https://img.shields.io/badge/tests-272%20passed-brightgreen)

英語版: [README_EN.md](./README_EN.md)

---

## 1. プロジェクト概要

**Industrial Agent System** は、製造現場の業務を複数の Agent に分解し、LLM の出力をそのまま信頼せず、検証・評価・人間レビューを通じて安全に扱うための研究用システムです。

主なキーワード:

- Industrial AI
- Manufacturing Workflow
- Multi-Agent System
- Evaluation First
- Human Review Gate
- Workflow Orchestration

このプロジェクトの中心は「より賢いAIモデルを作ること」ではなく、**LLMを産業ワークフローの中で安全に使うための仕組みを作ること** です。

---

## 2. 現在の状態

```text
Phase1 Quality Workflow Completed
```

現在、品質問題に対する Phase1 Workflow が完成しています。

実装済み:

- Provider Abstraction
- Sakana Fugu Provider
- Ollama Provider
- Strict JSON Validation
- Schema Validation
- Policy Validation
- Human Review Gate
- Workflow Orchestration
- Integration Tests

pytest 結果:

```text
248 passed
1 deselected
1 warning
```

---

## 3. 実装済み Agent

### QualityIssueAnalysisAgent

品質問題ケースを受け取り、品質問題の初期分析を行います。

主な出力:

- issue_summary
- suspected_causes
- containment_actions
- investigation_plan
- additional_data_needed
- risk_if_unresolved
- verification_points

### RootCauseAnalysisAgent

QualityIssueAnalysisAgent の出力を受け取り、根本原因分析を行います。

主な責務:

- 5Why分析
- FTA風の分解
- Fact / Assumption / Hypothesis の分離
- missing evidence の抽出
- 原因仮説ごとの confidence 付与

### CountermeasurePlanningAgent

RootCauseAnalysisAgent の出力を受け取り、対策計画を作成します。

主な責務:

- 封じ込め対策
- 恒久対策
- 検証計画
- リスク評価
- 優先順位付け
- evidence 不足時の review_reasons 生成

### QualityEvaluationAgent

RootCauseAnalysisAgent と CountermeasurePlanningAgent の出力を受け取り、最終品質ゲートとして評価します。

主な責務:

- 根拠十分性評価
- ハルシネーションリスク検出
- RCA / CMP 整合性確認
- 対策計画の妥当性評価
- 検証計画の十分性評価
- approval_readiness 評価
- 最終判定 `accepted` / `needs_review`

---

## 4. アーキテクチャ原則

### Evaluation First

LLM 出力はそのまま正しいものとして扱いません。

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

各 Agent は、LLM 出力を候補として扱い、決定論的な検証を通過した場合のみ次のステップへ渡します。

### Provider Abstraction

Agent は具体的な Provider 実装に依存せず、`LLMProvider` 抽象を通じて LLM を呼び出します。

対応 Provider:

- Sakana Fugu
- Ollama

### Strict Structured Outputs

Phase1 Quality Workflow の Agent はすべて以下の方針に統一されています。

- `json.loads(text)` による whole-response JSON parse
- Markdown 禁止
- code fence 禁止
- JSON 前後の prose 禁止
- `{` から `}` の substring 抽出禁止
- schema validation
- policy validation
- `needs_review` fallback

### Human Review Gate

Workflow は各 Agent の `status` を見て次段に進むか停止するかを決定します。

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

`needs_review` 状態の出力は、次段 Agent に渡されません。

---

## 5. Phase1 Quality Workflow

現在の Workflow は以下です。

```text
QualityIssueAnalysisAgent  (QIA)
↓
RootCauseAnalysisAgent     (RCA)
↓
CountermeasurePlanningAgent (CMP)
↓
QualityEvaluationAgent     (QEA)
```

省略図:

```text
QIA
↓
RCA
↓
CMP
↓
QEA
```

実装場所:

```text
src/workflows/quality_workflow.py
```

実行スクリプト:

```text
scripts/run_quality_workflow.py
```

---

## 6. Workflow Result States

### success

すべての Agent が `status="success"` で完了し、最終 Agent である QualityEvaluationAgent が以下を返した状態です。

```text
overall_judgement == accepted
```

Workflow Result:

```text
status = success
final_judgement = accepted
stopped_at = None
```

### needs_review

任意の Agent が `needs_review` になった場合、または QualityEvaluationAgent が accepted と判断しなかった場合です。

Workflow Result:

```text
status = needs_review
final_judgement = needs_review
stopped_at = <agent_name>
```

想定外例外も安全側に倒し、`needs_review` として返します。

---

## 7. 実行例

### Sakana Fugu で実行

環境変数を設定します。

```bash
export FUGU_API_KEY="<your-key>"
export FUGU_BASE_URL="https://<your-fugu-endpoint>/v1"
export FUGU_MODEL="<your-fugu-model>"
```

実行:

```bash
python scripts/run_quality_workflow.py --provider fugu
```

JSON 結果を保存:

```bash
python scripts/run_quality_workflow.py \
  --provider fugu \
  --output outputs/result.json
```

評価分析用に 1 行 1 run の JSONL trace を追記:

```bash
python scripts/run_quality_workflow.py \
  --provider fugu \
  --output outputs/result.json \
  --trace outputs/quality_workflow_runs.jsonl
```

### Ollama で実行

```bash
python scripts/run_quality_workflow.py \
  --provider ollama \
  --model qwen2.5:1.5b \
  --output outputs/quality_workflow_ollama.json
```

CLI オプション:

```text
--provider {ollama,fugu}
--model MODEL
--output OUTPUT
--trace TRACE
--include-raw-output
--case-version CASE_VERSION
```

### Workflow trace JSONL

`--trace PATH` を指定すると、1 run につき 1 行の JSON エントリが追記されます。
Fugu / GPT / Claude 比較評価の再現性確保が目的です。
各エントリには `run_id`, `case_id`, `case_version`, `provider_name`, `model`, workflow 全体の status, `started_at` / `finished_at` / `total_latency_seconds`, 各 step の status と latency, および `qea_scores` のフラットな roll-up が含まれます。 raw LLM 出力はデフォルトで含めません
（ローカルデバッグ時のみ `--include-raw-output` を使用）。

QEA の各 score は `0.0-1.0` に正規化されています。

---

## 8. サンプル製造品質ケース

`run_quality_workflow.py` には、ねじ締結不良のサンプルケースが含まれています。

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

## 9. プロジェクト目標

本プロジェクトの目的:

- Industrial AI workflow research
- Manufacturing quality workflow support
- Agent evaluation framework
- Sakana Fugu evaluation platform
- Human-in-the-loop AI workflow の設計検証

---

## 10. 主要ディレクトリ

```text
src/
  agents/
    quality_issue_analysis_agent/
    root_cause_analysis_agent/
    countermeasure_planning_agent/
    quality_evaluation_agent/
  llm/
    base.py
    factory.py
    fugu_provider.py
    ollama_provider.py
  workflows/
    quality_workflow.py

scripts/
  run_quality_workflow.py

tests/
  test_quality_workflow.py
  test_quality_issue_analysis_agent.py
  test_root_cause_analysis_agent.py
  test_countermeasure_planning_agent.py
  test_quality_evaluation_agent.py
```

---

## 11. テスト

全テスト実行:

```bash
python -m pytest
```

現在の結果:

```text
272 passed
1 deselected
1 warning
```

テスト対象:

- 正常系
- strict JSON parse failure
- schema validation failure
- policy validation / override
- upstream trust gate
- workflow early stop
- workflow result JSON serialization

---

## 12. Roadmap

### Completed

- Phase1 Quality Workflow
- Provider Abstraction
- Sakana Fugu Provider
- Ollama Provider
- Workflow Orchestration
- Integration Tests

### Planned

- Benchmark Cases
- Fugu / GPT / Claude Comparison
- Workflow Analytics
- Additional Manufacturing Agents
- Quality Workflow の trace logging / run history 強化
- `hypothesis_id` など安定 ID による cross-agent reference 改善

---

## 13. 注意事項

- 本リポジトリは研究 / PoC 用です。
- 実製造データは含まれていません。
- LLM 出力は必ず人間が確認してください。
- 本システムは意思決定支援と評価を目的としており、自律的な製造制御を目的としていません。

---

## 最後に

産業AIに必要なのは、単に強いモデルではありません。

```text
構造化されたワークフロー
+ 制約された生成
+ 決定論的な検証
+ Human Review Gate
+ 継続的な評価
```

Industrial Agent System は、その土台を製造品質ワークフローから実装しています。

