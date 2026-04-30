# Architecture — Industrial Agent System

## 1. 概要

本ドキュメントは、Industrial Agent System の全体構造を示す。
本システムは、単一のAgentではなく、複数Agentを **Harness上で制御するMulti-Agent System** として設計されている。



## 2. 全体構造

```text
User Input
 ↓
AgentInput (schema)
 ↓
[ Harness ]
 ├── Pre-validation
 ├── Execution
 │    └── Agent (Process Planning Agent など)
 ├── Post-validation
 ├── Evaluation
 ├── Logging
 └── Human-in-the-loop
 ↓
AgentOutput (schema)
```



## 3. レイヤ構成

システムは以下の3層で構成される。

### 3.1 Interface Layer

* AgentInput / AgentOutput
* JSONベースの構造化データ
* 外部システムとの接点



### 3.2 Control Layer（Harness）

* validator.py
* executor.py
* logger.py
* evaluation.py

責務：

* 入出力検証
* 実行制御
* エラーハンドリング
* 評価
* ログ記録



### 3.3 Agent Layer

* process_planning_agent
* （将来）quality_agent, scheduling_agent など

責務：

* 業務ロジックの実行
* 構造化出力の生成



## 4. データフロー

```text
Input(JSON)
 ↓
validate_agent_input
 ↓
execute_agent
 ↓
validate_agent_output
 ↓
log_execution
 ↓
evaluate_agent_output
 ↓
Output(JSON)
```



## 5. 設計上の重要ポイント

### 5.1 Harness-first

Agentではなく、Harnessを中心に設計する。

* Agentは差し替え可能
* Harnessは長期的資産



### 5.2 分離設計

```text
Execution ≠ Evaluation
```

* executor は実行のみ
* evaluation は評価のみ



### 5.3 状態管理

AgentOutputは以下の状態を持つ：

* success
* failure
* needs_review

これにより、処理分岐が明確になる。



### 5.4 安全停止

入力不正・出力不正の場合：

```text
Agentを実行しない or 処理停止
```



## 6. 今後の拡張

* Multi-Agent連携（Agent chaining）
* 非同期実行
* 評価自動化
* コスト最適化
* フィードバック学習



## 7. まとめ

本システムは、
「賢いAIを作る」のではなく「AIを安全に動かす構造を作る」
ことを目的としている。

そのため、すべての処理はHarness上で制御される。
