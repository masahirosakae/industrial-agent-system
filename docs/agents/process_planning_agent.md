# Process Planning Agent 仕様書 (v0.1)

## 1. 概要

Process Planning Agentは、図面または仕様情報を入力として受け取り、製造に必要な工程・作業内容・数量・品質チェックポイントを構造化して出力するAgentである。
本Agentは、工程設計者の判断を完全に代替するものではなく、判断支援・一次案生成を目的とする。



## 2. 目的

- 図面・仕様情報から加工工程を整理する
- 作業内容を構造化する
- 数量・対象箇所を明示する
- 品質確認ポイントを抽出する
- 後続のQuality AgentやEvaluation Harnessに渡せる形式で出力する



## 3. 入力

```json
{
  "task_id": "string",
  "input_type": "drawing | specification",
  "source": {
    "file_path": "string",
    "file_type": "pdf | image | text"
  },
  "metadata": {
    "part_name": "string",
    "drawing_type": "plate | shaft | assembly | unknown",
    "created_at": "timestamp"
  }
}
```



## 4. 出力

```json
{
  "task_id": "string",
  "agent_name": "planning_agent",
  "status": "success | failure | needs_review",
  "result": {
    "manufacturing_processes": [
      {
        "process_name": "string",
        "description": "string",
        "target_feature": "string",
        "quantity": "number | unknown",
        "basis": "string"
      }
    ],
    "work_items": [
      {
        "work_name": "string",
        "required_input": "string",
        "expected_output": "string"
      }
    ],
    "quality_checkpoints": [
      {
        "checkpoint": "string",
        "reason": "string",
        "inspection_method": "string | unknown"
      }
    ]
  },
  "confidence": 0.0,
  "errors": [],
  "notes": []
}
```



## 5. 判断ルール

- 図面または仕様に明記された情報のみを使用する
- 推測で工程・数量を補完しない
- 不明な場合は unknown または needs_review とする
- 判断根拠は basis に記録する
- 品質上重要な不確実性は errors または notes に明示する



## 6. Harnessとの契約

Process Planning AgentはHarness上で実行される。

Harnessは以下を担当する：

- 入力スキーマ検証
- Agent実行
- 出力スキーマ検証
- 評価
- ログ記録
- 必要時のHuman Review分岐

Process Planning Agentは副作用を持たず、入力に対して構造化出力のみを返す。



## 7. 成功条件

- 出力がスキーマに準拠している
- 工程・作業・品質確認ポイントが分離されている
- 判断根拠が明示されている
- 不確実な項目を推測せずに扱えている



