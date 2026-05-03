# Process Planning Agent 仕様書 (v0.2)

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
        "process_id": "string",
        "process_type": "machining | drilling | tapping | milling | surface_finishing | inspection | unknown",
        "process_name": "string",
        "description": "string",
        "target_feature": "string",
        "quantity": "number | null",
        "basis": ["string"],

        "quality_checkpoints": [
          {
            "checkpoint_type": "string",
            "description": "string",
            "inspection_method": "string | unknown",
            "basis": ["string"]
          }
        ],

        "confidence": 0.0,
        "needs_review": false
      }
    ],

    "findings": [
      {
        "severity": "info | warning | error",
        "message": "string",
        "related_process_id": "string | null"
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
- 品質上重要な不確実性は findings に記録し、Agent実行上の失敗は errors に記録する



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



## 8. 再検討条件

- 工程分類が unknown の場合
- quantity が null の場合
- basis が不足している場合
- confidence が 0.7 未満の場合
- 品質チェックポイントを十分に付与できない場合

## 9. 失敗条件

- result が null の場合
- manufacturing_processes が空の場合
- 全工程で basis が空の場合
- 出力スキーマに違反した場合
- Agent実行中に処理不能な errors が発生した場合



## 10. 出力フィールドの意味定義

各フィールドの役割は以下の通りとする：

- findings: 業務上の発見・注意点
- errors: Agent実行上の失敗
- notes: 補足説明

### 記録ルール

- findings は業務上の不確実性・注意点・判断保留事項を記録する
- errors は処理失敗・スキーマ違反・実行不能状態を記録する
- notes は補足的な情報（入力の不足、前提条件など）を記録する

### 例

- findings: "穴数が図面上で明確に確認できない"
- errors: "schema validation failed"
- notes: "drawing_type is unknown"



## 11. Evaluationでの扱い

- errors が1件以上ある場合、原則 failure とする
- findings が1件以上ある場合、原則 needs_review とする
- notes は評価スコアには直接反映せず、ログ・補足情報として扱う
- findings は不確実性を正しく表明できている場合、必ずしも悪い出力とはみなさない

