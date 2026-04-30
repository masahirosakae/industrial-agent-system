# Harness 仕様書 (v0.1)

## 1. 概要

本Harnessは、Industrial Agent SystemにおいてAgentを安全かつ再現性高く実行するための制御レイヤである。
Agent単体の知能ではなく、Harness上での実行を前提とする。



## 2. 設計原則

* Harness-first（Agentより先に制御を定義する）
* Evaluation-first（評価を設計に含める）
* Human-in-the-loop（重要判断は人間が関与する）
* Reproducibility（再現可能）
* Traceability（追跡可能）



## 3. 入力スキーマ

すべての入力は構造化データとして扱う。

例：

```json
{
  "task_id": "string",
  "input_type": "drawing | spec",
  "content": "string or file path",
  "metadata": {
    "created_at": "timestamp",
    "source": "string"
  }
}
```


## 4. 出力スキーマ

Agent出力は必ず構造化される。

```json
{
  "task_id": "string",
  "status": "success | failure",
  "result": {
    "processes": [],
    "quantities": [],
    "quality_checks": []
  },
  "confidence": "float",
  "errors": []
}
```



## 5. 実行フロー

```text
Input
 ↓
Pre-validation
 ↓
Agent Execution
 ↓
Post-validation
 ↓
Evaluation
 ↓
Logging
 ↓
Human Review (optional)
```



## 6. 検証ルール

* 必須フィールドの存在確認
* 型チェック
* 空値チェック
* 禁止パターン（推測・未定義項目）



## 7. 評価指標

* 精度（Accuracy）
* 再現性（Consistency）
* 出力妥当性（Validity）
* カバレッジ（Coverage）



## 8. 記録

すべての実行を記録：

* Input
* Output
* Validation結果
* 評価結果
* 実行時間



## 9. 今後の拡張

* Agent chaining
* Multi-agent coordination
* Feedback learning
* Cost optimization



## 10. Agent契約

Agentは以下の契約に従う：

- 入力は必ず入力スキーマに従う
- 出力は必ず出力スキーマに従う
- 不確実な場合は推測せず、エラーとして明示する
- 副作用を持たない（stateless）

HarnessはAgentをブラックボックスとして扱い、入出力のみで制御する。