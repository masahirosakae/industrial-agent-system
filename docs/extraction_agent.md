# Drawing / Specification Extraction Agent 設計書

## 1. 概要（Overview）

Drawing / Specification Extraction Agent は、機械図面（PDF）および仕様テキストを解析し、図面上に明示された情報（寸法・加工指示・表面性状・幾何公差など）を構造化データとして抽出するエージェントである。

本Agentは、後続の Process Planning Agent に対して「工程計画に必要な事実情報」を提供することを目的とする。



## 2. 目的（Purpose）

* 図面PDF / 仕様テキストから、加工・品質に関わる情報を抽出する
* 抽出結果を構造化し、Process Planning Agent に渡す
* ハルシネーションを排除した「完全な事実ベース抽出」を実現する
* 抽出精度を評価可能な形式で出力する



## 3. スコープ（Scope）

### 対象

* 機械図面（PDF）
* 仕様書（テキスト）

### 非対象（MVP時点）

* CADネイティブデータ（STEP, IGES など）
* 完全な図形理解（幾何構造解析）
* 工程計画の決定



## 4. 責務（Responsibilities）

### 本Agentが行うこと

* 図面から以下の情報を抽出する

  * 寸法記号
  * 穴・ねじ加工指示
  * 表面性状
  * 幾何公差・データム
  * 溶接記号
  * 注記テキスト
* 抽出対象の網羅的探索（図面全体スキャン）
* 抽出結果の構造化（JSON）
* 判定ステータス（確信度）の付与
* 抽出根拠（basis）の保持



### 本Agentが行わないこと

* 工程順序の決定
* 加工方法の最終判断
* 図面に存在しない情報の推測
* 補完・一般化による意味付け



## 5. アーキテクチャ（Architecture）

```text
PDF / Text
    ↓
Document Loader
    ↓
RawDocument
    ↓
Extraction Agent
    ↓
ExtractionResult
    ↓
Adapter
    ↓
AgentInput
    ↓
Process Planning Agent
```



## 6. 入力仕様（Input）

### ExtractionInput

```python
@dataclass
class ExtractionInput:
    task_id: str
    document: RawDocument
```



### RawDocument

```python
@dataclass
class RawDocument:
    task_id: str
    source_type: Literal["text", "pdf"]
    pages: list[DocumentPage]
    warnings: list[str]
    metadata: dict[str, Any]
```



### DocumentPage

```python
@dataclass
class DocumentPage:
    page_number: int
    text: str
    extraction_method: Literal["text", "ocr", "vision"]
    confidence: float
```



## 7. 出力仕様（Output）

### DrawingExtractionResult

```python
@dataclass
class DrawingExtractionResult:
    task_id: str
    source_type: str
    raw_items: list[str]
    symbols: list[ExtractionSymbol]
    notes: list[str]
    warnings: list[str]
    confidence: float
```



### ExtractionSymbol

```python
@dataclass
class ExtractionSymbol:
    symbol: str
    raw_text: str
    category: str
    meaning: str
    judgment: Literal["clear", "uncertain", "unreadable"]
    basis: str
```



## 8. 抽出カテゴリ（Extraction Categories）

```text
- dimension_symbol（寸法記号）
- thread_fit（ねじ・はめあい）
- hole_process（穴加工）
- surface_finish（表面性状）
- geometric_tolerance（幾何公差）
- datum（データム）
- welding_symbol（溶接記号）
- reference_dimension（参考寸法）
- theoretical_dimension（理論寸法）
- text_instruction（注記テキスト）
- unknown
```



## 9. 判定ステータス（Judgment）

| 表示    | 内部値        |
| ----- | ---------- |
| 明確    | clear      |
| やや不確実 | uncertain  |
| 判読不可  | unreadable |



## 10. 抽出ルール（Extraction Rules）

### 10.1 事実ベース原則

* 図面上に存在する情報のみを抽出する
* 推測・補完・一般化は禁止



### 10.2 網羅性

* 図面全体を走査する
* 特に以下を重点探索

  * 寸法線周辺
  * 指示線の先端
  * 表題欄
  * 注記事項エリア



### 10.3 情報粒度

* 記号単体ではなく、数値・サフィックスを含める

  * 例：`φ20 H7`, `Ra 3.2`, `M8×1.25`



### 10.4 重複処理

* 同一記号は集約する
* 数量情報として保持する



### 10.5 表面粗さの特例

* 旧JIS記号（▽）を必ず探索
* 近傍数値をセットで取得



## 11. 内部処理フロー（Chain of Thought）

```text
Step 1:
図面上の全テキスト・寸法を列挙（raw_items）

Step 2:
カテゴリ分類

Step 3:
構造化（symbols生成）

Step 4:
自己検証

Step 5:
出力生成
```



## 12. 自己検証（Validation）

以下を満たすこと：

* 抽出情報はすべて入力に存在する
* 抽出漏れがない
* 重複が統合されている
* basisがすべての項目に存在する
* 判読不可に対して推測していない



## 13. PDF対応方針（PDF Handling）

### v0.1

* テキスト抽出PDFのみ対応

### v0.2

* OCR対応

### v0.3

* Vision LLM対応（図面領域解析）



## 14. Rule-based Fallback

LLMが使用できない場合：

* 正規表現ベース抽出
* キーワード辞書による分類
* 未抽出項目は warning とする



## 15. LLM利用方針

* 抽出精度向上のために使用
* 出力は必ず構造化
* parserで検証
* 不正出力はfallbackへ



## 16. Process Planning Agent連携

```text
ExtractionResult
    ↓
Adapter
    ↓
AgentInput
```

* symbols → manufacturing features に変換
* basis → 根拠情報として保持



## 17. テスト方針（Testing）

### 単体テスト

* テキスト入力から正しく抽出できるか

### 異常系

* 不明瞭データ → unreadable

### 統合テスト

* Extraction → Process Planning 連携



## 18. 今後の拡張

* CAD連携
* Feature Recognition
* 自動工程生成との統合
* マルチエージェント化



## 19. 設計原則（Design Principles）

* Harness-first
* Evaluation-first
* No hallucination
* Explainability（basis必須）
* Modular architecture


