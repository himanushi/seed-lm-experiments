# SeedLM: Semantic-Importance-Ordered Text Generation

意味的重要度に基づく階層的テキスト成長型生成の実験リポジトリ

## Overview

従来の言語モデルは左から右へ逐次的にテキストを生成する（autoregressive）。
本プロジェクトでは、**文の意味的核（seed）から外側へ有機的にテキストを成長させる**という新しい生成パラダイムを探索する。

```
step0: 美味しい
step1: 美味しいです。
step2: が美味しいです。
step3: コーヒーが美味しいです。
step4: 挽いたコーヒーが美味しいです。
step5: タイムモアで挽いたコーヒーが美味しいです。
```

画像生成における低解像度→高解像度のプロセスに似ているが、テキストは離散的（UTF-8）であるため連続空間の補間ができない。
本アプローチでは、意味の「種」から構文的・修飾的要素を段階的に付加していくことで、この課題に対する新しい解を提案する。

## Motivation

### なぜ左から右でなければならないのか？

人間が文章を考えるとき、必ずしも先頭から順番に組み立てるわけではない。
「コーヒーが美味しい」と伝えたいとき、まず「美味しい」という核があり、そこに「何が？」→「コーヒー」、「どんな？」→「挽いた」と詳細が付加されていく。

小説も同様で、一番伝えたいテーマ（seed）を核として、それを補完・拡張していく営みである。
step を無限にすれば小説になり、step を 1 にすればキーワードになる。

### 日本語の言語的優位性

日本語の助詞（は、が、を、で、に）は独立したトークンとして文法的役割を明示する。
これにより「構文的骨格」と「意味的内容語」を明確に分離した実験が可能になる。
英語は語順に依存するため、文法を先に生成する実験が本質的に難しい。

## Key Research Questions

1. ~~**意味的重要度 vs 構文的重要度**: 既存の拡散型モデル（LLaDA等）は、どちらの順序でトークンをアンマスクするのか？~~ → BERTは構文的要素（助詞）を先に復元する（Exp1で確認）
2. ~~**機能語 vs 内容語**: Ford et al. (2018) は「機能語を先に生成する方がモデル品質が高い」と報告したが、これは意味的生成と矛盾するのか？~~ → 矛盾するが、逆方向（内容語→機能語補完）も56.2%の精度で実現可能（Exp1c で確認）
3. **成長型生成の実現可能性**: 固定長の穴埋め（POINTER）ではなく、テキスト自体が有機的に膨張する生成は可能か？ → 次の実験で検証予定
4. **粒度制御**: step 数をパラメータとして、1語の要約〜長文の展開まで連続的に制御できるか？ → 未検証

## Related Work

### 生成順序の研究

| 論文                                                                                      | 年         | 概要                                                        |
| ----------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------- |
| Ford et al. "The Importance of Generation Order in Language Modeling"                     | EMNLP 2018 | 2パスモデルで生成順序を比較。機能語→内容語の順が最も効果的  |
| Welleck et al. "Non-Monotonic Sequential Text Generation"                                 | ICML 2019  | 生成順序を事前に指定せず学習。easy-first 的な振る舞いを獲得 |
| Gu et al. "InDIGO: Insertion-based Decoding with Automatically Inferred Generation Order" | TACL 2019  | 挿入操作による任意順序生成。モデルが適応的な生成順序を学習  |

### 挿入・段階的生成

| 論文                                                      | 年         | 概要                                                          |
| --------------------------------------------------------- | ---------- | ------------------------------------------------------------- |
| Stern et al. "Insertion Transformer"                      | ICML 2019  | 並列挿入により対数時間で生成。バランス二分木的順序を探索      |
| Zhang et al. "POINTER"                                    | EMNLP 2020 | キーワード制約から段階的に文を生成。名詞・動詞→助詞・冠詞の順 |
| Shen et al. "Blank Language Models"                       | EMNLP 2020 | 空白を動的に挿入・展開。任意位置・長さのテキスト生成          |
| Tan et al. "Progressive Generation of Long Text" (ProGen) | NAACL 2021 | TF-IDFによる重要度順の語彙拡大。低→高解像度的な段階的生成     |

### 構文木ベースの生成

| 論文                                                             | 年   | 概要                                                  |
| ---------------------------------------------------------------- | ---- | ----------------------------------------------------- |
| Guo et al. "Top-Down Tree Structured Text Generation"            | 2018 | 構成素構文木をトップダウン・幅優先で生成              |
| Casas et al. "Syntax-driven Iterative Expansion Language Models" | 2020 | 依存構文解析木で Transformer を駆動し反復的に文を生成 |

### 拡散型言語モデル（生成順序が固定されない）

| 論文 / モデル                      | 年   | 概要                                                                         |
| ---------------------------------- | ---- | ---------------------------------------------------------------------------- |
| LLaDA                              | 2025 | マスク拡散方式の 8B パラメータモデル。ランダムにマスクし確信度順にアンマスク |
| Mercury (Inception Labs)           | 2025 | 初の商用拡散型 LLM。10x 高速。coarse-to-fine 生成                            |
| Gemini Diffusion (Google DeepMind) | 2025 | 1479 tokens/sec。コーディングに強く推論に弱い                                |

## 本プロジェクトの差別化ポイント

既存研究との差分を明確にする。

| 観点         | 既存研究                           | SeedLM                                     |
| ------------ | ---------------------------------- | ------------------------------------------ |
| 生成方式     | 固定長の穴埋め or 挿入             | テキスト自体が成長（出力長が動的に決まる） |
| 重要度の基準 | TF-IDF / モデル確信度 / 構文的位置 | 意味的重要度（情報量ベース）               |
| 対象言語     | ほぼ全て英語                       | 日本語（助詞構造の活用）                   |
| 動機         | 高速化 / 制約付き生成              | 人間の思考過程に近い生成                   |
| 粒度制御     | なし（固定 step）                  | step 数による連続的な詳細度制御            |

## Experiments

学習なし（推論のみ）で実施。モデルは `cl-tohoku/bert-base-japanese-whole-word-masking` を使用。

### Experiment 1: BERT 品詞別アンマスク順序分析

全マスク状態からBERTがどの品詞を先に復元するかを観察。

**結果**: BERTは構文的骨格（助詞）を最初に復元する。

| 品詞   | 平均アンマスクステップ | 分類   |
| ------ | ---------------------- | ------ |
| 助詞   | 4.5                    | 機能語 |
| 動詞   | 6.4                    | 内容語 |
| 名詞   | 6.7                    | 内容語 |
| 助動詞 | 7.2                    | 機能語 |
| 形容詞 | 8.7                    | 内容語 |
| 副詞   | 12.0                   | 内容語 |

Ford et al. (2018) の「機能語→内容語の順が最適」という報告を日本語の助詞構造で独立に確認。BERTにとって自然な順序は「構文→意味」であり、SeedLMが目指す「意味→構文」とは真逆であることが判明。

### Experiment 1b: 方向別マスク補完精度

内容語と機能語、どちらを手がかりとして与えた方がBERTは補完できるかを検証。

**結果**: 圧倒的な非対称性を確認。

| パターン             | 精度         | 平均確信度 |
| -------------------- | ------------ | ---------- |
| A: 内容語→機能語補完 | 21.9% (7/32) | 0.1166     |
| B: 機能語→内容語補完 | 0.0% (0/28)  | 0.0017     |

確信度に約70倍の差。内容語が与えられれば助詞をある程度補完できるが、助詞だけでは内容語はほぼ予測不能（最頻出名詞に崩壊）。SeedLMの「意味の核から構文を補完する」方向が、逆方向よりもはるかに実現可能性が高いことを示唆。

### Experiment 1c: 反復的機能語補完

一斉マスクではなく、確信度順に1語ずつアンマスクする反復方式で精度を検証。

**結果**: 反復的アンマスクで精度が2.5倍以上に向上。

| 方式             | 精度          | 改善幅   |
| ---------------- | ------------- | -------- |
| 一斉マスク       | 21.9% (7/32)  | baseline |
| 反復的アンマスク | 56.2% (18/32) | +34.4pt  |

精度推移にも明確なパターンが存在：

| 進捗           | 精度   | 解釈                                              |
| -------------- | ------ | ------------------------------------------------- |
| 序盤 (0-25%)   | 50%    | 高確信度の助詞から着手                            |
| 中盤 (26-75%)  | 75-86% | 文脈が揃うほど精度が急上昇                        |
| 終盤 (76-100%) | 27%    | 「です/ます」等の文体選択は意味的制約で決まらない |

終盤の精度低下は「です/ます」vs「だ/である」のような文体レベルの選択であり、意味的制約だけでは決定できないもの。これは欠点ではなく、意味レイヤーと文体レイヤーの自然な分離を示している。

### 暫定的な生成レイヤーモデル

実験結果から、以下の3層構造が浮かび上がった：

```
Layer 1: 意味の核（内容語）     ← 人間 or LLM が生成
Layer 2: 文法的接続（助詞）     ← BERT で 56-86% 補完可能
Layer 3: 文体選択（です/ます）  ← パラメータで制御
```

### 今後の実験

- **Experiment 2: LLaDA アンマスク可視化** — 拡散型LLMが品詞をどの順でアンマスクするか観察（Colab予定）
- **Experiment 3: 段階的成長の実装** — 核となる1語から始めて、内容語→機能語の順で段階的に文を成長させる

## Tech Stack

| 用途       | 技術                                        |
| ---------- | ------------------------------------------- |
| モデル推論 | Python, transformers, PyTorch (MPS backend) |
| 形態素解析 | fugashi (MeCab wrapper)                     |
| 可視化     | matplotlib / plotly                         |

## Project Structure

```
seed-lm-experiments/
├── README.md
├── experiments/
│   ├── bert_confidence/       # Experiment 1: BERT 確信度分析
│   ├── llada_unmasking/       # Experiment 2: LLaDA アンマスク可視化
│   └── prompt_growth/         # Experiment 3: プロンプトによる成長型生成
├── data/                      # 実験データ・結果
├── notebooks/                 # Jupyter notebooks
└── docs/                      # 追加ドキュメント
    └── related_work.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch transformers fugashi unidic-lite matplotlib
```

## Environment

- Apple M1 Mac, 16GB Unified Memory
- Python 3.11+

## References

- Ford, N., Duckworth, D., Norouzi, M., & Dahl, G. (2018). The Importance of Generation Order in Language Modeling. _EMNLP 2018_.
- Gu, J., Liu, Q., & Cho, K. (2019). Insertion-based Decoding with Automatically Inferred Generation Order. _TACL_.
- Zhang, Y., et al. (2020). POINTER: Constrained Progressive Text Generation via Insertion-based Generative Pre-training. _EMNLP 2020_.
- Shen, T., et al. (2020). Blank Language Models. _EMNLP 2020_.
- Tan, B., et al. (2021). Progressive Generation of Long Text with Pretrained Language Models. _NAACL 2021_.
- Guo, Q., et al. (2018). Top-Down Tree Structured Text Generation. _arXiv:1808.04865_.
- Casas, N., et al. (2020). Syntax-driven Iterative Expansion Language Models for Controllable Text Generation. _arXiv:2004.02211_.
- Nie, S., et al. (2025). LLaDA: Large Language Diffusion with mAsking. _arXiv_.
- Welleck, S., et al. (2019). Non-Monotonic Sequential Text Generation. _ICML 2019_.

## License

MIT

## Author

個人開発・研究プロジェクト
