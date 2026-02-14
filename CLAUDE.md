# CLAUDE.md

## プロジェクト概要

@README.md

SeedLM: 意味的重要度に基づく階層的テキスト成長型生成の実験プロジェクト。
従来の左から右への逐次生成ではなく、意味の核（seed）から外側へテキストを成長させるパラダイムを探索する。

## 技術的制約

- **Python のみ**（Rust は使わない）
- **学習なし・推論のみ**（モデルの訓練は行わない）
- **実行環境**: Apple M1 Mac, 16GB Unified Memory
- **PyTorch バックエンド**: MPS（Metal Performance Shaders）

## 言語

- コード内のコメント・docstring: **日本語**
- コミットメッセージ: **日本語、自由形式**
- README・ドキュメント: **日本語**

## 主要ライブラリ

- torch (MPS backend)
- transformers
- fugashi + unidic-lite（形態素解析）
- matplotlib / plotly（可視化）

## ディレクトリ構成

```
seed-lm-experiments/
├── experiments/
│   ├── bert_confidence/       # BERT 確信度分析
│   ├── llada_unmasking/       # LLaDA アンマスク可視化
│   └── prompt_growth/         # プロンプトによる成長型生成
├── data/                      # 実験データ・結果
├── notebooks/                 # Jupyter notebooks
└── docs/                      # 追加ドキュメント
```

## やらないこと

- モデルの学習・ファインチューニングは行わない
- 16GB に収まらないモデルをローカルで動かそうとしない（Colab を使う）

## 実験の背景知識

- Ford et al. (2018) が「機能語→内容語」の順が最適と報告済み
- POINTER, InDIGO, BLM 等の挿入ベース生成が先行研究として存在
- LLaDA（マスク拡散型）の確信度順アンマスクが比較対象
- 日本語の助詞構造を活用した実験が本プロジェクトの独自性
