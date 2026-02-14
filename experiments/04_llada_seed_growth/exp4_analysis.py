"""
Experiment 4: 分析・可視化スクリプト

exp4_seed_growth.py の実行結果（JSON）を読み込み、
品詞別集計 CSV、3 層分析 CSV、Exp3 比較 CSV、可視化を生成する。

使い方:
    python experiments/04_llada_seed_growth/exp4_analysis.py
    python experiments/04_llada_seed_growth/exp4_analysis.py --results-dir results/
    python experiments/04_llada_seed_growth/exp4_analysis.py --no-plot  # 可視化なし
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# プロジェクトルートをパスに追加
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from utils.pos_alignment import (
    CONTENT_POS,
    LAYER_NAMES,
    LAYER_POS,
    classify_layer,
)


# ============================================================
# データ読み込み
# ============================================================

def load_results(results_dir):
    """exp4_raw_results.json を読み込む"""
    path = os.path.join(results_dir, "exp4_raw_results.json")
    if not os.path.exists(path):
        print(f"結果ファイルが見つかりません: {path}")
        print("先に exp4_seed_growth.py を実行してください。")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"読み込み: {path} ({len(data)} エントリ)")
    return data


# ============================================================
# CSV 生成
# ============================================================

def generate_pos_summary_csv(all_results, output_dir):
    """
    品詞別集計 CSV: exp4_pos_summary.csv

    品詞, 平均正規化ステップ, 標準偏差, n, 分類(機能語/内容語), モード
    """
    for mode in ("A", "B", "C"):
        results = [r for r in all_results if r["mode"] == mode]
        if not results:
            continue

        pos_stats = defaultdict(lambda: {"norm_steps": [], "confidences": []})

        for r in results:
            for d in r.get("merged_data", []):
                if d.get("is_seed"):
                    continue
                s = pos_stats[d["pos"]]
                s["norm_steps"].append(d["norm_step"])
                s["confidences"].append(d["confidence"])

        rows = []
        for pos, s in sorted(
            pos_stats.items(), key=lambda x: np.mean(x[1]["norm_steps"]),
        ):
            word_type = "内容語" if pos in CONTENT_POS else "機能語"
            rows.append({
                "mode": mode,
                "pos": pos,
                "mean_norm_step": round(np.mean(s["norm_steps"]), 4),
                "std_norm_step": round(np.std(s["norm_steps"]), 4),
                "mean_confidence": round(np.mean(s["confidences"]), 4),
                "n": len(s["norm_steps"]),
                "word_type": word_type,
            })

        filename = f"exp4_pos_summary_mode{mode.lower()}.csv"
        path = os.path.join(output_dir, filename)
        _write_csv(path, rows, [
            "mode", "pos", "mean_norm_step", "std_norm_step",
            "mean_confidence", "n", "word_type",
        ])


def generate_layer_analysis_csv(all_results, output_dir):
    """
    3 層モデル検証 CSV: exp4_layer_analysis.csv

    文, L1_avg_step, L2_avg_step, L3_avg_step, L1<L2<L3, モード
    """
    rows = []

    for r in all_results:
        if r["mode"] == "C":
            continue  # Mode C は目標文がないので別扱い

        merged = r.get("merged_data", [])
        if not merged:
            continue

        layer_avgs = {}
        for layer_name in LAYER_NAMES:
            steps = [
                d["norm_step"] for d in merged
                if d.get("layer") == layer_name and not d.get("is_seed")
            ]
            layer_avgs[layer_name] = np.mean(steps) if steps else None

        l1 = layer_avgs.get(LAYER_NAMES[0])
        l2 = layer_avgs.get(LAYER_NAMES[1])
        l3 = layer_avgs.get(LAYER_NAMES[2])

        ordered = "N/A"
        if l1 is not None and l2 is not None and l3 is not None:
            ordered = "Yes" if l1 < l2 < l3 else "No"

        rows.append({
            "mode": r["mode"],
            "target_text": r.get("target_text", ""),
            "seed_text": r.get("seed_text", ""),
            "L1_avg_step": round(l1, 4) if l1 is not None else "",
            "L2_avg_step": round(l2, 4) if l2 is not None else "",
            "L3_avg_step": round(l3, 4) if l3 is not None else "",
            "L1_lt_L2_lt_L3": ordered,
        })

    if rows:
        path = os.path.join(output_dir, "exp4_layer_analysis.csv")
        _write_csv(path, rows, [
            "mode", "target_text", "seed_text",
            "L1_avg_step", "L2_avg_step", "L3_avg_step", "L1_lt_L2_lt_L3",
        ])


def generate_exp3_comparison_csv(all_results, output_dir):
    """
    Exp3 との比較 CSV: exp4_vs_exp3_comparison.csv

    Exp3 の BERT 結果（ハードコード）と Exp4 Mode A の結果を比較。
    """
    # Exp3 Mode A の結果（README.md より）
    exp3_results = {
        "タイムモアで挽いたコーヒーが美味しいです。": {"accuracy": 0.0, "seed": "美味しい"},
        "東京の桜は春に最も美しく咲きます。": {"accuracy": 20.0, "seed": "咲き"},
        "彼女は毎朝公園でジョギングをしています。": {"accuracy": 18.2, "seed": "ジョギング"},
        "この本はとても面白かったので友達に勧めました。": {"accuracy": 25.0, "seed": "面白かった"},
        "雨が降っているから傘を持っていきなさい。": {"accuracy": 27.3, "seed": "降って"},
    }

    mode_a_results = [r for r in all_results if r["mode"] == "A"]
    if not mode_a_results:
        return

    rows = []
    for r in mode_a_results:
        text = r.get("target_text", "")
        exp3 = exp3_results.get(text)
        exp4_accuracy = r.get("accuracy", 0) * 100

        rows.append({
            "target_text": text,
            "seed_text": r.get("seed_text", ""),
            "exp3_bert_accuracy": round(exp3["accuracy"], 1) if exp3 else "",
            "exp4_llada_accuracy": round(exp4_accuracy, 1),
            "diff": round(exp4_accuracy - exp3["accuracy"], 1) if exp3 else "",
        })

    if rows:
        path = os.path.join(output_dir, "exp4_vs_exp3_comparison.csv")
        _write_csv(path, rows, [
            "target_text", "seed_text",
            "exp3_bert_accuracy", "exp4_llada_accuracy", "diff",
        ])


def _write_csv(path, rows, fieldnames):
    """CSV ファイルを書き出す"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"保存: {path} ({len(rows)} rows)")


# ============================================================
# 可視化
# ============================================================

def generate_plots(all_results, output_dir):
    """全可視化を生成"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib が見つかりません。可視化をスキップします。")
        return

    # 日本語フォント設定（Colab / Mac 対応）
    for font in [
        "Noto Sans CJK JP", "IPAGothic", "Hiragino Sans",
        "MS Gothic", "DejaVu Sans",
    ]:
        try:
            matplotlib.rcParams["font.family"] = font
            break
        except Exception:
            continue

    # --- 品詞別アンマスク順序（Mode A） ---
    mode_a = [r for r in all_results if r["mode"] == "A"]
    if mode_a:
        _plot_pos_order(mode_a, output_dir, plt, Patch)

    # --- BERT vs LLaDA 比較 ---
    if mode_a:
        _plot_bert_comparison(mode_a, output_dir, plt, Patch)

    # --- 3 層モデル比較（Mode A） ---
    if mode_a:
        _plot_layer_comparison(mode_a, output_dir, plt)

    # --- Exp3 との精度比較 ---
    if mode_a:
        _plot_exp3_comparison(mode_a, output_dir, plt)


def _plot_pos_order(mode_a_results, output_dir, plt, Patch):
    """品詞別アンマスク順序の棒グラフ"""
    pos_stats = defaultdict(list)
    for r in mode_a_results:
        for d in r.get("merged_data", []):
            if not d.get("is_seed"):
                pos_stats[d["pos"]].append(d["norm_step"])

    if not pos_stats:
        return

    sorted_pos = sorted(pos_stats.items(), key=lambda x: np.mean(x[1]))
    labels = [p[0] for p in sorted_pos]
    values = [np.mean(p[1]) for p in sorted_pos]
    colors = ["#e74c3c" if p in CONTENT_POS else "#3498db" for p in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("Normalized Unmask Step (0=first, 1=last)")
    ax.set_title("Exp4 LLaDA Mode A: POS-based Unmask Order (seed growth)")

    legend_elements = [
        Patch(facecolor="#e74c3c", label="Content words"),
        Patch(facecolor="#3498db", label="Function words"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "exp4_pos_unmask_order.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {path}")


def _plot_bert_comparison(mode_a_results, output_dir, plt, Patch):
    """BERT（Exp1）vs LLaDA（Exp4）品詞別比較"""
    # Exp1 BERT ベースライン（README.md より）
    bert_baseline = {
        "助詞": 4.5 / 12.0,
        "動詞": 6.4 / 12.0,
        "名詞": 6.7 / 12.0,
        "助動詞": 7.2 / 12.0,
        "形容詞": 8.7 / 12.0,
        "副詞": 12.0 / 12.0,
    }

    llada_pos = defaultdict(list)
    for r in mode_a_results:
        for d in r.get("merged_data", []):
            if not d.get("is_seed"):
                llada_pos[d["pos"]].append(d["norm_step"])

    llada_summary = {pos: np.mean(vals) for pos, vals in llada_pos.items()}

    common_pos = sorted(
        set(bert_baseline.keys()) & set(llada_summary.keys()),
        key=lambda p: bert_baseline.get(p, 1),
    )
    if not common_pos:
        return

    bert_vals = [bert_baseline[p] for p in common_pos]
    llada_vals = [llada_summary[p] for p in common_pos]

    x_pos = np.arange(len(common_pos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - width / 2, bert_vals, width, label="BERT (Exp1)", color="#2ecc71")
    ax.bar(x_pos + width / 2, llada_vals, width, label="LLaDA (Exp4)", color="#9b59b6")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(common_pos)
    ax.set_ylabel("Normalized Unmask Step (0=first, 1=last)")
    ax.set_title("BERT (Exp1) vs LLaDA (Exp4 Mode A): POS Unmask Order")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "exp4_bert_vs_llada.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {path}")


def _plot_layer_comparison(mode_a_results, output_dir, plt):
    """3 層モデルの文別比較"""
    data = []
    for r in mode_a_results:
        merged = r.get("merged_data", [])
        if not merged:
            continue

        entry = {"text": r.get("target_text", "")[:15] + "..."}
        for i, layer_name in enumerate(LAYER_NAMES):
            steps = [
                d["norm_step"] for d in merged
                if d.get("layer") == layer_name and not d.get("is_seed")
            ]
            entry[f"L{i+1}"] = np.mean(steps) if steps else None
        data.append(entry)

    if not data:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(data))
    width = 0.25
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    for i in range(3):
        key = f"L{i+1}"
        vals = [d.get(key, 0) or 0 for d in data]
        ax.bar(x + i * width, vals, width, label=LAYER_NAMES[i], color=colors[i])

    ax.set_xticks(x + width)
    ax.set_xticklabels([d["text"] for d in data], rotation=15, ha="right")
    ax.set_ylabel("Normalized Unmask Step")
    ax.set_title("Exp4 Mode A: 3-Layer Model per Sentence")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "exp4_layer_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {path}")


def _plot_exp3_comparison(mode_a_results, output_dir, plt):
    """Exp3（BERT）vs Exp4（LLaDA）精度比較"""
    exp3_results = {
        "タイムモアで挽いたコーヒーが美味しいです。": 0.0,
        "東京の桜は春に最も美しく咲きます。": 20.0,
        "彼女は毎朝公園でジョギングをしています。": 18.2,
        "この本はとても面白かったので友達に勧めました。": 25.0,
        "雨が降っているから傘を持っていきなさい。": 27.3,
    }

    labels = []
    bert_acc = []
    llada_acc = []

    for r in mode_a_results:
        text = r.get("target_text", "")
        if text in exp3_results:
            labels.append(text[:12] + "...")
            bert_acc.append(exp3_results[text])
            llada_acc.append(r.get("accuracy", 0) * 100)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, bert_acc, width, label="Exp3 BERT", color="#2ecc71")
    ax.bar(x + width / 2, llada_acc, width, label="Exp4 LLaDA", color="#9b59b6")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Exp3 (BERT) vs Exp4 (LLaDA): Seed-guided Growth Accuracy")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "exp4_vs_exp3_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存: {path}")


# ============================================================
# サマリー出力
# ============================================================

def print_summary(all_results):
    """全モードのサマリーを表示"""
    mode_groups = defaultdict(list)
    for r in all_results:
        mode_groups[r["mode"]].append(r)

    for mode, results in sorted(mode_groups.items()):
        print(f"\n{'=' * 65}")
        print(f"  Mode {mode} サマリー ({len(results)} 文)")
        print(f"{'=' * 65}")

        all_merged = []
        for r in results:
            all_merged.extend(r.get("merged_data", []))

        non_seed = [d for d in all_merged if not d.get("is_seed")]

        if not non_seed:
            print("  データなし")
            continue

        # 品詞別
        pos_stats = defaultdict(list)
        for d in non_seed:
            pos_stats[d["pos"]].append(d["norm_step"])

        print(f"\n  {'品詞':<10} {'平均 norm_step':>14} {'N':>4}")
        print(f"  {'-' * 32}")
        for pos, steps in sorted(
            pos_stats.items(), key=lambda x: np.mean(x[1]),
        ):
            word_type = "内容語" if pos in CONTENT_POS else "機能語"
            print(f"  {pos:<10} {np.mean(steps):14.3f} {len(steps):4d}  ({word_type})")

        # 内容語 vs 機能語
        content = [d["norm_step"] for d in non_seed if d["word_type"] == "内容語"]
        function = [d["norm_step"] for d in non_seed if d["word_type"] == "機能語"]
        if content and function:
            print(f"\n  内容語: {np.mean(content):.3f} (n={len(content)})")
            print(f"  機能語: {np.mean(function):.3f} (n={len(function)})")

        # 3 層
        print(f"\n  {'層':<28} {'平均':>8} {'N':>4}")
        print(f"  {'-' * 44}")
        for layer_name in LAYER_NAMES:
            steps = [
                d["norm_step"] for d in non_seed
                if d.get("layer") == layer_name
            ]
            if steps:
                print(f"  {layer_name:<28} {np.mean(steps):8.3f} {len(steps):4d}")
            else:
                print(f"  {layer_name:<28}        - {'':>4}")

        # Mode A/B: 精度
        if mode in ("A", "B"):
            total_correct = sum(r.get("correct", 0) for r in results)
            total_total = sum(r.get("total", 0) for r in results)
            if total_total > 0:
                print(f"\n  精度: {total_correct}/{total_total} "
                      f"({total_correct / total_total * 100:.1f}%)")

        # Mode C: seed 含有率
        if mode == "C":
            seed_count = sum(1 for r in results if r.get("seed_in_output"))
            print(f"\n  seed 含有率: {seed_count}/{len(results)}")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: 分析・可視化",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="結果ディレクトリ（デフォルト: experiments/04_llada_seed_growth/results/）",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="可視化をスキップ",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        args.results_dir = str(Path(__file__).parent / "results")

    # データ読み込み
    all_results = load_results(args.results_dir)

    # サマリー表示
    print_summary(all_results)

    # CSV 生成
    print(f"\n{'─' * 40}")
    print("CSV 生成")
    print(f"{'─' * 40}")
    generate_pos_summary_csv(all_results, args.results_dir)
    generate_layer_analysis_csv(all_results, args.results_dir)
    generate_exp3_comparison_csv(all_results, args.results_dir)

    # 可視化
    if not args.no_plot:
        print(f"\n{'─' * 40}")
        print("可視化")
        print(f"{'─' * 40}")
        generate_plots(all_results, args.results_dir)

    print("\n分析完了")


if __name__ == "__main__":
    main()
