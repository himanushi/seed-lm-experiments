"""
逆順マスキング実験: 内容語と機能語、どちらを与えればBERTは残りを補完できるか？

パターンA「内容語→機能語補完」: 名詞・動詞・形容詞・副詞を残し、助詞・助動詞をマスク
パターンB「機能語→内容語補完」: 助詞・助動詞を残し、名詞・動詞・形容詞・副詞をマスク

両パターンの予測精度を比較し、BERTにとってどちらの方向の補完が容易かを検証する。

使い方:
    python experiments/bert_confidence/reverse_masking.py
"""

import argparse
from collections import defaultdict

import torch
import fugashi
from transformers import AutoTokenizer, AutoModelForMaskedLM


SAMPLE_SENTENCES = [
    "タイムモアで挽いたコーヒーが美味しいです。",
    "東京の桜は春に最も美しく咲きます。",
    "彼女は毎朝公園でジョギングをしています。",
    "この本はとても面白かったので友達に勧めました。",
    "雨が降っているから傘を持っていきなさい。",
]

MODEL_ID = "cl-tohoku/bert-base-japanese-whole-word-masking"

# 内容語とみなす品詞
CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞"}


def load_model(device=None):
    """BERTモデルとトークナイザーをロード"""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"モデルをロード中: {MODEL_ID}")
    print(f"  デバイス: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()

    print("✓ ロード完了\n")
    return tokenizer, model, device


def group_subwords(tokens):
    """サブワードトークンを単語単位にグループ化する（##接頭辞で判定）"""
    groups = []
    current_group = []

    for i, token in enumerate(tokens):
        if token in ("[CLS]", "[SEP]"):
            continue
        if token.startswith("##"):
            current_group.append(i)
        else:
            if current_group:
                groups.append(current_group)
            current_group = [i]

    if current_group:
        groups.append(current_group)

    return groups


def align_with_pos(sentence, tokenizer):
    """
    BERT単語グループにfugashiの品詞情報を対応付ける

    文字位置の累積でアライメントを取る。
    BERT側の単語境界とfugashi側の形態素境界が一致しない場合は、
    最初にオーバーラップするfugashi形態素の品詞を採用する。

    Returns:
        list[dict]: 各単語の {group, surface, pos, word_type} のリスト
    """
    encoding = tokenizer(sentence, return_tensors="pt")
    input_ids = encoding["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    word_groups = group_subwords(tokens)

    # BERT単語の表層形を復元
    bert_words = []
    for group in word_groups:
        ids = [input_ids[idx].item() for idx in group]
        # サブワードを結合して表層形を復元
        surface = "".join(
            tokenizer.convert_ids_to_tokens([tid])[0].replace("##", "")
            for tid in ids
        )
        bert_words.append(surface)

    # fugashi形態素解析
    tagger = fugashi.Tagger()
    morphemes = tagger(sentence)

    # 文字位置ベースのアライメント
    results = []
    fug_idx = 0
    fug_char_pos = 0
    bert_char_pos = 0

    for group, bert_surface in zip(word_groups, bert_words):
        bert_char_end = bert_char_pos + len(bert_surface)

        # このBERT単語の範囲にあるfugashi形態素を探す
        matched_pos = None
        while fug_idx < len(morphemes) and fug_char_pos < bert_char_end:
            m = morphemes[fug_idx]
            if matched_pos is None:
                matched_pos = m.feature.pos1
            fug_char_pos += len(m.surface)
            fug_idx += 1

        pos = matched_pos or "不明"
        word_type = "内容語" if pos in CONTENT_POS else "機能語"

        results.append({
            "group": group,
            "surface": bert_surface,
            "pos": pos,
            "word_type": word_type,
        })

        bert_char_pos = bert_char_end

    return results, input_ids


def masked_prediction(input_ids, mask_groups, tokenizer, model, device):
    """
    指定グループをマスクしてBERTに予測させる

    Args:
        input_ids: 元のトークンID列
        mask_groups: マスクする単語グループのリスト（各グループはインデックスのリスト）
        tokenizer, model, device: BERT関連

    Returns:
        list[dict]: 各マスク単語の予測結果
            {surface, predicted, is_correct, confidence, pos, word_type}
    """
    masked_ids = input_ids.clone()
    mask_id = tokenizer.mask_token_id

    # 対象グループをマスク
    for info in mask_groups:
        for idx in info["group"]:
            masked_ids[idx] = mask_id

    # BERT推論
    with torch.no_grad():
        outputs = model(masked_ids.unsqueeze(0).to(device))
        logits = outputs.logits[0].cpu()

    # 各マスク単語の予測を評価
    results = []
    for info in mask_groups:
        group = info["group"]

        # 各サブワードの予測
        predicted_ids = []
        probs_list = []
        for idx in group:
            probs = torch.softmax(logits[idx], dim=-1)
            top_id = probs.argmax(dim=-1).item()
            predicted_ids.append(top_id)
            # 正解トークンの確率
            correct_prob = probs[input_ids[idx]].item()
            probs_list.append(correct_prob)

        # 予測単語の復元
        predicted_surface = "".join(
            tokenizer.convert_ids_to_tokens([tid])[0].replace("##", "")
            for tid in predicted_ids
        )

        # 正解判定（表層形の一致）
        is_correct = predicted_surface == info["surface"]

        # 確信度（正解トークンの確率の幾何平均）
        import math
        if any(p == 0 for p in probs_list):
            confidence = 0.0
        else:
            confidence = math.exp(
                sum(math.log(p) for p in probs_list) / len(probs_list)
            )

        results.append({
            "surface": info["surface"],
            "predicted": predicted_surface,
            "is_correct": is_correct,
            "confidence": confidence,
            "pos": info["pos"],
            "word_type": info["word_type"],
        })

    return results


def run_experiment(sentence, tokenizer, model, device):
    """
    1つの文に対してパターンA・Bの両方を実行する

    Returns:
        (pattern_a_results, pattern_b_results, aligned_words)
    """
    aligned, input_ids = align_with_pos(sentence, tokenizer)

    # 内容語グループと機能語グループに分類
    content_words = [w for w in aligned if w["word_type"] == "内容語"]
    function_words = [w for w in aligned if w["word_type"] == "機能語"]

    # パターンA: 内容語を残す → 機能語をマスクして予測
    pattern_a = []
    if function_words:
        pattern_a = masked_prediction(input_ids, function_words, tokenizer, model, device)

    # パターンB: 機能語を残す → 内容語をマスクして予測
    pattern_b = []
    if content_words:
        pattern_b = masked_prediction(input_ids, content_words, tokenizer, model, device)

    return pattern_a, pattern_b, aligned


def print_sentence_result(sentence, pattern_a, pattern_b, aligned):
    """1文の結果を表示する"""
    print(f"\n{'━' * 70}")
    print(f"入力文: {sentence}")
    print(f"{'━' * 70}")

    # アライメント結果
    print("\n  単語分類:")
    content_str = " ".join(
        f"[{w['surface']}/{w['pos']}]" for w in aligned if w["word_type"] == "内容語"
    )
    function_str = " ".join(
        f"[{w['surface']}/{w['pos']}]" for w in aligned if w["word_type"] == "機能語"
    )
    print(f"    内容語: {content_str}")
    print(f"    機能語: {function_str}")

    # パターンA: 内容語を残す → 機能語を補完
    print(f"\n  パターンA: 内容語を与えて機能語を補完")
    print(f"  {'正解':<10} {'予測':<10} {'確信度':>8} {'判定':>4}")
    print(f"  {'-' * 36}")
    for r in pattern_a:
        mark = "✓" if r["is_correct"] else "✗"
        print(f"  {r['surface']:<10} {r['predicted']:<10} "
              f"{r['confidence']:8.4f} {mark:>4}")

    a_correct = sum(1 for r in pattern_a if r["is_correct"])
    a_total = len(pattern_a)
    a_rate = a_correct / a_total * 100 if a_total > 0 else 0
    print(f"  → 精度: {a_correct}/{a_total} ({a_rate:.1f}%)")

    # パターンB: 機能語を残す → 内容語を補完
    print(f"\n  パターンB: 機能語を与えて内容語を補完")
    print(f"  {'正解':<10} {'予測':<10} {'確信度':>8} {'判定':>4}")
    print(f"  {'-' * 36}")
    for r in pattern_b:
        mark = "✓" if r["is_correct"] else "✗"
        print(f"  {r['surface']:<10} {r['predicted']:<10} "
              f"{r['confidence']:8.4f} {mark:>4}")

    b_correct = sum(1 for r in pattern_b if r["is_correct"])
    b_total = len(pattern_b)
    b_rate = b_correct / b_total * 100 if b_total > 0 else 0
    print(f"  → 精度: {b_correct}/{b_total} ({b_rate:.1f}%)")


def print_summary(all_a, all_b):
    """全文を通じた集計を表示する"""
    print(f"\n{'━' * 70}")
    print("全文集計: 補完方向の比較")
    print(f"{'━' * 70}")

    # パターンA: 機能語の補完精度
    a_correct = sum(1 for r in all_a if r["is_correct"])
    a_total = len(all_a)
    a_rate = a_correct / a_total * 100 if a_total > 0 else 0
    a_avg_conf = sum(r["confidence"] for r in all_a) / a_total if a_total > 0 else 0

    # パターンB: 内容語の補完精度
    b_correct = sum(1 for r in all_b if r["is_correct"])
    b_total = len(all_b)
    b_rate = b_correct / b_total * 100 if b_total > 0 else 0
    b_avg_conf = sum(r["confidence"] for r in all_b) / b_total if b_total > 0 else 0

    print(f"""
  ┌──────────────────────────────────────────────────────────┐
  │  パターン                  正答数   精度     平均確信度  │
  ├──────────────────────────────────────────────────────────┤
  │  A: 内容語→機能語補完    {a_correct:3d}/{a_total:<3d}   {a_rate:5.1f}%   {a_avg_conf:.4f}     │
  │  B: 機能語→内容語補完    {b_correct:3d}/{b_total:<3d}   {b_rate:5.1f}%   {b_avg_conf:.4f}     │
  └──────────────────────────────────────────────────────────┘""")

    # 品詞別の詳細
    print("\n  品詞別精度:")

    # パターンA（機能語の予測）
    print("\n  [パターンA] 機能語の補完精度（内容語が手がかり）")
    pos_stats_a = defaultdict(lambda: {"correct": 0, "total": 0, "conf_sum": 0.0})
    for r in all_a:
        s = pos_stats_a[r["pos"]]
        s["total"] += 1
        s["correct"] += int(r["is_correct"])
        s["conf_sum"] += r["confidence"]

    print(f"    {'品詞':<10} {'正答':>4} {'総数':>4} {'精度':>8} {'平均確信度':>10}")
    print(f"    {'-' * 40}")
    for pos, s in sorted(pos_stats_a.items(), key=lambda x: -x[1]["correct"] / max(x[1]["total"], 1)):
        rate = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        avg_conf = s["conf_sum"] / s["total"] if s["total"] > 0 else 0
        print(f"    {pos:<10} {s['correct']:4d} {s['total']:4d} {rate:7.1f}% {avg_conf:10.4f}")

    # パターンB（内容語の予測）
    print("\n  [パターンB] 内容語の補完精度（機能語が手がかり）")
    pos_stats_b = defaultdict(lambda: {"correct": 0, "total": 0, "conf_sum": 0.0})
    for r in all_b:
        s = pos_stats_b[r["pos"]]
        s["total"] += 1
        s["correct"] += int(r["is_correct"])
        s["conf_sum"] += r["confidence"]

    print(f"    {'品詞':<10} {'正答':>4} {'総数':>4} {'精度':>8} {'平均確信度':>10}")
    print(f"    {'-' * 40}")
    for pos, s in sorted(pos_stats_b.items(), key=lambda x: -x[1]["correct"] / max(x[1]["total"], 1)):
        rate = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        avg_conf = s["conf_sum"] / s["total"] if s["total"] > 0 else 0
        print(f"    {pos:<10} {s['correct']:4d} {s['total']:4d} {rate:7.1f}% {avg_conf:10.4f}")

    # 考察
    print(f"\n  --- 考察 ---")
    if a_rate > b_rate:
        diff = a_rate - b_rate
        print(f"  パターンA（内容語→機能語補完）の精度が {diff:.1f}pt 高い。")
        print(f"  → 内容語が与えられれば、BERTは機能語（助詞・助動詞）を容易に補完できる。")
        print(f"  → 意味の核から構文を生成する「SeedLM」的アプローチと整合する。")
    elif b_rate > a_rate:
        diff = b_rate - a_rate
        print(f"  パターンB（機能語→内容語補完）の精度が {diff:.1f}pt 高い。")
        print(f"  → 構文骨格が与えられれば、BERTは内容語を容易に補完できる。")
        print(f"  → Ford et al. (2018) の機能語優先アプローチと整合する。")
    else:
        print(f"  両パターンの精度は同等。方向性に有意な差は見られない。")

    print(f"\n  補足: パターンAが高精度なら、意味的核（内容語）から出発して")
    print(f"  構文的要素を自動補完させる成長型生成の可能性を示唆する。")


def main():
    parser = argparse.ArgumentParser(
        description="逆順マスキング実験: 内容語 vs 機能語の補完精度比較"
    )
    parser.add_argument(
        "--sentence", "-s", type=str, default=None,
        help="分析する文。指定しない場合は全サンプル文を使用。",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="推論デバイス (mps / cpu)",
    )
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.device)

    sentences = [args.sentence] if args.sentence else SAMPLE_SENTENCES

    all_a = []
    all_b = []

    for sentence in sentences:
        pattern_a, pattern_b, aligned = run_experiment(
            sentence, tokenizer, model, device
        )
        print_sentence_result(sentence, pattern_a, pattern_b, aligned)
        all_a.extend(pattern_a)
        all_b.extend(pattern_b)

    if len(sentences) > 1:
        print_summary(all_a, all_b)


if __name__ == "__main__":
    main()
