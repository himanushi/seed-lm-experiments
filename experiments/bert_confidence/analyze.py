"""
BERT確信度分析: 全マスク→確信度順アンマスキングによるテキスト成長

文を全マスクし、BERTの予測確信度が高い位置（単語）から順にアンマスクしていく。
これにより、BERTが「意味的に重要な語」と「構文的な語」のどちらを先に復元するかを観察する。

使い方:
    python experiments/bert_confidence/analyze.py
    python experiments/bert_confidence/analyze.py --sentence "東京の桜は春に美しく咲きます。"
"""

import argparse
import sys
from pathlib import Path

import torch
import fugashi
from transformers import AutoTokenizer, AutoModelForMaskedLM


# サンプル文（日本語の多様な構造を含む）
SAMPLE_SENTENCES = [
    "タイムモアで挽いたコーヒーが美味しいです。",
    "東京の桜は春に最も美しく咲きます。",
    "彼女は毎朝公園でジョギングをしています。",
    "この本はとても面白かったので友達に勧めました。",
    "雨が降っているから傘を持っていきなさい。",
]

MODEL_ID = "cl-tohoku/bert-base-japanese-whole-word-masking"


def load_model(device=None):
    """BERTモデルとトークナイザーをロード"""
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"モデルをロード中: {MODEL_ID}")
    print(f"  デバイス: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()

    print("✓ ロード完了")
    return tokenizer, model, device


def group_subwords(tokens):
    """
    サブワードトークンを単語単位にグループ化する

    BERTの日本語トークナイザーは "美味しい" → ["美味", "##しい"] のように分割する。
    ##で始まるトークンは直前の単語の続きとして扱う。

    Returns:
        list[list[int]]: 各単語に属するトークンインデックスのリスト
        （[CLS], [SEP] は除外済み）
    """
    groups = []
    current_group = []

    for i, token in enumerate(tokens):
        # [CLS], [SEP] はスキップ
        if token in ("[CLS]", "[SEP]"):
            continue

        if token.startswith("##"):
            # サブワード: 現在のグループに追加
            current_group.append(i)
        else:
            # 新しい単語
            if current_group:
                groups.append(current_group)
            current_group = [i]

    if current_group:
        groups.append(current_group)

    return groups


def get_word_confidence(logits, word_indices, original_ids):
    """
    単語（サブワードグループ）の確信度を計算する

    各サブワードの予測確率の幾何平均を使用。
    """
    log_probs = []
    for idx in word_indices:
        probs = torch.softmax(logits[idx], dim=-1)
        # 正解トークンの確率
        correct_prob = probs[original_ids[idx]].item()
        log_probs.append(correct_prob)

    # 幾何平均（対数空間で平均）
    import math
    if any(p == 0 for p in log_probs):
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in log_probs) / len(log_probs))
    return geo_mean


def iterative_unmask(sentence, tokenizer, model, device):
    """
    全マスク状態から確信度順に単語を1つずつアンマスクしていく

    Args:
        sentence: 入力文
        tokenizer: BERTトークナイザー
        model: BERTモデル
        device: 推論デバイス

    Returns:
        steps: 各ステップの情報リスト
            [(step, word_surface, confidence, predicted_word, is_correct, current_text)]
    """
    # トークナイズ
    encoding = tokenizer(sentence, return_tensors="pt")
    input_ids = encoding["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    # 単語グループの構築
    word_groups = group_subwords(tokens)
    total_words = len(word_groups)

    # 元のIDを保存
    original_ids = input_ids.clone()

    # 全マスク化
    masked_ids = input_ids.clone()
    mask_id = tokenizer.mask_token_id
    for group in word_groups:
        for idx in group:
            masked_ids[idx] = mask_id

    # 未アンマスクの単語グループインデックス
    remaining = list(range(total_words))

    steps = []

    # 初期状態（全マスク）
    initial_text = tokenizer.decode(masked_ids[1:-1], clean_up_tokenization_spaces=False)
    steps.append({
        "step": 0,
        "word": "",
        "confidence": 0.0,
        "predicted": "",
        "is_correct": None,
        "text": initial_text,
    })

    for step_num in range(1, total_words + 1):
        # BERT推論
        with torch.no_grad():
            outputs = model(masked_ids.unsqueeze(0).to(device))
            logits = outputs.logits[0].cpu()

        # 各残存単語の確信度を計算
        best_group_idx = None
        best_confidence = -1.0

        for gi in remaining:
            conf = get_word_confidence(logits, word_groups[gi], original_ids)
            if conf > best_confidence:
                best_confidence = conf
                best_group_idx = gi

        # 最も確信度の高い単語をアンマスク（元のトークンを復元）
        group = word_groups[best_group_idx]

        # BERTの予測トークンを取得（比較用）
        predicted_ids = []
        for idx in group:
            probs = torch.softmax(logits[idx], dim=-1)
            predicted_ids.append(probs.argmax(dim=-1).item())

        predicted_word = tokenizer.decode(predicted_ids)
        original_word = tokenizer.decode([original_ids[idx] for idx in group])

        # 元のトークンで復元
        for idx in group:
            masked_ids[idx] = original_ids[idx]

        remaining.remove(best_group_idx)

        current_text = tokenizer.decode(masked_ids[1:-1], clean_up_tokenization_spaces=False)

        steps.append({
            "step": step_num,
            "word": original_word,
            "confidence": best_confidence,
            "predicted": predicted_word,
            "is_correct": predicted_word.strip() == original_word.strip(),
            "text": current_text,
        })

    return steps


def analyze_pos_order(sentence, steps):
    """
    アンマスク順序を品詞で分析する

    Returns:
        list[(step, word, pos1, pos_detail)]
    """
    tagger = fugashi.Tagger()
    morphemes = tagger(sentence)

    # fugashi形態素 → 品詞マップを構築
    word_pos = {}
    for m in morphemes:
        word_pos[m.surface] = {
            "pos1": m.feature.pos1,  # 大分類（名詞、動詞、助詞...）
            "pos2": m.feature.pos2 if m.feature.pos2 else "",
        }

    pos_order = []
    for s in steps[1:]:  # step 0（全マスク）はスキップ
        word = s["word"].strip()
        pos_info = word_pos.get(word, {"pos1": "不明", "pos2": ""})

        # 品詞の大分類
        pos1 = pos_info["pos1"]

        # 内容語 vs 機能語の分類
        content_pos = {"名詞", "動詞", "形容詞", "副詞"}
        word_type = "内容語" if pos1 in content_pos else "機能語"

        pos_order.append({
            "step": s["step"],
            "word": word,
            "pos": pos1,
            "type": word_type,
            "confidence": s["confidence"],
            "is_correct": s["is_correct"],
        })

    return pos_order


def print_growth(steps):
    """成長過程をきれいに表示する"""
    print("\n" + "=" * 60)
    print("テキスト成長過程（BERT確信度順アンマスク）")
    print("=" * 60)

    for s in steps:
        step = s["step"]
        text = s["text"]
        if step == 0:
            print(f"\n  step {step:2d}: {text}")
        else:
            word = s["word"]
            conf = s["confidence"]
            pred = s["predicted"]
            mark = "✓" if s["is_correct"] else "✗"
            print(f"  step {step:2d}: {text}")
            print(f"           → 復元: 「{word}」 (確信度: {conf:.4f}) "
                  f"[予測: 「{pred}」 {mark}]")


def print_pos_analysis(pos_order):
    """品詞分析結果を表示する"""
    print("\n" + "=" * 60)
    print("品詞別アンマスク順序分析")
    print("=" * 60)

    print(f"\n  {'Step':>4}  {'単語':<10} {'品詞':<8} {'分類':<6} {'確信度':>8}  {'予測':>4}")
    print("  " + "-" * 52)

    for p in pos_order:
        mark = "✓" if p["is_correct"] else "✗"
        print(f"  {p['step']:4d}  {p['word']:<10} {p['pos']:<8} {p['type']:<6} "
              f"{p['confidence']:8.4f}  {mark:>4}")

    # 統計サマリー
    content_steps = [p["step"] for p in pos_order if p["type"] == "内容語"]
    function_steps = [p["step"] for p in pos_order if p["type"] == "機能語"]

    print("\n  --- サマリー ---")
    if content_steps:
        print(f"  内容語の平均ステップ: {sum(content_steps)/len(content_steps):.1f} "
              f"(n={len(content_steps)})")
    if function_steps:
        print(f"  機能語の平均ステップ: {sum(function_steps)/len(function_steps):.1f} "
              f"(n={len(function_steps)})")

    if content_steps and function_steps:
        avg_content = sum(content_steps) / len(content_steps)
        avg_function = sum(function_steps) / len(function_steps)
        if avg_content < avg_function:
            print("  → BERTは内容語を先にアンマスクする傾向")
        else:
            print("  → BERTは機能語を先にアンマスクする傾向")


def main():
    parser = argparse.ArgumentParser(
        description="BERT確信度分析: 全マスク→確信度順アンマスキング"
    )
    parser.add_argument(
        "--sentence", "-s",
        type=str,
        default=None,
        help="分析する文。指定しない場合はサンプル文を使用。",
    )
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="全サンプル文で実行する。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推論デバイス (mps / cpu)",
    )
    args = parser.parse_args()

    # モデルロード
    tokenizer, model, device = load_model(args.device)

    # 対象文の決定
    if args.sentence:
        sentences = [args.sentence]
    elif args.all_samples:
        sentences = SAMPLE_SENTENCES
    else:
        sentences = [SAMPLE_SENTENCES[0]]

    # 全文での集計用
    all_pos_orders = []

    for sentence in sentences:
        print(f"\n{'━' * 60}")
        print(f"入力文: {sentence}")
        print(f"{'━' * 60}")

        # 反復的アンマスク
        steps = iterative_unmask(sentence, tokenizer, model, device)
        print_growth(steps)

        # 品詞分析
        pos_order = analyze_pos_order(sentence, steps)
        print_pos_analysis(pos_order)

        all_pos_orders.extend(pos_order)

    # 全文を通じた集計（複数文の場合）
    if len(sentences) > 1:
        print(f"\n{'━' * 60}")
        print("全文を通じた集計")
        print(f"{'━' * 60}")

        content_steps = [p["step"] for p in all_pos_orders if p["type"] == "内容語"]
        function_steps = [p["step"] for p in all_pos_orders if p["type"] == "機能語"]

        correct = sum(1 for p in all_pos_orders if p["is_correct"])
        total = len(all_pos_orders)

        print(f"\n  全体の予測正答率: {correct}/{total} ({correct/total*100:.1f}%)")
        if content_steps:
            print(f"  内容語の平均ステップ: {sum(content_steps)/len(content_steps):.1f}")
        if function_steps:
            print(f"  機能語の平均ステップ: {sum(function_steps)/len(function_steps):.1f}")

        # 品詞ごとの平均ステップ
        from collections import defaultdict
        pos_steps = defaultdict(list)
        for p in all_pos_orders:
            pos_steps[p["pos"]].append(p["step"])

        print("\n  品詞別平均ステップ（早い順）:")
        sorted_pos = sorted(pos_steps.items(), key=lambda x: sum(x[1]) / len(x[1]))
        for pos, steps_list in sorted_pos:
            avg = sum(steps_list) / len(steps_list)
            print(f"    {pos:<8}: {avg:.1f} (n={len(steps_list)})")


if __name__ == "__main__":
    main()
