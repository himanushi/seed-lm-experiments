"""
反復的機能語補完: 内容語を全て残し、機能語を1語ずつ確信度順にアンマスクする

一斉マスク（reverse_masking.py）では精度21.9%だった機能語補完を、
1語ずつ反復的にアンマスクすることで精度がどこまで向上するかを検証する。

手順:
  1. 内容語を全て残し、機能語を全てマスク
  2. 全マスク位置の予測確信度を計算 → 最高確信度の1語をアンマスク（正解トークンを復元）
  3. 復元後の文で再度予測 → 次の1語をアンマスク
  4. 全機能語が復元されるまで繰り返す
  5. 各ステップの予測正否・確信度・累積精度を記録

使い方:
    python experiments/bert_confidence/iterative_complement.py
"""

import argparse
import math
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
    """サブワードトークンを単語単位にグループ化する"""
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


def decode_surface(token_ids, tokenizer):
    """トークンIDリストからサブワードを結合して表層形を復元する"""
    return "".join(
        tokenizer.convert_ids_to_tokens([tid])[0].replace("##", "")
        for tid in token_ids
    )


def align_with_pos(sentence, tokenizer):
    """BERT単語グループにfugashiの品詞情報を対応付ける"""
    encoding = tokenizer(sentence, return_tensors="pt")
    input_ids = encoding["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    word_groups = group_subwords(tokens)

    bert_words = []
    for group in word_groups:
        ids = [input_ids[idx].item() for idx in group]
        bert_words.append(decode_surface(ids, tokenizer))

    tagger = fugashi.Tagger()
    morphemes = tagger(sentence)

    results = []
    fug_idx = 0
    fug_char_pos = 0
    bert_char_pos = 0

    for group, bert_surface in zip(word_groups, bert_words):
        bert_char_end = bert_char_pos + len(bert_surface)
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


def word_confidence(logits, group_indices, original_ids):
    """単語（サブワードグループ）の正解トークンに対する確信度（幾何平均）"""
    probs_list = []
    for idx in group_indices:
        probs = torch.softmax(logits[idx], dim=-1)
        probs_list.append(probs[original_ids[idx]].item())
    if any(p == 0 for p in probs_list):
        return 0.0
    return math.exp(sum(math.log(p) for p in probs_list) / len(probs_list))


def word_top_prediction(logits, group_indices, tokenizer):
    """単語（サブワードグループ）のtop-1予測を返す"""
    predicted_ids = []
    max_probs = []
    for idx in group_indices:
        probs = torch.softmax(logits[idx], dim=-1)
        top_prob, top_id = probs.max(dim=-1)
        predicted_ids.append(top_id.item())
        max_probs.append(top_prob.item())
    surface = decode_surface(predicted_ids, tokenizer)
    # 予測の確信度は各サブワードのmax probの幾何平均
    if any(p == 0 for p in max_probs):
        confidence = 0.0
    else:
        confidence = math.exp(sum(math.log(p) for p in max_probs) / len(max_probs))
    return surface, confidence


def iterative_complement(sentence, tokenizer, model, device):
    """
    内容語を残し、機能語を1語ずつ確信度順に反復アンマスクする

    Returns:
        steps: 各ステップの情報リスト
        function_count: 機能語の総数
        aligned: アライメント結果
    """
    aligned, input_ids = align_with_pos(sentence, tokenizer)

    # 機能語のインデックス（aligned内での位置）を特定
    function_indices = [
        i for i, w in enumerate(aligned) if w["word_type"] == "機能語"
    ]

    # マスク用IDの準備
    masked_ids = input_ids.clone()
    mask_id = tokenizer.mask_token_id

    # 機能語を全てマスク
    for fi in function_indices:
        for idx in aligned[fi]["group"]:
            masked_ids[idx] = mask_id

    remaining = list(function_indices)  # まだマスクされている機能語
    steps = []

    # 初期状態を記録
    initial_text = tokenizer.decode(masked_ids[1:-1], clean_up_tokenization_spaces=False)
    steps.append({
        "step": 0,
        "surface": "",
        "predicted": "",
        "is_correct": None,
        "confidence": 0.0,
        "pred_confidence": 0.0,
        "pos": "",
        "text": initial_text,
        "cumulative_correct": 0,
        "cumulative_total": 0,
    })

    cumulative_correct = 0

    for step_num in range(1, len(function_indices) + 1):
        # BERT推論
        with torch.no_grad():
            outputs = model(masked_ids.unsqueeze(0).to(device))
            logits = outputs.logits[0].cpu()

        # 残りのマスク単語の中で最高確信度のものを選択
        best_fi = None
        best_conf = -1.0
        for fi in remaining:
            conf = word_confidence(logits, aligned[fi]["group"], input_ids)
            if conf > best_conf:
                best_conf = conf
                best_fi = fi

        info = aligned[best_fi]
        group = info["group"]

        # BERTのtop-1予測を取得
        predicted_surface, pred_conf = word_top_prediction(logits, group, tokenizer)
        is_correct = predicted_surface == info["surface"]
        cumulative_correct += int(is_correct)

        # 正解トークンでアンマスク（oracle mode）
        for idx in group:
            masked_ids[idx] = input_ids[idx]
        remaining.remove(best_fi)

        current_text = tokenizer.decode(
            masked_ids[1:-1], clean_up_tokenization_spaces=False
        )

        steps.append({
            "step": step_num,
            "surface": info["surface"],
            "predicted": predicted_surface,
            "is_correct": is_correct,
            "confidence": best_conf,
            "pred_confidence": pred_conf,
            "pos": info["pos"],
            "text": current_text,
            "cumulative_correct": cumulative_correct,
            "cumulative_total": step_num,
        })

    return steps, len(function_indices), aligned


def oneshot_complement(sentence, tokenizer, model, device):
    """一斉マスク（比較用）: 全機能語を同時にマスクし一括予測"""
    aligned, input_ids = align_with_pos(sentence, tokenizer)
    function_words = [w for w in aligned if w["word_type"] == "機能語"]

    if not function_words:
        return 0, 0

    masked_ids = input_ids.clone()
    mask_id = tokenizer.mask_token_id
    for info in function_words:
        for idx in info["group"]:
            masked_ids[idx] = mask_id

    with torch.no_grad():
        outputs = model(masked_ids.unsqueeze(0).to(device))
        logits = outputs.logits[0].cpu()

    correct = 0
    for info in function_words:
        predicted, _ = word_top_prediction(logits, info["group"], tokenizer)
        if predicted == info["surface"]:
            correct += 1

    return correct, len(function_words)


def print_sentence_result(sentence, steps, aligned):
    """1文の結果を表示する"""
    print(f"\n{'━' * 70}")
    print(f"入力文: {sentence}")

    # 単語分類を表示
    content = " ".join(f"{w['surface']}" for w in aligned if w["word_type"] == "内容語")
    function = " ".join(f"{w['surface']}" for w in aligned if w["word_type"] == "機能語")
    print(f"  内容語（残す）: {content}")
    print(f"  機能語（マスク）: {function}")
    print(f"{'━' * 70}")

    # ステップごとの表示
    print(f"\n  {'Step':>4}  {'テキスト'}")
    print(f"  {'':>4}  {'正解':<8} {'予測':<8} {'確信度':>8} {'判定':>4} {'累積精度':>10}")
    print(f"  {'-' * 60}")

    for s in steps:
        step = s["step"]
        if step == 0:
            print(f"  {step:4d}  {s['text']}")
            continue

        mark = "✓" if s["is_correct"] else "✗"
        cum_rate = s["cumulative_correct"] / s["cumulative_total"] * 100
        print(f"  {step:4d}  {s['text']}")
        print(f"        {s['surface']:<8} {s['predicted']:<8} "
              f"{s['confidence']:8.4f} {mark:>4} "
              f"{s['cumulative_correct']}/{s['cumulative_total']} ({cum_rate:5.1f}%)")

    # 最終精度
    final = steps[-1]
    total = final["cumulative_total"]
    correct = final["cumulative_correct"]
    rate = correct / total * 100 if total > 0 else 0
    print(f"\n  最終精度: {correct}/{total} ({rate:.1f}%)")


def print_comparison(all_iterative, all_oneshot):
    """反復 vs 一斉の比較表を出力する"""
    print(f"\n{'━' * 70}")
    print("比較: 反復的アンマスク vs 一斉マスク")
    print(f"{'━' * 70}")

    # 文ごとの比較
    print(f"\n  {'文':<30} {'反復的':>12} {'一斉':>12} {'差分':>8}")
    print(f"  {'-' * 64}")

    iter_total_correct = 0
    iter_total_count = 0
    one_total_correct = 0
    one_total_count = 0

    for (sentence, i_correct, i_total), (o_correct, o_total) in zip(
        all_iterative, all_oneshot
    ):
        i_rate = i_correct / i_total * 100 if i_total > 0 else 0
        o_rate = o_correct / o_total * 100 if o_total > 0 else 0
        diff = i_rate - o_rate

        # 文を短縮表示
        short = sentence[:28] + "…" if len(sentence) > 28 else sentence
        print(f"  {short:<30} {i_correct}/{i_total} ({i_rate:5.1f}%) "
              f"{o_correct}/{o_total} ({o_rate:5.1f}%) {diff:+6.1f}pt")

        iter_total_correct += i_correct
        iter_total_count += i_total
        one_total_correct += o_correct
        one_total_count += o_total

    # 全体
    iter_rate = iter_total_correct / iter_total_count * 100 if iter_total_count > 0 else 0
    one_rate = one_total_correct / one_total_count * 100 if one_total_count > 0 else 0
    diff_total = iter_rate - one_rate

    print(f"  {'-' * 64}")
    print(f"  {'全体':<30} "
          f"{iter_total_correct}/{iter_total_count} ({iter_rate:5.1f}%) "
          f"{one_total_correct}/{one_total_count} ({one_rate:5.1f}%) "
          f"{diff_total:+6.1f}pt")

    print(f"""
  ┌────────────────────────────────────────────────┐
  │  方式              正答数     精度     改善幅   │
  ├────────────────────────────────────────────────┤
  │  一斉マスク       {one_total_correct:3d}/{one_total_count:<3d}    {one_rate:5.1f}%   baseline │
  │  反復的アンマスク  {iter_total_correct:3d}/{iter_total_count:<3d}    {iter_rate:5.1f}%   {diff_total:+.1f}pt    │
  └────────────────────────────────────────────────┘""")

    # 品詞別精度（反復的）
    print("\n  反復的アンマスクの品詞別精度:")
    # all_iterative には (sentence, correct, total) だけでは足りないので、
    # この関数の呼び出し元でstepsも渡す必要がある → 別の仕組みで
    # → print_pos_detail() で対応


def print_pos_detail(all_steps_flat):
    """品詞別の詳細精度を表示する"""
    print(f"\n  反復的アンマスクの品詞別精度:")
    pos_stats = defaultdict(lambda: {"correct": 0, "total": 0, "conf_sum": 0.0})
    for s in all_steps_flat:
        if s["step"] == 0:
            continue
        st = pos_stats[s["pos"]]
        st["total"] += 1
        st["correct"] += int(s["is_correct"])
        st["conf_sum"] += s["confidence"]

    print(f"    {'品詞':<10} {'正答':>4} {'総数':>4} {'精度':>8} {'平均確信度':>10}")
    print(f"    {'-' * 40}")
    for pos, st in sorted(
        pos_stats.items(), key=lambda x: -x[1]["correct"] / max(x[1]["total"], 1)
    ):
        rate = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
        avg_conf = st["conf_sum"] / st["total"] if st["total"] > 0 else 0
        print(f"    {pos:<10} {st['correct']:4d} {st['total']:4d} {rate:7.1f}% {avg_conf:10.4f}")


def print_step_progression(all_steps_by_sentence):
    """ステップごとの精度推移（全文平均）を表示する"""
    print(f"\n  ステップごとの累積精度推移:")
    print(f"    （各文を正規化: step/total_steps → 0%～100%の進捗で表示）")

    # 各文のステップを進捗率に正規化して集計
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for steps in all_steps_by_sentence:
        real_steps = [s for s in steps if s["step"] > 0]
        n = len(real_steps)
        for s in real_steps:
            # 進捗率を25%刻みでバケット化
            progress = s["step"] / n
            if progress <= 0.25:
                bucket = "0-25%"
            elif progress <= 0.50:
                bucket = "26-50%"
            elif progress <= 0.75:
                bucket = "51-75%"
            else:
                bucket = "76-100%"
            b = buckets[bucket]
            b["total"] += 1
            b["correct"] += int(s["is_correct"])

    print(f"    {'進捗':>10} {'正答':>4} {'総数':>4} {'精度':>8}")
    print(f"    {'-' * 30}")
    for bucket in ["0-25%", "26-50%", "51-75%", "76-100%"]:
        b = buckets[bucket]
        rate = b["correct"] / b["total"] * 100 if b["total"] > 0 else 0
        print(f"    {bucket:>10} {b['correct']:4d} {b['total']:4d} {rate:7.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="反復的機能語補完: 1語ずつ確信度順にアンマスク"
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

    # 反復的アンマスクの結果
    all_iterative = []       # [(sentence, correct, total)]
    all_steps_flat = []      # 全ステップをフラットに
    all_steps_by_sentence = []  # 文ごとのステップリスト

    # 一斉マスクの結果（比較用）
    all_oneshot = []         # [(correct, total)]

    for sentence in sentences:
        # 反復的アンマスク
        steps, func_count, aligned = iterative_complement(
            sentence, tokenizer, model, device
        )
        print_sentence_result(sentence, steps, aligned)

        final = steps[-1]
        all_iterative.append((sentence, final["cumulative_correct"], final["cumulative_total"]))
        all_steps_flat.extend(steps)
        all_steps_by_sentence.append(steps)

        # 一斉マスク（比較用）
        o_correct, o_total = oneshot_complement(sentence, tokenizer, model, device)
        all_oneshot.append((o_correct, o_total))

    # 全体比較
    if len(sentences) > 1:
        print_comparison(all_iterative, all_oneshot)
        print_pos_detail(all_steps_flat)
        print_step_progression(all_steps_by_sentence)


if __name__ == "__main__":
    main()
