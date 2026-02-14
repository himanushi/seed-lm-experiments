"""
Experiment 3: Seedからの段階的テキスト成長

Mode A: 目標文ガイド付き成長（主要・分析用）
  - 目標文 + seed語を入力
  - seed以外を全マスク → 確信度順に1語ずつ復元
  - Exp1のiterative_unmaskと同じ手法だが、seed語が最初から見えている点が異なる
  - seedがあることで復元順序がどう変わるかをExp1と比較

Mode B: 自由成長（探索用）
  - seed語のみを入力（目標文なし）
  - [CLS] seed [SEP] → 隣接位置に[MASK]挿入 → BERT予測 → 確信度閾値で受理 → 繰り返し

使い方:
    # Mode A
    python experiments/03_prompt_growth/grow.py --target "コーヒーが美味しいです。"
    python experiments/03_prompt_growth/grow.py --target "コーヒーが美味しいです。" --seed "美味しい"

    # Mode B
    python experiments/03_prompt_growth/grow.py --free --seed "美味しい"

    # 全サンプル一括
    python experiments/03_prompt_growth/grow.py --all-samples

    # Exp1との比較付き
    python experiments/03_prompt_growth/grow.py --target "タイムモアで挽いたコーヒーが美味しいです。" --compare-exp1
"""

import argparse
import math
from collections import defaultdict

import torch
import fugashi
from transformers import AutoTokenizer, AutoModelForMaskedLM


# --- 定数 ---

SAMPLE_SENTENCES = [
    ("タイムモアで挽いたコーヒーが美味しいです。", "美味しい"),
    ("東京の桜は春に最も美しく咲きます。", "咲き"),
    ("彼女は毎朝公園でジョギングをしています。", "ジョギング"),
    ("この本はとても面白かったので友達に勧めました。", "面白かった"),
    ("雨が降っているから傘を持っていきなさい。", "降って"),
]

MODEL_ID = "cl-tohoku/bert-base-japanese-whole-word-masking"
CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞"}

# 3層分類マップ（Exp1cの知見に基づく）
LAYER_NAMES = [
    "Layer 1: 意味の核",
    "Layer 2: 文法的接続",
    "Layer 3: 修飾・文体",
]
LAYER_POS = {
    "Layer 1: 意味の核": {"名詞", "動詞"},
    "Layer 2: 文法的接続": {"助詞", "接続詞"},
    "Layer 3: 修飾・文体": {"助動詞", "形容詞", "副詞", "記号", "補助記号"},
}


# ============================================================
# ユーティリティ（Exp1から移植）
# ============================================================

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
    """単語の正解トークンに対する確信度（幾何平均）"""
    probs_list = []
    for idx in group_indices:
        probs = torch.softmax(logits[idx], dim=-1)
        probs_list.append(probs[original_ids[idx]].item())
    if any(p == 0 for p in probs_list):
        return 0.0
    return math.exp(sum(math.log(p) for p in probs_list) / len(probs_list))


def word_top_prediction(logits, group_indices, tokenizer):
    """単語のtop-1予測を返す"""
    predicted_ids = []
    max_probs = []
    for idx in group_indices:
        probs = torch.softmax(logits[idx], dim=-1)
        top_prob, top_id = probs.max(dim=-1)
        predicted_ids.append(top_id.item())
        max_probs.append(top_prob.item())
    surface = decode_surface(predicted_ids, tokenizer)
    if any(p == 0 for p in max_probs):
        confidence = 0.0
    else:
        confidence = math.exp(sum(math.log(p) for p in max_probs) / len(max_probs))
    return surface, confidence


# ============================================================
# Seed選択
# ============================================================

def find_seed_in_tokens(seed_text, aligned):
    """
    seed語のBERTトークン位置を特定する

    aligned内の表層形でseed_textを完全一致→部分一致で検索し、
    一致する単語グループのインデックスを返す。
    """
    # 完全一致を優先
    for i, w in enumerate(aligned):
        if w["surface"] == seed_text:
            return [i]

    # 部分一致（seedが複数単語にまたがる場合）
    surfaces = [w["surface"] for w in aligned]
    concat = ""
    char_to_word = {}
    for wi, s in enumerate(surfaces):
        for c in s:
            char_to_word[len(concat)] = wi
            concat += c

    start = concat.find(seed_text)
    if start >= 0:
        end = start + len(seed_text)
        word_indices = sorted(set(
            char_to_word[pos] for pos in range(start, end)
            if pos in char_to_word
        ))
        return word_indices

    return None


def auto_select_seed(aligned):
    """
    文末述語を自動選択する（日本語SOV構造を利用）

    後ろから探して最初に見つかった動詞・形容詞を返す。
    見つからなければ最後の内容語を返す。
    """
    predicate_pos = {"動詞", "形容詞"}

    for i in range(len(aligned) - 1, -1, -1):
        if aligned[i]["pos"] in predicate_pos:
            return [i]

    for i in range(len(aligned) - 1, -1, -1):
        if aligned[i]["word_type"] == "内容語":
            return [i]

    return [0]


# ============================================================
# Mode A: 目標文ガイド付き成長
# ============================================================

def guided_growth(sentence, seed_text, tokenizer, model, device):
    """
    目標文ガイド付き成長: seed以外を全マスク→確信度順に1語ずつ復元

    Exp1のiterative_unmaskと似ているが、seed語が最初から見えている点が異なる。
    正解トークンで復元（oracle mode）。

    Returns:
        steps: 各ステップの情報リスト
        aligned: アライメント結果
        seed_indices: seed語の位置
    """
    aligned, input_ids = align_with_pos(sentence, tokenizer)

    # seed位置の特定
    if seed_text:
        seed_indices = find_seed_in_tokens(seed_text, aligned)
        if seed_indices is None:
            print(f"  警告: seed「{seed_text}」が文中に見つかりません。自動選択に切り替えます。")
            seed_indices = auto_select_seed(aligned)
    else:
        seed_indices = auto_select_seed(aligned)

    seed_surfaces = "".join(aligned[i]["surface"] for i in seed_indices)
    print(f"  seed語: 「{seed_surfaces}」 (位置: {seed_indices})")

    # seed以外を全マスク
    masked_ids = input_ids.clone()
    mask_id = tokenizer.mask_token_id

    remaining = []
    for i, w in enumerate(aligned):
        if i not in seed_indices:
            for idx in w["group"]:
                masked_ids[idx] = mask_id
            remaining.append(i)

    total_to_unmask = len(remaining)

    steps = []

    # 初期状態（seedのみ可視）
    initial_text = tokenizer.decode(
        masked_ids[1:-1], clean_up_tokenization_spaces=False
    )
    steps.append({
        "step": 0,
        "surface": seed_surfaces,
        "predicted": "",
        "is_correct": None,
        "confidence": 0.0,
        "pos": "",
        "word_type": "",
        "text": initial_text,
        "is_seed": True,
    })

    for step_num in range(1, total_to_unmask + 1):
        # BERT推論
        with torch.no_grad():
            outputs = model(masked_ids.unsqueeze(0).to(device))
            logits = outputs.logits[0].cpu()

        # 残存マスク語の確信度を計算
        best_wi = None
        best_conf = -1.0
        for wi in remaining:
            conf = word_confidence(logits, aligned[wi]["group"], input_ids)
            if conf > best_conf:
                best_conf = conf
                best_wi = wi

        info = aligned[best_wi]
        group = info["group"]

        # top-1予測を取得
        predicted_surface, _ = word_top_prediction(logits, group, tokenizer)
        is_correct = predicted_surface == info["surface"]

        # 正解トークンでアンマスク（oracle mode）
        for idx in group:
            masked_ids[idx] = input_ids[idx]
        remaining.remove(best_wi)

        current_text = tokenizer.decode(
            masked_ids[1:-1], clean_up_tokenization_spaces=False
        )

        steps.append({
            "step": step_num,
            "surface": info["surface"],
            "predicted": predicted_surface,
            "is_correct": is_correct,
            "confidence": best_conf,
            "pos": info["pos"],
            "word_type": info["word_type"],
            "text": current_text,
            "is_seed": False,
        })

    return steps, aligned, seed_indices


# ============================================================
# Mode B: 自由成長
# ============================================================

def insert_masks(token_ids, mask_id):
    """
    全トークンの隣接位置に[MASK]を挿入する

    [CLS] t1 t2 t3 [SEP]
    → [CLS] [MASK] t1 [MASK] t2 [MASK] t3 [MASK] [SEP]

    Returns:
        new_ids: [MASK]挿入済みのトークンID（tensor）
        mask_positions: [MASK]の位置リスト
    """
    ids = token_ids.tolist()
    new_ids = [ids[0]]  # [CLS]
    mask_positions = []

    for i in range(1, len(ids) - 1):  # [CLS]と[SEP]の間
        # 元のトークンの前に[MASK]を挿入
        new_ids.append(mask_id)
        mask_positions.append(len(new_ids) - 1)
        new_ids.append(ids[i])

    # 最後のトークンの後にも[MASK]
    new_ids.append(mask_id)
    mask_positions.append(len(new_ids) - 1)

    new_ids.append(ids[-1])  # [SEP]

    return torch.tensor(new_ids), mask_positions


def free_growth(seed_text, tokenizer, model, device, threshold=0.3, max_steps=15):
    """
    自由成長: seed語のみから[MASK]挿入→予測→受理を繰り返す

    1. [CLS] seed [SEP] から開始
    2. 全隣接位置に[MASK]を挿入
    3. BERTで予測し、最も確信度の高い予測を受理（閾値以上の場合）
    4. 受理されたトークンを追加して2に戻る
    5. 閾値未満 or max_steps到達 or 反復検出で停止

    Returns:
        steps: 各ステップの情報リスト
    """
    # seedをトークナイズ
    seed_encoding = tokenizer(seed_text, return_tensors="pt")
    seed_ids = seed_encoding["input_ids"][0]  # [CLS] seed_tokens [SEP]

    current_ids = seed_ids.clone()
    mask_id = tokenizer.mask_token_id

    # 特殊トークンID（除外用）
    special_ids = set(tokenizer.all_special_ids)

    steps = []
    initial_text = tokenizer.decode(
        current_ids[1:-1], clean_up_tokenization_spaces=False
    )
    steps.append({
        "step": 0,
        "text": initial_text,
        "added_token": "",
        "confidence": 0.0,
    })

    seen_texts = {initial_text}
    recent_tokens = []  # 直近の追加トークンを記録（反復検出用）

    for step in range(1, max_steps + 1):
        # 全隣接位置に[MASK]を挿入
        expanded_ids, mask_positions = insert_masks(current_ids, mask_id)

        # BERT推論
        with torch.no_grad():
            outputs = model(expanded_ids.unsqueeze(0).to(device))
            logits = outputs.logits[0].cpu()

        # 各[MASK]位置の予測を評価
        candidates = []
        for mpos in mask_positions:
            probs = torch.softmax(logits[mpos], dim=-1)
            top_prob, top_id = probs.max(dim=-1)

            # 特殊トークンは除外
            if top_id.item() in special_ids:
                continue

            candidates.append({
                "position": mpos,
                "token_id": top_id.item(),
                "token": tokenizer.convert_ids_to_tokens([top_id.item()])[0],
                "confidence": top_prob.item(),
            })

        if not candidates:
            steps.append({
                "step": step,
                "text": steps[-1]["text"],
                "added_token": "",
                "confidence": 0.0,
                "stop_reason": "候補なし",
            })
            break

        # 確信度順にソート
        candidates.sort(key=lambda x: -x["confidence"])
        best = candidates[0]

        # 閾値チェック
        if best["confidence"] < threshold:
            steps.append({
                "step": step,
                "text": steps[-1]["text"],
                "added_token": "",
                "confidence": best["confidence"],
                "stop_reason": f"閾値未満 (最大確信度: {best['confidence']:.4f})",
            })
            break

        # 反復トークン検出（直近2回と同じトークンなら停止）
        if len(recent_tokens) >= 2 and all(
            t == best["token_id"] for t in recent_tokens[-2:]
        ):
            steps.append({
                "step": step,
                "text": steps[-1]["text"],
                "added_token": best["token"],
                "confidence": best["confidence"],
                "stop_reason": f"反復トークン検出 (「{best['token']}」が3回連続)",
            })
            break

        # 新しいトークン列を構築（bestの位置だけ予測トークンに、他の[MASK]は除去）
        new_ids_list = []
        for i, tid in enumerate(expanded_ids.tolist()):
            if i == best["position"]:
                new_ids_list.append(best["token_id"])
            elif tid == mask_id:
                continue  # 他の[MASK]は除去
            else:
                new_ids_list.append(tid)

        current_ids = torch.tensor(new_ids_list)
        current_text = tokenizer.decode(
            current_ids[1:-1], clean_up_tokenization_spaces=False
        )

        # テキスト全体の反復検出
        if current_text in seen_texts:
            steps.append({
                "step": step,
                "text": current_text,
                "added_token": best["token"],
                "confidence": best["confidence"],
                "stop_reason": "反復検出",
            })
            break

        seen_texts.add(current_text)
        recent_tokens.append(best["token_id"])

        steps.append({
            "step": step,
            "text": current_text,
            "added_token": best["token"],
            "confidence": best["confidence"],
        })

    else:
        # max_steps到達（forが自然終了した場合）
        if steps and "stop_reason" not in steps[-1]:
            steps[-1]["stop_reason"] = "最大ステップ到達"

    return steps


# ============================================================
# 分析
# ============================================================

def classify_layer(pos):
    """品詞を3層に分類する"""
    for layer_name, pos_set in LAYER_POS.items():
        if pos in pos_set:
            return layer_name
    return "Layer 2: 文法的接続"  # デフォルト


def analyze_growth_order(steps, total_words):
    """
    各ステップの品詞を3層に分類し、正規化ステップを計算する

    Returns:
        layer_stats: 各層の統計情報 dict
    """
    layer_stats = defaultdict(lambda: {
        "words": [],
        "steps": [],
        "norm_steps": [],
        "correct": 0,
        "total": 0,
    })

    for s in steps:
        if s["step"] == 0 or s.get("is_seed"):
            continue

        layer = classify_layer(s["pos"])
        norm_step = s["step"] / total_words

        stats = layer_stats[layer]
        stats["words"].append(s["surface"])
        stats["steps"].append(s["step"])
        stats["norm_steps"].append(norm_step)
        stats["total"] += 1
        if s["is_correct"]:
            stats["correct"] += 1

    return dict(layer_stats)


def compare_with_exp1(sentence, seed_text, tokenizer, model, device):
    """
    同じ文でExp1（全マスク復元）とExp3（seed付き復元）を比較する

    Returns:
        exp1_steps: Exp1スタイルの復元ステップ
        exp3_steps: Exp3のseed付き復元ステップ
        comparison: 比較結果 dict
    """
    aligned, input_ids = align_with_pos(sentence, tokenizer)
    total_words = len(aligned)

    # --- Exp1: 全マスク復元（analyze.pyのiterative_unmaskと同等） ---
    masked_ids = input_ids.clone()
    mask_id = tokenizer.mask_token_id
    for w in aligned:
        for idx in w["group"]:
            masked_ids[idx] = mask_id

    remaining = list(range(total_words))
    exp1_steps = []

    for step_num in range(1, total_words + 1):
        with torch.no_grad():
            outputs = model(masked_ids.unsqueeze(0).to(device))
            logits = outputs.logits[0].cpu()

        best_wi = None
        best_conf = -1.0
        for wi in remaining:
            conf = word_confidence(logits, aligned[wi]["group"], input_ids)
            if conf > best_conf:
                best_conf = conf
                best_wi = wi

        info = aligned[best_wi]
        predicted, _ = word_top_prediction(logits, info["group"], tokenizer)

        for idx in info["group"]:
            masked_ids[idx] = input_ids[idx]
        remaining.remove(best_wi)

        exp1_steps.append({
            "step": step_num,
            "surface": info["surface"],
            "pos": info["pos"],
            "word_type": info["word_type"],
            "confidence": best_conf,
            "is_correct": predicted == info["surface"],
            "norm_step": step_num / total_words,
        })

    # --- Exp3: seed付き復元 ---
    exp3_steps_raw, _, seed_indices = guided_growth(
        sentence, seed_text, tokenizer, model, device
    )

    total_non_seed = total_words - len(seed_indices)
    exp3_steps = []
    for s in exp3_steps_raw:
        if s["step"] == 0:
            continue
        exp3_steps.append({
            "step": s["step"],
            "surface": s["surface"],
            "pos": s["pos"],
            "word_type": s["word_type"],
            "confidence": s["confidence"],
            "is_correct": s["is_correct"],
            "norm_step": s["step"] / total_non_seed if total_non_seed > 0 else 0,
        })

    # --- 比較: 品詞ごと・タイプごとの平均正規化ステップ ---
    def avg_by_key(steps_list, key):
        groups = defaultdict(list)
        for s in steps_list:
            groups[s[key]].append(s["norm_step"])
        return {k: sum(v) / len(v) for k, v in groups.items()}

    comparison = {
        "exp1_by_pos": avg_by_key(exp1_steps, "pos"),
        "exp3_by_pos": avg_by_key(exp3_steps, "pos"),
        "exp1_by_type": avg_by_key(exp1_steps, "word_type"),
        "exp3_by_type": avg_by_key(exp3_steps, "word_type"),
        "exp1_correct": sum(1 for s in exp1_steps if s["is_correct"]),
        "exp3_correct": sum(1 for s in exp3_steps if s["is_correct"]),
        "exp1_total": len(exp1_steps),
        "exp3_total": len(exp3_steps),
    }

    return exp1_steps, exp3_steps, comparison


# ============================================================
# 表示
# ============================================================

def print_growth_process(steps, title="テキスト成長過程"):
    """成長過程のステップ表示"""
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")

    for s in steps:
        step = s["step"]
        text = s["text"]

        if step == 0:
            label = " [seed]" if s.get("is_seed") else ""
            print(f"\n  step {step:2d}: {text}{label}")
        else:
            print(f"  step {step:2d}: {text}")

            # Mode A: 詳細情報あり
            if "predicted" in s and s.get("is_correct") is not None:
                mark = "✓" if s["is_correct"] else "✗"
                print(
                    f"           → 「{s['surface']}」"
                    f" ({s['pos']}/{s['word_type']})"
                    f" 確信度: {s['confidence']:.4f}"
                    f" [予測: 「{s['predicted']}」 {mark}]"
                )
            # Mode B: 追加トークン情報
            elif s.get("added_token"):
                print(
                    f"           → 追加: 「{s['added_token']}」"
                    f" 確信度: {s['confidence']:.4f}"
                )

            if "stop_reason" in s:
                print(f"           ★ 停止: {s['stop_reason']}")


def print_layer_analysis(layer_stats, seed_surfaces=""):
    """3層仮説の検証結果を表示"""
    print(f"\n{'=' * 65}")
    print(f"  3層モデル検証")
    print(f"{'=' * 65}")

    if seed_surfaces:
        print(f"\n  seed: 「{seed_surfaces}」")

    print(f"\n  {'層':<28} {'単語数':>6} {'平均norm_step':>13} {'精度':>8}")
    print(f"  {'-' * 60}")

    avgs = []
    for layer in LAYER_NAMES:
        if layer in layer_stats:
            stats = layer_stats[layer]
            avg_norm = sum(stats["norm_steps"]) / len(stats["norm_steps"])
            rate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            words = ", ".join(stats["words"][:5])
            if len(stats["words"]) > 5:
                words += "..."
            print(f"  {layer:<28} {stats['total']:6d} {avg_norm:13.3f} {rate:7.1f}%")
            print(f"    語例: {words}")
            avgs.append(avg_norm)
        else:
            print(f"  {layer:<28}      0             -        -")
            avgs.append(None)

    # 3層仮説との整合性判定
    print(f"\n  --- 3層仮説との整合性 ---")
    if avgs[0] is not None and avgs[2] is not None:
        if avgs[0] < avgs[2]:
            print(f"  ✓ Layer 1（意味の核）→ Layer 3（修飾・文体）の順序が確認された")
            if avgs[1] is not None and avgs[0] < avgs[1] < avgs[2]:
                print(
                    f"  ✓ 3層全てが期待通りの順序:"
                    f" L1({avgs[0]:.3f}) < L2({avgs[1]:.3f}) < L3({avgs[2]:.3f})"
                )
            elif avgs[1] is not None:
                print(
                    f"  △ Layer 2の位置が期待と異なる:"
                    f" L1({avgs[0]:.3f}), L2({avgs[1]:.3f}), L3({avgs[2]:.3f})"
                )
        else:
            print(f"  ✗ 順序が逆: L1({avgs[0]:.3f}) > L3({avgs[2]:.3f})")
    else:
        print(f"  - データ不足のため判定不能")


def print_comparison_result(sentence, comparison, seed_text):
    """Exp1とExp3の比較結果を表示"""
    print(f"\n{'=' * 65}")
    print(f"  Exp1 vs Exp3 比較")
    print(f"  文: {sentence}")
    print(f"  seed: 「{seed_text}」")
    print(f"{'=' * 65}")

    # 精度比較
    e1_total = comparison["exp1_total"]
    e3_total = comparison["exp3_total"]
    e1_rate = comparison["exp1_correct"] / e1_total * 100 if e1_total > 0 else 0
    e3_rate = comparison["exp3_correct"] / e3_total * 100 if e3_total > 0 else 0

    print(f"\n  予測精度:")
    print(
        f"    Exp1（全マスク）:"
        f" {comparison['exp1_correct']}/{e1_total} ({e1_rate:.1f}%)"
    )
    print(
        f"    Exp3（seed付き）:"
        f" {comparison['exp3_correct']}/{e3_total} ({e3_rate:.1f}%)"
    )
    diff = e3_rate - e1_rate
    print(f"    差分: {diff:+.1f}pt")

    # 品詞別の比較
    all_pos = sorted(
        set(list(comparison["exp1_by_pos"]) + list(comparison["exp3_by_pos"]))
    )

    if all_pos:
        print(f"\n  品詞別 平均正規化ステップ (0=最初, 1=最後):")
        print(f"    {'品詞':<8} {'Exp1':>8} {'Exp3':>8} {'差分':>8}")
        print(f"    {'-' * 36}")
        for pos in all_pos:
            e1 = comparison["exp1_by_pos"].get(pos)
            e3 = comparison["exp3_by_pos"].get(pos)
            e1_str = f"{e1:8.3f}" if e1 is not None else "       -"
            e3_str = f"{e3:8.3f}" if e3 is not None else "       -"
            if e1 is not None and e3 is not None:
                d_str = f"{e3 - e1:+8.3f}"
            else:
                d_str = "       -"
            print(f"    {pos:<8} {e1_str} {e3_str} {d_str}")

    # 内容語・機能語の比較
    print(f"\n  内容語 vs 機能語:")
    for wtype in ["内容語", "機能語"]:
        e1 = comparison["exp1_by_type"].get(wtype)
        e3 = comparison["exp3_by_type"].get(wtype)
        if e1 is not None and e3 is not None:
            print(f"    {wtype}: Exp1={e1:.3f}, Exp3={e3:.3f} ({e3 - e1:+.3f})")
        elif e1 is not None:
            print(f"    {wtype}: Exp1={e1:.3f}, Exp3=- (seedにより除外)")


# ============================================================
# メイン
# ============================================================

def run_mode_a_single(sentence, seed_text, tokenizer, model, device, compare_exp1_flag):
    """Mode A: 単一文の処理"""
    steps, aligned, seed_indices = guided_growth(
        sentence, seed_text, tokenizer, model, device
    )
    print_growth_process(steps, title="seed付き成長（Mode A）")

    # 精度
    real_steps = [s for s in steps if s["step"] > 0]
    correct = sum(1 for s in real_steps if s["is_correct"])
    total = len(real_steps)
    if total > 0:
        print(f"\n  精度: {correct}/{total} ({correct / total * 100:.1f}%)")

    # 3層分析
    layer_stats = analyze_growth_order(steps, len(aligned))
    seed_surfaces = "".join(aligned[i]["surface"] for i in seed_indices)
    print_layer_analysis(layer_stats, seed_surfaces=seed_surfaces)

    # Exp1との比較
    if compare_exp1_flag:
        _, _, comparison = compare_with_exp1(
            sentence, seed_text, tokenizer, model, device
        )
        print_comparison_result(sentence, comparison, seed_text)

    return steps, aligned, seed_indices


def run_mode_a_all(args, tokenizer, model, device):
    """Mode A: 全サンプル一括処理"""
    all_layer_stats = defaultdict(lambda: {
        "words": [], "steps": [], "norm_steps": [],
        "correct": 0, "total": 0,
    })
    all_correct = 0
    all_total = 0

    for sentence, default_seed in SAMPLE_SENTENCES:
        seed_text = args.seed or default_seed

        print(f"\n{'━' * 65}")
        print(f"  入力文: {sentence}")

        steps, aligned, seed_indices = guided_growth(
            sentence, seed_text, tokenizer, model, device
        )
        print_growth_process(steps, title="seed付き成長（Mode A）")

        # 精度集計
        real_steps = [s for s in steps if s["step"] > 0]
        correct = sum(1 for s in real_steps if s["is_correct"])
        total = len(real_steps)
        all_correct += correct
        all_total += total
        if total > 0:
            print(f"\n  精度: {correct}/{total} ({correct / total * 100:.1f}%)")

        # 3層分析
        layer_stats = analyze_growth_order(steps, len(aligned))
        seed_surfaces = "".join(aligned[i]["surface"] for i in seed_indices)
        print_layer_analysis(layer_stats, seed_surfaces=seed_surfaces)

        # 全体集計にマージ
        for layer, stats in layer_stats.items():
            a = all_layer_stats[layer]
            a["words"].extend(stats["words"])
            a["steps"].extend(stats["steps"])
            a["norm_steps"].extend(stats["norm_steps"])
            a["correct"] += stats["correct"]
            a["total"] += stats["total"]

    # --- 全体サマリー ---
    print(f"\n{'━' * 65}")
    print(f"  全サンプル集計")
    print(f"{'━' * 65}")
    if all_total > 0:
        print(f"\n  全体精度: {all_correct}/{all_total}"
              f" ({all_correct / all_total * 100:.1f}%)")
    print_layer_analysis(dict(all_layer_stats), seed_surfaces="（全サンプル集計）")

    # Exp1との比較（オプション）
    if args.compare_exp1:
        print(f"\n{'━' * 65}")
        print(f"  Exp1との比較（全サンプル）")
        print(f"{'━' * 65}")

        for sentence, default_seed in SAMPLE_SENTENCES:
            seed_text = args.seed or default_seed
            _, _, comparison = compare_with_exp1(
                sentence, seed_text, tokenizer, model, device
            )
            print_comparison_result(sentence, comparison, seed_text)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Seedからの段階的テキスト成長"
    )
    parser.add_argument(
        "--target", "-t", type=str, default=None,
        help="目標文（Mode A）",
    )
    parser.add_argument(
        "--seed", "-s", type=str, default=None,
        help="seed語。省略時は自動選択（文末述語）。",
    )
    parser.add_argument(
        "--free", action="store_true",
        help="Mode B: 自由成長（目標文なし）",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Mode Bの確信度閾値（デフォルト: 0.3）",
    )
    parser.add_argument(
        "--max-steps", type=int, default=15,
        help="Mode Bの最大ステップ数（デフォルト: 15）",
    )
    parser.add_argument(
        "--all-samples", action="store_true",
        help="全サンプル文でMode Aを実行",
    )
    parser.add_argument(
        "--compare-exp1", action="store_true",
        help="Exp1（全マスク復元）との比較を追加",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="推論デバイス (mps / cpu)",
    )
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.device)

    if args.free:
        # --- Mode B: 自由成長 ---
        if not args.seed:
            print("エラー: Mode Bでは --seed が必要です。")
            return

        print(f"\n{'━' * 65}")
        print(f"  Mode B: 自由成長")
        print(f"  seed: 「{args.seed}」  閾値: {args.threshold}"
              f"  最大ステップ: {args.max_steps}")
        print(f"{'━' * 65}")

        steps = free_growth(
            args.seed, tokenizer, model, device,
            threshold=args.threshold, max_steps=args.max_steps,
        )
        print_growth_process(steps, title="自由成長過程（Mode B）")

    elif args.all_samples:
        # --- 全サンプルでMode A ---
        run_mode_a_all(args, tokenizer, model, device)

    else:
        # --- Mode A: 単一文 ---
        if args.target:
            sentence = args.target
            seed_text = args.seed  # Noneの場合は自動選択
        else:
            sentence, default_seed = SAMPLE_SENTENCES[0]
            seed_text = args.seed or default_seed

        print(f"\n{'━' * 65}")
        print(f"  入力文: {sentence}")

        run_mode_a_single(
            sentence, seed_text, tokenizer, model, device, args.compare_exp1
        )


if __name__ == "__main__":
    main()
