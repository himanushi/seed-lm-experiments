"""
BPE トークン ↔ 形態素（品詞）アライメント ユーティリティ

LLaDA 等の byte-level BPE トークナイザと fugashi（MeCab）形態素解析の結果を
文字位置ベースで対応付ける。Exp2 / Exp4 で共用。
"""

import fugashi


# 品詞分類定数
CONTENT_POS = {"名詞", "動詞", "形容詞", "副詞"}

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


def align_bpe_with_pos(token_strings):
    """
    BPE トークン文字列列に fugashi の品詞情報を対応付ける

    文字位置の累積でアライメントを取る。
    1 つの形態素が複数 BPE トークンにまたがる場合、
    最初にマッチした形態素の品詞を割り当てる。

    Args:
        token_strings: list[str] - 各 BPE トークンのデコード結果

    Returns:
        list[dict]: [{token_str, pos, word_type}, ...]
    """
    full_text = "".join(token_strings)
    if not full_text.strip():
        return [
            {"token_str": t, "pos": "不明", "word_type": "機能語"}
            for t in token_strings
        ]

    tagger = fugashi.Tagger()
    morphemes = tagger(full_text)

    results = []
    morph_idx = 0
    morph_char_pos = 0
    token_char_pos = 0

    for token_str in token_strings:
        token_char_end = token_char_pos + len(token_str)

        matched_pos = None
        while morph_idx < len(morphemes) and morph_char_pos < token_char_end:
            m = morphemes[morph_idx]
            if matched_pos is None:
                matched_pos = m.feature.pos1
            morph_char_pos += len(m.surface)
            morph_idx += 1

        pos = matched_pos or "不明"
        word_type = "内容語" if pos in CONTENT_POS else "機能語"

        results.append({
            "token_str": token_str,
            "pos": pos,
            "word_type": word_type,
        })
        token_char_pos = token_char_end

    return results


def classify_layer(pos):
    """品詞を 3 層に分類する"""
    for layer_name, pos_set in LAYER_POS.items():
        if pos in pos_set:
            return layer_name
    return "Layer 2: 文法的接続"  # デフォルト（不明な品詞は L2）


def aggregate_by_morpheme(token_data):
    """
    BPE トークン単位のデータを形態素単位に集約する

    同じ形態素に属する複数 BPE トークンの norm_step を平均化する。
    形態素境界は品詞の変化で推定する（近似的）。

    Args:
        token_data: list[dict] - merge_log_with_pos の出力

    Returns:
        list[dict]: 形態素単位に集約されたデータ
    """
    if not token_data:
        return []

    morphemes = []
    current_group = [token_data[0]]

    for d in token_data[1:]:
        prev = current_group[-1]
        # 品詞が変わったら新しい形態素とみなす
        if d["pos"] != prev["pos"] or d.get("is_seed") != prev.get("is_seed"):
            morphemes.append(_merge_group(current_group))
            current_group = [d]
        else:
            current_group.append(d)

    if current_group:
        morphemes.append(_merge_group(current_group))

    return morphemes


def _merge_group(group):
    """BPE トークングループを 1 つの形態素エントリにマージ"""
    surface = "".join(d["token_str"] for d in group)
    avg_norm_step = sum(d["norm_step"] for d in group) / len(group)
    avg_confidence = sum(d["confidence"] for d in group) / len(group)
    return {
        "surface": surface,
        "norm_step": avg_norm_step,
        "confidence": avg_confidence,
        "pos": group[0]["pos"],
        "word_type": group[0]["word_type"],
        "layer": classify_layer(group[0]["pos"]),
        "is_seed": group[0].get("is_seed", False),
        "num_bpe_tokens": len(group),
    }
