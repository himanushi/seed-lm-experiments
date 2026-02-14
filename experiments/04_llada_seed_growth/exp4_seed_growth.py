"""
Experiment 4: LLaDA による seed 付き段階的テキスト成長

Exp3 で BERT が失敗した「seed 付き成長」を LLaDA（マスク拡散型 LLM）で再検証する。
3 つのモードで段階的に難易度を上げる構成。

Mode A: ガイド付き成長（Exp3 直接比較用）
  - 目標文 + seed → seed 以外を全マスク → LLaDA 拡散で復元
  - 復元精度と品詞別アンマスク順序を Exp3（BERT）と比較

Mode B: 部分ガイド（長さのみ既知）
  - 目標文の長さ + seed の位置のみ → LLaDA 拡散で生成
  - Mode A と同じ初期状態だが、生成の自由度を観察

Mode C: 自由成長（Exp3 Mode B 対応）
  - seed 語のみ → プロンプトで指示 → 拡散で生成
  - BERT で崩壊した自由成長が LLaDA で機能するかを検証

実行環境: Google Colab (A100 GPU)

使い方:
    # セットアップ
    !pip install transformers==4.38.2 accelerate fugashi unidic-lite

    # Mode A: 全サンプル
    python experiments/04_llada_seed_growth/exp4_seed_growth.py --mode a --all

    # Mode B: 全サンプル
    python experiments/04_llada_seed_growth/exp4_seed_growth.py --mode b --all

    # Mode C: 全サンプル
    python experiments/04_llada_seed_growth/exp4_seed_growth.py --mode c --all

    # 全モード一括
    python experiments/04_llada_seed_growth/exp4_seed_growth.py --all-modes

    # 単一文（Mode A）
    python experiments/04_llada_seed_growth/exp4_seed_growth.py --mode a \\
        --target "コーヒーが美味しいです。" --seed "美味しい"
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# プロジェクトルートをパスに追加
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from utils.pos_alignment import (
    align_bpe_with_pos,
    classify_layer,
    CONTENT_POS,
    LAYER_NAMES,
    LAYER_POS,
)


# ============================================================
# 定数
# ============================================================

MASK_ID = 126336   # LLaDA の [MASK] トークン ID
EOT_ID = 126081    # LLaDA の EOT トークン ID

SAMPLE_SENTENCES = [
    {"text": "タイムモアで挽いたコーヒーが美味しいです。", "seed": "美味しい"},
    {"text": "東京の桜は春に最も美しく咲きます。", "seed": "桜"},
    {"text": "彼女は毎朝公園でジョギングをしています。", "seed": "ジョギング"},
    {"text": "この本はとても面白かったので友達に勧めました。", "seed": "面白かった"},
    {"text": "雨が降っているから傘を持っていきなさい。", "seed": "傘"},
]


def select_model_id():
    """GPU メモリに基づいてモデルを選択"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU が検出されません。Google Colab で GPU ランタイムを選択してください。"
        )
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    if gpu_mem >= 30:
        model_id = "GSAI-ML/LLaDA-8B-Instruct"
        print(f"GPU メモリ {gpu_mem:.0f}GB → LLaDA-8B を使用")
    else:
        model_id = "inclusionAI/LLaDA-MoE-7B-A1B-Instruct"
        print(f"GPU メモリ {gpu_mem:.0f}GB → LLaDA-MoE を使用（省メモリ版）")
    return model_id


def load_model(model_id=None):
    """モデルとトークナイザをロード"""
    if model_id is None:
        model_id = select_model_id()
    print(f"\nモデルをロード中: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    print(f"ロード完了 (dtype: {model.dtype})")
    print(f"  MASK token ID: {MASK_ID}")
    print(f"  vocab size: {tokenizer.vocab_size}\n")
    return tokenizer, model


# ============================================================
# LLaDA 拡散生成（Exp2 から移植 + seed 対応）
# ============================================================

def add_gumbel_noise(logits, temperature):
    """Gumbel-max サンプリング用のノイズ付与（LLaDA 公式準拠）"""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """各ステップでアンマスクするトークン数を事前計算（線形スケジュール）"""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64,
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_seed_diffusion(
    model, initial_ids, fixed_positions, steps,
    temperature=0., remasking="low_confidence", mask_id=MASK_ID,
):
    """
    seed 位置を固定した LLaDA 拡散生成（Mode A / Mode B 用）

    initial_ids 内の fixed_positions は固定（seed として扱い、再マスクしない）。
    残りの MASK 位置を拡散ステップで順次アンマスクする。

    Args:
        model: LLaDA モデル
        initial_ids: 初期トークン ID 列 (1D tensor, length = seq_len)
                     seed 位置は正しいトークン、それ以外は mask_id
        fixed_positions: set[int] - 固定位置（seed トークン）
        steps: 拡散ステップ数
        temperature: サンプリング温度（0 = 決定的）
        remasking: リマスク戦略
        mask_id: MASK トークン ID

    Returns:
        final_ids: 最終トークン ID 列 (1D tensor, CPU)
        unmask_log: list[dict] - 各アンマスクイベント
        step_snapshots: list[list[int]] - 各ステップ後のトークン ID 列
    """
    x = initial_ids.clone().unsqueeze(0).to(model.device)  # (1, seq_len)
    seq_len = x.shape[1]

    # 固定位置マーク（seed は再マスクしない）
    prompt_index = torch.zeros(1, seq_len, dtype=torch.bool, device=model.device)
    for pos in fixed_positions:
        prompt_index[0, pos] = True

    # 転送スケジュール計算
    mask_index = (x == mask_id)
    num_masks = mask_index.sum().item()
    if num_masks == 0:
        return x[0].cpu(), [], [x[0].cpu().tolist()]

    # ステップ数がマスク数より多い場合は調整
    effective_steps = min(steps, num_masks)
    num_transfer_tokens = get_num_transfer_tokens(mask_index, effective_steps)

    unmask_log = []
    step_snapshots = [x[0].cpu().tolist()]

    for i in range(effective_steps):
        mask_index = (x == mask_id)
        if not mask_index.any():
            break

        # モデル推論
        logits = model(x).logits

        # サンプリング
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        # 確信度計算
        if remasking == "low_confidence":
            p = F.softmax(logits.float(), dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1,
            )
        elif remasking == "random":
            x0_p = torch.rand(x0.shape, device=x0.device)
        else:
            raise ValueError(f"Unknown remasking: {remasking}")

        # マスク位置のみを対象にする
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -torch.inf)

        # top-k 選択（確信度の高い位置から）
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        k = num_transfer_tokens[0, i].item()
        if k > 0:
            _, select_index = torch.topk(confidence[0], k=k)
            transfer_index[0, select_index] = True

        # ログ記録
        newly_unmasked = transfer_index[0].nonzero(as_tuple=True)[0]
        for pos in newly_unmasked:
            pos_int = pos.item()
            unmask_log.append({
                "step": i + 1,
                "position": pos_int,
                "token_id": x0[0, pos_int].item(),
                "confidence": x0_p[0, pos_int].item(),
            })

        # アンマスク実行
        x[transfer_index] = x0[transfer_index]
        step_snapshots.append(x[0].cpu().tolist())

    return x[0].cpu(), unmask_log, step_snapshots


@torch.no_grad()
def generate_with_prompt_logging(
    model, prompt, steps=64, gen_length=64, block_length=64,
    temperature=0., cfg_scale=0., remasking="low_confidence",
    mask_id=MASK_ID,
):
    """
    プロンプト付き LLaDA 生成（Mode C 用、Exp2 準拠）

    Returns:
        x: 生成結果のトークン ID 列 (1, prompt_len + gen_length)
        unmask_log: list[dict] - 生成領域のアンマスクイベント
        prompt_len: int - プロンプトの長さ
        step_snapshots: list[list[int]] - 各ステップの生成領域スナップショット
    """
    prompt_len = prompt.shape[1]
    x = torch.full(
        (1, prompt_len + gen_length), mask_id, dtype=torch.long, device=model.device,
    )
    x[:, :prompt_len] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    unmask_log = []
    step_snapshots = []
    global_step = 0

    for num_block in range(num_blocks):
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index, steps_per_block,
        )

        for i in range(steps_per_block):
            global_step += 1
            mask_index = (x == mask_id)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1,
                )
            elif remasking == "random":
                x0_p = torch.rand(x0.shape, device=x0.device)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -torch.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, i].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True

            newly_unmasked = transfer_index[0].nonzero(as_tuple=True)[0]
            for pos in newly_unmasked:
                pos_int = pos.item()
                if pos_int >= prompt_len:
                    unmask_log.append({
                        "step": global_step,
                        "position": pos_int,
                        "gen_position": pos_int - prompt_len,
                        "token_id": x0[0, pos_int].item(),
                        "confidence": x0_p[0, pos_int].item(),
                    })

            x[transfer_index] = x0[transfer_index]
            step_snapshots.append(x[0, prompt_len:].cpu().tolist())

    return x, unmask_log, prompt_len, step_snapshots


# ============================================================
# Seed 位置特定
# ============================================================

def find_seed_positions(target_text, seed_text, tokenizer):
    """
    target_text 中の seed_text に対応する BPE トークン位置を特定する

    文字位置ベースでアライメントを取る。

    Returns:
        list[int]: seed に対応するトークンインデックス（0-indexed）
        None: seed が見つからない場合
    """
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)

    # 各トークンの文字位置範囲を計算
    token_strings = [tokenizer.decode([tid]) for tid in target_ids]
    char_pos = 0
    token_char_ranges = []
    for ts in token_strings:
        token_char_ranges.append((char_pos, char_pos + len(ts)))
        char_pos += len(ts)

    # 再構成テキスト上で seed を検索
    reconstructed = "".join(token_strings)
    seed_start = reconstructed.find(seed_text)

    if seed_start < 0:
        # 元テキスト上の位置で近似
        seed_start_orig = target_text.find(seed_text)
        if seed_start_orig < 0:
            return None
        seed_start = seed_start_orig

    seed_end = seed_start + len(seed_text)

    # 重なるトークンを特定
    seed_positions = []
    for i, (start, end) in enumerate(token_char_ranges):
        if start < seed_end and end > seed_start:
            seed_positions.append(i)

    return seed_positions if seed_positions else None


# ============================================================
# トークン抽出・分析
# ============================================================

def extract_valid_tokens(token_ids, tokenizer, mask_id=MASK_ID):
    """
    トークン ID 列から有効なトークン（特殊トークン除く）を抽出

    最初の特殊トークンで打ち切る。

    Returns:
        list[dict]: [{position, token_id, token_str}, ...]
    """
    special_ids = {mask_id, EOT_ID}
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    tokens = []
    for i, tid in enumerate(token_ids):
        if tid in special_ids:
            break
        token_str = tokenizer.decode([tid])
        tokens.append({
            "position": i,
            "token_id": tid,
            "token_str": token_str,
        })
    return tokens


def merge_log_with_pos(unmask_log, aligned_tokens, seed_positions=None):
    """
    アンマスクログと品詞情報を結合する

    Returns:
        list[dict]: 各トークンのアンマスク情報 + 品詞情報
    """
    log_map = {entry["position"]: entry for entry in unmask_log}
    max_step = max((e["step"] for e in unmask_log), default=1)
    seed_pos_set = set(seed_positions) if seed_positions else set()

    merged = []
    for t in aligned_tokens:
        pos_idx = t["position"]
        log_entry = log_map.get(pos_idx)

        if pos_idx in seed_pos_set:
            merged.append({
                "step": 0,
                "norm_step": 0.0,
                "position": pos_idx,
                "token_str": t["token_str"],
                "confidence": 1.0,
                "pos": t.get("pos", "不明"),
                "word_type": t.get("word_type", "不明"),
                "layer": classify_layer(t.get("pos", "不明")),
                "is_seed": True,
            })
        elif log_entry:
            merged.append({
                "step": log_entry["step"],
                "norm_step": log_entry["step"] / max_step,
                "position": pos_idx,
                "token_str": t["token_str"],
                "confidence": log_entry["confidence"],
                "pos": t.get("pos", "不明"),
                "word_type": t.get("word_type", "不明"),
                "layer": classify_layer(t.get("pos", "不明")),
                "is_seed": False,
            })

    return merged


# ============================================================
# Mode A: ガイド付き成長
# ============================================================

def run_mode_a_single(target_text, seed_text, model, tokenizer, steps=None):
    """
    Mode A: 目標文 + seed → LLaDA 拡散で復元

    目標文のトークン列で seed 以外を全マスクし、拡散ステップで復元。
    生成結果と目標文のトークン一致率を計算する。

    Returns:
        dict: 実験結果（merged_data, accuracy 等）
    """
    print(f"\n  目標文: {target_text}")
    print(f"  seed: 「{seed_text}」")

    # 1. トークン化
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    target_ids_tensor = torch.tensor(target_ids, dtype=torch.long)
    num_tokens = len(target_ids)
    print(f"  トークン数: {num_tokens}")

    # 2. seed 位置特定
    seed_positions = find_seed_positions(target_text, seed_text, tokenizer)
    if seed_positions is None:
        print(f"  seed「{seed_text}」が見つかりません。スキップします。")
        return None

    seed_strs = [tokenizer.decode([target_ids[p]]) for p in seed_positions]
    print(f"  seed 位置: {seed_positions} → 「{''.join(seed_strs)}」")

    # 3. 初期状態: seed 以外を全マスク
    initial_ids = torch.full((num_tokens,), MASK_ID, dtype=torch.long)
    for pos in seed_positions:
        initial_ids[pos] = target_ids_tensor[pos]

    # 4. 拡散ステップ数
    num_masks = num_tokens - len(seed_positions)
    if steps is None:
        steps = num_masks  # マスク数と同じ（≒ 1 トークン/ステップ）
    print(f"  マスク数: {num_masks}, 拡散ステップ: {steps}")

    # 5. 拡散実行
    final_ids, unmask_log, snapshots = generate_seed_diffusion(
        model, initial_ids, set(seed_positions), steps,
        temperature=0., mask_id=MASK_ID,
    )

    # 6. 結果テキスト
    generated_text = tokenizer.decode(final_ids.tolist(), skip_special_tokens=True)
    print(f"  生成結果: {generated_text}")

    # 7. 精度計算
    correct = 0
    total = 0
    for i in range(num_tokens):
        if i not in set(seed_positions):
            total += 1
            if final_ids[i].item() == target_ids[i]:
                correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f"  精度: {correct}/{total} ({accuracy * 100:.1f}%)")

    # 8. 品詞アライメント
    all_token_strs = [tokenizer.decode([final_ids[i].item()]) for i in range(num_tokens)]
    aligned = align_bpe_with_pos(all_token_strs)
    for i, a in enumerate(aligned):
        a["position"] = i
        a["token_id"] = final_ids[i].item()

    # 9. ログと品詞をマージ
    merged = merge_log_with_pos(unmask_log, aligned, seed_positions)

    return {
        "mode": "A",
        "target_text": target_text,
        "seed_text": seed_text,
        "generated_text": generated_text,
        "num_tokens": num_tokens,
        "seed_positions": seed_positions,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "merged_data": merged,
        "step_snapshots": snapshots,
        "target_ids": target_ids,
    }


# ============================================================
# Mode B: 部分ガイド
# ============================================================

def run_mode_b_single(target_text, seed_text, model, tokenizer, steps=None):
    """
    Mode B: 長さ + seed 位置のみ → LLaDA 拡散で生成

    Mode A と同じ初期状態・同じ生成プロセス。
    ただし分析の焦点は「目標文との一致」ではなく
    「生成テキストの自然さとアンマスク順序」。

    Returns:
        dict: 実験結果
    """
    # Mode A と同じ生成を実行
    result = run_mode_a_single(target_text, seed_text, model, tokenizer, steps)
    if result is not None:
        result["mode"] = "B"
    return result


# ============================================================
# Mode C: 自由成長
# ============================================================

def run_mode_c_single(seed_text, model, tokenizer, gen_length=64, steps=64):
    """
    Mode C: seed 語のみ → プロンプトで指示 → LLaDA 拡散で生成

    BERT（Exp3 Mode B）で崩壊した自由成長が LLaDA で機能するかを検証。
    拡散ステップの中間状態を全記録し、どの順でトークンが確定するかを観察。

    Returns:
        dict: 実験結果
    """
    print(f"\n  seed: 「{seed_text}」")

    prompt_text = (
        f"「{seed_text}」という語を使って、"
        f"自然な日本語の短い文を1つ書いてください。"
    )
    print(f"  プロンプト: {prompt_text}")

    # chat template でフォーマット
    messages = [{"role": "user", "content": prompt_text}]
    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )
    else:
        text = (
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

    prompt = input_ids.to(model.device)
    prompt_len = prompt.shape[1]
    print(f"  プロンプト長: {prompt_len} tokens")
    print(f"  生成長: {gen_length} tokens, ステップ: {steps}")

    # 生成
    x, unmask_log, _, step_snapshots = generate_with_prompt_logging(
        model, prompt,
        steps=steps, gen_length=gen_length, block_length=gen_length,
        temperature=0., cfg_scale=0., mask_id=MASK_ID,
    )

    # 有効トークン抽出（生成領域のみ）
    gen_ids = x[0, prompt_len:].cpu()
    content_tokens = extract_valid_tokens(gen_ids, tokenizer, mask_id=MASK_ID)

    # position を生成領域基準に調整
    for t in content_tokens:
        t["gen_position"] = t["position"]
        t["position"] = t["position"] + prompt_len

    generated_text = tokenizer.decode(
        x[0, prompt_len:].tolist(), skip_special_tokens=True,
    ).strip()
    print(f"  生成結果: {generated_text}")
    print(f"  有効トークン数: {len(content_tokens)}")

    # seed が含まれているか
    seed_in_output = seed_text in generated_text
    print(f"  seed 含有: {'Yes' if seed_in_output else 'No'}")

    # 品詞アライメント
    token_strs = [t["token_str"] for t in content_tokens]
    aligned = align_bpe_with_pos(token_strs)
    for i, a in enumerate(aligned):
        a.update(content_tokens[i])

    # ログと品詞をマージ
    log_map = {e["position"]: e for e in unmask_log}
    max_step = max((e["step"] for e in unmask_log), default=1)

    merged = []
    for t in aligned:
        log_entry = log_map.get(t["position"])
        if log_entry:
            merged.append({
                "step": log_entry["step"],
                "norm_step": log_entry["step"] / max_step,
                "position": t["position"],
                "gen_position": t.get("gen_position", 0),
                "token_str": t["token_str"],
                "confidence": log_entry["confidence"],
                "pos": t["pos"],
                "word_type": t["word_type"],
                "layer": classify_layer(t["pos"]),
                "is_seed": False,
            })

    return {
        "mode": "C",
        "seed_text": seed_text,
        "prompt_text": prompt_text,
        "generated_text": generated_text,
        "seed_in_output": seed_in_output,
        "token_count": len(content_tokens),
        "merged_data": merged,
        "step_snapshots": step_snapshots,
    }


# ============================================================
# 表示・分析
# ============================================================

def print_growth_steps(result, tokenizer, max_display=25):
    """アンマスク順序のステップ表示"""
    mode = result["mode"]
    merged = result.get("merged_data", [])
    if not merged:
        print("  （データなし）")
        return

    print(f"\n  {'─' * 60}")
    print(f"  アンマスク順序 (Mode {mode})")
    print(f"  {'─' * 60}")

    sorted_data = sorted(merged, key=lambda d: d["step"])
    print(f"\n  {'Step':>4} {'Norm':>5} {'トークン':<12} {'品詞':<6} "
          f"{'層':<20} {'確信度':>8}")
    print(f"  {'-' * 60}")

    for d in sorted_data[:max_display]:
        seed_mark = " [seed]" if d.get("is_seed") else ""
        print(
            f"  {d['step']:4d} {d['norm_step']:5.2f} "
            f"{d['token_str']:<12} {d['pos']:<6} "
            f"{d['layer']:<20} {d['confidence']:8.4f}{seed_mark}"
        )
    if len(sorted_data) > max_display:
        print(f"  ... (残り {len(sorted_data) - max_display} トークン)")


def print_step_visualization(result, tokenizer):
    """段階的成長のステップ可視化（Exp3 形式）"""
    snapshots = result.get("step_snapshots", [])
    if not snapshots:
        return

    mode = result["mode"]
    print(f"\n  {'─' * 60}")
    print(f"  成長可視化 (Mode {mode})")
    print(f"  {'─' * 60}\n")

    total_steps = len(snapshots)
    if total_steps <= 15:
        display_indices = list(range(total_steps))
    else:
        # 初期・中間・最終を含めて ~15 ステップに間引く
        display_indices = sorted(set(
            [0]
            + list(range(1, total_steps - 1, max(1, (total_steps - 2) // 13)))
            + [total_steps - 1]
        ))

    for idx in display_indices:
        snapshot = snapshots[idx]
        tokens = []
        for tid in snapshot:
            if tid == MASK_ID:
                tokens.append("[M]")
            else:
                tokens.append(tokenizer.decode([tid]))
        text = " ".join(tokens)
        label = " [seed]" if idx == 0 else ""
        print(f"  step {idx:3d}: {text}{label}")


def analyze_layers(merged_data):
    """3 層モデルの検証"""
    layer_stats = defaultdict(lambda: {
        "tokens": [], "norm_steps": [], "count": 0,
    })

    for d in merged_data:
        if d.get("is_seed"):
            continue
        layer = d["layer"]
        layer_stats[layer]["tokens"].append(d["token_str"])
        layer_stats[layer]["norm_steps"].append(d["norm_step"])
        layer_stats[layer]["count"] += 1

    return dict(layer_stats)


def print_layer_analysis(layer_stats):
    """3 層分析結果の表示"""
    print(f"\n  {'─' * 55}")
    print(f"  3 層モデル検証")
    print(f"  {'─' * 55}")

    print(f"\n  {'層':<28} {'N':>4} {'平均 norm_step':>14}")
    print(f"  {'-' * 50}")

    avgs = []
    for layer in LAYER_NAMES:
        if layer in layer_stats:
            stats = layer_stats[layer]
            avg = np.mean(stats["norm_steps"])
            tokens_preview = ", ".join(stats["tokens"][:5])
            if len(stats["tokens"]) > 5:
                tokens_preview += "..."
            print(f"  {layer:<28} {stats['count']:4d} {avg:14.3f}")
            print(f"    例: {tokens_preview}")
            avgs.append(avg)
        else:
            print(f"  {layer:<28}    0              -")
            avgs.append(None)

    # 順序判定
    if avgs[0] is not None and avgs[2] is not None:
        if avgs[0] < avgs[2]:
            print(f"\n  L1({avgs[0]:.3f}) < L3({avgs[2]:.3f})")
            if avgs[1] is not None and avgs[0] < avgs[1] < avgs[2]:
                print(f"  → 完全順序: L1 < L2 < L3")
            elif avgs[1] is not None:
                print(f"  → 部分順序: L2={avgs[1]:.3f}")
        else:
            print(f"\n  L1({avgs[0]:.3f}) >= L3({avgs[2]:.3f})")


def print_pos_summary(all_merged):
    """品詞別サマリーの表示"""
    pos_stats = defaultdict(lambda: {
        "norm_steps": [], "confidences": [], "count": 0,
    })

    for d in all_merged:
        if d.get("is_seed"):
            continue
        s = pos_stats[d["pos"]]
        s["norm_steps"].append(d["norm_step"])
        s["confidences"].append(d["confidence"])
        s["count"] += 1

    print(f"\n  {'─' * 60}")
    print(f"  品詞別集計")
    print(f"  {'─' * 60}")

    print(f"\n  {'品詞':<10} {'平均 norm_step':>14} {'平均確信度':>10} "
          f"{'分類':<6} {'N':>4}")
    print(f"  {'-' * 50}")

    for pos, s in sorted(
        pos_stats.items(), key=lambda x: np.mean(x[1]["norm_steps"]),
    ):
        avg_step = np.mean(s["norm_steps"])
        avg_conf = np.mean(s["confidences"])
        word_type = "内容語" if pos in CONTENT_POS else "機能語"
        print(
            f"  {pos:<10} {avg_step:14.3f} {avg_conf:10.4f} "
            f"{word_type:<6} {s['count']:4d}"
        )

    # 内容語 vs 機能語
    content = [
        d["norm_step"] for d in all_merged
        if d["word_type"] == "内容語" and not d.get("is_seed")
    ]
    function = [
        d["norm_step"] for d in all_merged
        if d["word_type"] == "機能語" and not d.get("is_seed")
    ]

    if content and function:
        print(f"\n  内容語: {np.mean(content):.3f} (n={len(content)})")
        print(f"  機能語: {np.mean(function):.3f} (n={len(function)})")
        if np.mean(content) < np.mean(function):
            print(f"  → 内容語が先にアンマスクされる傾向")
        else:
            print(f"  → 機能語が先にアンマスクされる傾向")

    return pos_stats


# ============================================================
# 結果保存
# ============================================================

def save_results(all_results, output_dir):
    """全結果を JSON で保存（step_snapshots は別ファイル）"""
    os.makedirs(output_dir, exist_ok=True)

    # メイン結果（step_snapshots を除く）
    output_main = []
    output_snapshots = []

    for r in all_results:
        if r is None:
            continue
        # step_snapshots は大きいので分離
        entry = {k: v for k, v in r.items() if k != "step_snapshots"}
        if "target_ids" in entry:
            entry["target_ids"] = list(entry["target_ids"])
        if "seed_positions" in entry:
            entry["seed_positions"] = list(entry["seed_positions"])
        output_main.append(entry)

        # スナップショットは別ファイル
        if r.get("step_snapshots"):
            output_snapshots.append({
                "mode": r["mode"],
                "target_text": r.get("target_text", ""),
                "seed_text": r.get("seed_text", ""),
                "step_snapshots": r["step_snapshots"],
            })

    path_main = os.path.join(output_dir, "exp4_raw_results.json")
    with open(path_main, "w", encoding="utf-8") as f:
        json.dump(output_main, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {path_main}")

    if output_snapshots:
        path_snap = os.path.join(output_dir, "exp4_step_snapshots.json")
        with open(path_snap, "w", encoding="utf-8") as f:
            json.dump(output_snapshots, f, ensure_ascii=False)
        print(f"保存: {path_snap}")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: LLaDA による seed 付き段階的テキスト成長",
    )
    parser.add_argument(
        "--mode", choices=["a", "b", "c"], default="a",
        help="実験モード: a=ガイド付き, b=部分ガイド, c=自由成長",
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="目標文（Mode A/B）",
    )
    parser.add_argument(
        "--seed", type=str, default=None,
        help="seed 語",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="全サンプルで実行",
    )
    parser.add_argument(
        "--all-modes", action="store_true",
        help="全モード（A/B/C）を順に実行",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="拡散ステップ数（デフォルト: マスク数）",
    )
    parser.add_argument(
        "--gen-length", type=int, default=64,
        help="Mode C の生成トークン数（デフォルト: 64）",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="結果出力ディレクトリ",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="乱数シード（再現性用、デフォルト: 42）",
    )
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="モデル ID を直接指定（デフォルト: GPU に基づき自動選択）",
    )
    args = parser.parse_args()

    # 再現性
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # 出力ディレクトリ
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent / "results")

    # モデルロード
    tokenizer, model = load_model(args.model_id)

    # 実行パラメータ記録
    print(f"乱数シード: {args.random_seed}")
    print(f"出力先: {args.output_dir}")

    # モード決定
    if args.all_modes:
        modes = ["a", "b", "c"]
    else:
        modes = [args.mode]

    all_results = []

    for mode in modes:
        print(f"\n{'=' * 65}")
        print(f"  Mode {mode.upper()}")
        print(f"{'=' * 65}")

        if mode in ("a", "b"):
            # 対象文の決定
            if args.target and args.seed:
                sentences = [{"text": args.target, "seed": args.seed}]
            else:
                sentences = SAMPLE_SENTENCES

            for s in sentences:
                print(f"\n{'━' * 60}")
                run_fn = run_mode_a_single if mode == "a" else run_mode_b_single
                result = run_fn(
                    s["text"], s["seed"], model, tokenizer,
                    steps=args.steps,
                )
                if result:
                    all_results.append(result)
                    print_growth_steps(result, tokenizer)
                    print_step_visualization(result, tokenizer)
                    layer_stats = analyze_layers(result["merged_data"])
                    print_layer_analysis(layer_stats)

        elif mode == "c":
            if args.seed:
                seeds = [args.seed]
            else:
                seeds = [s["seed"] for s in SAMPLE_SENTENCES]

            for seed_text in seeds:
                print(f"\n{'━' * 60}")
                result = run_mode_c_single(
                    seed_text, model, tokenizer,
                    gen_length=args.gen_length,
                    steps=args.gen_length,
                )
                if result:
                    all_results.append(result)
                    print_growth_steps(result, tokenizer)

    # 全体集計
    mode_results = defaultdict(list)
    for r in all_results:
        if r:
            mode_results[r["mode"]].append(r)

    for mode, results in sorted(mode_results.items()):
        print(f"\n{'=' * 65}")
        print(f"  Mode {mode} 全体集計")
        print(f"{'=' * 65}")

        all_merged = []
        for r in results:
            all_merged.extend(r.get("merged_data", []))

        if all_merged:
            print_pos_summary(all_merged)
            layer_stats = analyze_layers(all_merged)
            print_layer_analysis(layer_stats)

        # Mode A/B: 精度集計
        if mode in ("A", "B"):
            total_correct = sum(r.get("correct", 0) for r in results)
            total_total = sum(r.get("total", 0) for r in results)
            if total_total > 0:
                print(f"\n  全体精度: {total_correct}/{total_total} "
                      f"({total_correct / total_total * 100:.1f}%)")

        # Mode C: seed 含有率
        if mode == "C":
            seed_count = sum(1 for r in results if r.get("seed_in_output"))
            print(f"\n  seed 含有率: {seed_count}/{len(results)} "
                  f"({seed_count / len(results) * 100:.1f}%)")

    # 結果保存
    save_results(all_results, args.output_dir)
    print("\n実験完了")


if __name__ == "__main__":
    main()
