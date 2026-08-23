"""
Multi-dimensional Sweep Runner for Graph-aware Decompiler (Antigravity version).
Automates end-to-end training, multi-sample inference, and compile/pass/CodeBLEU evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def copy_prediction_pool(source: str, destination: str) -> None:
    """Copy an identical candidate pool while preserving honest provenance."""
    source_provenance = source + '.provenance.json'
    destination_provenance = destination + '.provenance.json'
    if not os.path.exists(source_provenance):
        raise SystemExit(
            f"Cannot reuse prediction pool without provenance: {source_provenance}"
        )
    shutil.copy2(source, destination)
    with open(source_provenance, 'r', encoding='utf-8') as handle:
        provenance = json.load(handle)
    if isinstance(provenance.get('output'), dict):
        provenance['output']['path'] = str(Path(destination).resolve())
    provenance['reused_candidate_pool_from'] = str(Path(source).resolve())
    with open(destination_provenance, 'w', encoding='utf-8') as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)

# Config list definition and systematic generator
DECODERS = {
    'codet5p-770m': 'Salesforce/codet5p-770m',
    'codet5p-2b': 'Salesforce/codet5p-2b',
    'qwen-0.8b': 'Qwen/Qwen3.5-0.8B',
    'qwen-2b': 'Qwen/Qwen3.5-2B',
    'qwen-2b-base': 'Qwen/Qwen3.5-2B-Base',
    'qwen-4b-base': 'Qwen/Qwen3.5-4B-Base',
    'qwen-9b-base': 'Qwen/Qwen3.5-9B-Base',
    'qwen3-8b-base': 'Qwen/Qwen3-8B',
}

TUNING_MODES = {
    'full_ft': {
        'encoder_peft': 'none',
        'decoder_peft': 'none',
        'freeze_encoder': '0',
        'freeze_decoder': '0'
    },
    'freeze_enc_ft_dec': {
        'encoder_peft': 'none',
        'decoder_peft': 'none',
        'freeze_encoder': '1',
        'freeze_decoder': '0'
    },
    'lora_enc_dec': {
        'encoder_peft': 'lora',
        'decoder_peft': 'lora',
        'freeze_encoder': '0',
        'freeze_decoder': '0'
    },
    'dora_enc_dec': {
        'encoder_peft': 'dora',
        'decoder_peft': 'dora',
        'freeze_encoder': '0',
        'freeze_decoder': '0'
    },
    'freeze_enc_lora_dec': {
        'encoder_peft': 'none',
        'decoder_peft': 'lora',
        'freeze_encoder': '1',
        'freeze_decoder': '0'
    }
}

LRS = ['1e-5', '5e-6']

LORA_CONFIGS = [{'r': 16, 'alpha': 32}]

SWEEPS = []
for dec_key, dec_model in DECODERS.items():
    for mode_key, mode_cfg in TUNING_MODES.items():
        for lr in LRS:
            # If PEFT mode, sweep over LoRA configurations
            if mode_key in ['lora_enc_dec', 'dora_enc_dec']:
                for lora in LORA_CONFIGS:
                    r_val = lora['r']
                    alpha_val = lora['alpha']
                    name = f"{dec_key}_{mode_key}_r{r_val}_{lr.replace('-', '')}"
                    
                    is_large = any(size in dec_key for size in ['770m', '2b', '4b', '8b', '9b'])
                    is_medium = ('0.8b' in dec_key)
                    oom_risk = "high" if is_large else ("medium" if is_medium else "low")
                    
                    SWEEPS.append({
                        'name': name,
                        'decoder': dec_model,
                        'encoder_peft': mode_cfg['encoder_peft'],
                        'decoder_peft': mode_cfg['decoder_peft'],
                        'freeze_encoder': mode_cfg['freeze_encoder'],
                        'freeze_decoder': mode_cfg['freeze_decoder'],
                        'lr': lr,
                        'epochs': '1',
                        'oom_risk': oom_risk,
                        'lora_r': str(r_val),
                        'lora_alpha': str(alpha_val)
                    })
            else:
                # Non-PEFT modes
                name = f"{dec_key}_{mode_key}_{lr.replace('-', '')}"
                
                is_large = any(size in dec_key for size in ['770m', '2b', '4b', '8b', '9b'])
                is_medium = ('0.8b' in dec_key)
                is_full_ft = (mode_key in ['full_ft', 'freeze_enc_ft_dec'])
                
                if is_large:
                    oom_risk = "critical" if is_full_ft else "high"
                elif is_medium:
                    oom_risk = "high" if is_full_ft else "medium"
                else:
                    oom_risk = "low"
                    
                SWEEPS.append({
                    'name': name,
                    'decoder': dec_model,
                    'encoder_peft': mode_cfg['encoder_peft'],
                    'decoder_peft': mode_cfg['decoder_peft'],
                    'freeze_encoder': mode_cfg['freeze_encoder'],
                    'freeze_decoder': mode_cfg['freeze_decoder'],
                    'lr': lr,
                    'epochs': '1',
                    'oom_risk': oom_risk,
                    'lora_r': '16',
                    'lora_alpha': '32'
                })


def extract_json(text: str):
    idx = text.find('{')
    if idx != -1:
        return json.loads(text[idx:])
    return json.loads(text)


def _resolve_hf_token(token: str | None = None) -> str:
    resolved = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    )
    if resolved:
        return resolved
    # `hf auth login` stores the credential under HF_HOME. Requiring the same
    # secret to also be exported made authenticated rented pods silently skip
    # final artifact uploads.
    try:
        from huggingface_hub import get_token

        return get_token() or ""
    except Exception:
        return ""


def upload_to_huggingface(
    repo_id: str,
    local_path: str | Path,
    path_in_repo: str,
    *,
    token: str | None = None,
    private: bool = True,
    repo_type: str = "model",
    commit_message: str | None = None,
) -> bool:
    if not repo_id:
        return False

    local_path = Path(local_path)
    if not local_path.exists():
        print(f"HF upload skipped, missing path: {local_path}")
        return False

    hf_token = _resolve_hf_token(token)
    if not hf_token:
        print("HF upload skipped: set HF_TOKEN/HUGGINGFACE_TOKEN or pass --hf_token.")
        return False

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
        if local_path.is_dir():
            api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=str(local_path),
                path_in_repo=path_in_repo,
                commit_message=commit_message or f"Upload {path_in_repo}",
            )
        else:
            api.upload_file(
                repo_id=repo_id,
                repo_type=repo_type,
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                commit_message=commit_message or f"Upload {path_in_repo}",
            )
        print(f"HF upload complete: {repo_id}/{path_in_repo}")
        return True
    except Exception as exc:
        print(f"HF upload failed for {local_path}: {exc}")
        return False


def run_experiment(
    cfg,
    limit_eval=0,
    num_samples=5,
    pass_num_samples=10,
    generation_batch_size=5,
    eval_max_new_tokens=768,
    dry_run=False,
    use_grpo=False,
    use_local=False,
    num_gpus=1,
    train_file=None,
    eval_file=None,
    compile_dataset='data/testing/compile-test2.jsonl',
    pass_dataset='data/testing/grpo_data.jsonl',
    grpo_train_file=None,
    train_batch_size=None,
    grad_accum_override=None,
    load_4bit=None,
    grpo_group_size=None,
    grpo_epochs=None,
    grpo_test_timeout=None,
    grpo_max_new_tokens=None,
    grpo_reward_workers=None,
    metric_workers=None,
    compile_mode='legacy',
    sft_checkpoint=None,
    grpo_checkpoint=None,
    sft_lr=None,
    grpo_lr=None,
    grpo_perfect_bonus=None,
    grpo_reward_mode=None,
    grpo_verpo_alpha=None,
    grpo_verpo_anchor_weight=None,
    grpo_verpo_density_norm=None,
    grpo_binary_fail_reward=None,
    grpo_no_compile_penalty=None,
    grpo_compile_reward=None,
    grpo_partial_reward_cap=None,
    grpo_perfect_base_reward=None,
    grpo_overlap_weight=None,
    grpo_unique_test_bonus=None,
    grpo_duplicate_penalty=None,
    grpo_kl_coef=None,
    grpo_clip_eps=None,
    grpo_entropy_coef=None,
    grpo_gen_temperature=None,
    grpo_gen_top_p=None,
    grpo_adv_norm=None,
    grpo_min_reward_range=None,
    grpo_passk_k=None,
    grpo_score_chunk_size=None,
    grpo_loss_pooling=None,
    grpo_simko_k=None,
    grpo_overlong_filter=None,
    grpo_clip_eps_high=None,
    grpo_reward_preflight_batches=None,
    qwen_prefix_tokens=None,
    qwen_prefix_gate_init=None,
    qwen_prefix_gate_mode=None,
    qwen_prefix_dynamic=None,
    qwen_prefix_min_tokens=None,
    qwen_prefix_tokens_per_log2=None,
    qwen_prefix_gate_override=None,
    qwen_prefix_rms_match=None,
    decoder_prompt_max_length=None,
    prompt_fit_assembly=None,
    auto_cfg=None,
    prompt_assembly_mode=None,
    prompt_clean_asm=None,
    max_block_instrs=None,
    dfg_mode=None,
    edge_ablation=None,
    gnn_ablation=None,
    gnn_layers=None,
    global_attention_ablation=None,
    region_compression=None,
    region_max_blocks=None,
    block_pooling=None,
    block_vectors_per_block=None,
    add_reverse_edges=None,
    block_position_mode=None,
    position_scheme=None,
    causal_position_ids=None,
    grpo_train_graph_glue=None,
    use_reasoning=None,
    attn_implementation=None,
    gradient_checkpointing=None,
    hf_repo=None,
    hf_token=None,
    hf_private=True,
    hf_upload_checkpoints=True,
    save_strategy=None,
    save_steps=None,
    save_total_limit=None,
    skip_training=False,
    skip_inference=False,
    quiet=True,
    decoder_revision='',
    encoder_revision='',
    seed=42,
    eval_graph_input_ablation='none',
    eval_graph_ablation_seed=None,
):
    print(f"\n==================================================")
    print(f"RUNNING EXPERIMENT: {cfg['name']}")
    print(f"Decoder: {cfg['decoder']}")
    print(f"Encoder PEFT: {cfg['encoder_peft']} | Decoder PEFT: {cfg['decoder_peft']}")
    print(f"LR: {cfg['lr']} | Epochs: {cfg['epochs']}")
    print(f"OOM Risk (on 6GB VRAM): {cfg['oom_risk'].upper()}")
    if use_grpo:
        print("Training Mode: GRPO Reinforcement Learning")
    else:
        print("Training Mode: SFT (Supervised Fine-Tuning)")
    if dry_run:
        print("Running in DRY-RUN mode (2 training steps, 2 eval samples)")
    print(f"==================================================\n")

    workspace_root = Path(__file__).resolve().parent.parent
    msc_code_root = workspace_root
    
    env = os.environ.copy()
    env['PYTHONPATH'] = str(msc_code_root)

    # Quiet mode: silence framework warning spam in ALL child processes
    # (training, inference, metrics). Errors and tracebacks stay loud -
    # suppressed errors are how fake zero-metric runs happened before.
    if quiet:
        env['GRAPH_QUIET'] = '1'
        env.setdefault('TRANSFORMERS_VERBOSITY', 'error')
        env.setdefault('DATASETS_VERBOSITY', 'error')
        env.setdefault('TOKENIZERS_PARALLELISM', 'false')
        env.setdefault('PYTHONWARNINGS', 'ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning')
    else:
        env['GRAPH_QUIET'] = '0'
    
    path_sep = ';' if os.name == 'nt' else ':'
    if os.name != 'nt':
        dart_bin = '/home/zeus/dart-sdk/bin' if os.path.exists('/home/zeus/dart-sdk/bin') else os.path.join(os.path.expanduser('~'), 'dart-sdk', 'bin')
        env['PATH'] = f"{dart_bin}{path_sep}{env.get('PATH', '')}"
        
    env['GRAPH_ENCODER_MODEL'] = os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base")
    env['GRAPH_DECODER_MODEL'] = cfg['decoder']
    env['GRAPH_DECODER_REVISION'] = str(decoder_revision or '')
    env['GRAPH_ENCODER_REVISION'] = str(encoder_revision or '')
    env['GRAPH_SEED'] = str(seed)
    env['GRAPH_ENCODER_PEFT'] = cfg['encoder_peft']
    env['GRAPH_DECODER_PEFT'] = cfg['decoder_peft']
    env['GRAPH_FREEZE_ENCODER'] = cfg['freeze_encoder']
    env['GRAPH_FREEZE_DECODER'] = cfg['freeze_decoder']
    env['GRAPH_LR'] = cfg['lr']
    env['GRAPH_EPOCHS'] = cfg['epochs']
    if train_file is not None:
        env['GRAPH_TRAIN_FILE'] = str(train_file)
    if eval_file is not None:
        env['GRAPH_EVAL_FILE'] = str(eval_file)
    if grpo_train_file is not None:
        env['GRPO_TRAIN_FILE'] = str(grpo_train_file)
    if grpo_reward_mode is not None:
        env['GRPO_REWARD_MODE'] = str(grpo_reward_mode)
    if grpo_verpo_alpha is not None:
        env['GRPO_VERPO_ALPHA'] = str(grpo_verpo_alpha)
    if grpo_verpo_anchor_weight is not None:
        env['GRPO_VERPO_ANCHOR_WEIGHT'] = str(grpo_verpo_anchor_weight)
    if grpo_verpo_density_norm is not None:
        env['GRPO_VERPO_DENSITY_NORM'] = '1' if grpo_verpo_density_norm else '0'
    if grpo_binary_fail_reward is not None:
        env['GRPO_BINARY_FAIL_REWARD'] = str(grpo_binary_fail_reward)
    if grpo_no_compile_penalty is not None:
        env['GRPO_NO_COMPILE_PENALTY'] = str(grpo_no_compile_penalty)
    if grpo_compile_reward is not None:
        env['GRPO_COMPILE_REWARD'] = str(grpo_compile_reward)
    if grpo_partial_reward_cap is not None:
        env['GRPO_PARTIAL_REWARD_CAP'] = str(grpo_partial_reward_cap)
    if grpo_perfect_base_reward is not None:
        env['GRPO_PERFECT_BASE_REWARD'] = str(grpo_perfect_base_reward)
    if grpo_overlap_weight is not None:
        env['GRPO_OVERLAP_WEIGHT'] = str(grpo_overlap_weight)
    if grpo_unique_test_bonus is not None:
        env['GRPO_UNIQUE_TEST_BONUS'] = str(grpo_unique_test_bonus)
    if grpo_duplicate_penalty is not None:
        env['GRPO_DUPLICATE_PENALTY'] = str(grpo_duplicate_penalty)
    if grpo_kl_coef is not None:
        env['GRPO_KL_COEF'] = str(grpo_kl_coef)
    if grpo_clip_eps is not None:
        env['GRPO_CLIP_EPS'] = str(grpo_clip_eps)
    if grpo_entropy_coef is not None:
        env['GRPO_ENTROPY_COEF'] = str(grpo_entropy_coef)
    if grpo_gen_temperature is not None:
        env['GRPO_GEN_TEMPERATURE'] = str(grpo_gen_temperature)
    if grpo_gen_top_p is not None:
        env['GRPO_GEN_TOP_P'] = str(grpo_gen_top_p)
    if grpo_adv_norm is not None:
        env['GRPO_ADV_NORM'] = str(grpo_adv_norm)
    if grpo_min_reward_range is not None:
        env['GRPO_MIN_REWARD_RANGE'] = str(grpo_min_reward_range)
    if grpo_passk_k is not None:
        env['GRPO_PASSK_K'] = str(grpo_passk_k)
    if grpo_score_chunk_size is not None:
        env['GRPO_SCORE_CHUNK_SIZE'] = str(grpo_score_chunk_size)
    if grpo_loss_pooling is not None:
        env['GRPO_LOSS_POOLING'] = str(grpo_loss_pooling)
    if grpo_simko_k is not None:
        env['GRPO_SIMKO_K'] = str(grpo_simko_k)
    if grpo_overlong_filter is not None:
        env['GRPO_OVERLONG_FILTER'] = str(grpo_overlong_filter)
    if grpo_clip_eps_high is not None:
        env['GRPO_CLIP_EPS_HIGH'] = str(grpo_clip_eps_high)
    if grpo_reward_preflight_batches is not None:
        env['GRPO_REWARD_PREFLIGHT_BATCHES'] = str(grpo_reward_preflight_batches)
    if sft_lr is not None:
        env['GRAPH_LR'] = str(sft_lr)
    if qwen_prefix_tokens is not None:
        env['GRAPH_QWEN_PREFIX_TOKENS'] = str(qwen_prefix_tokens)
    if qwen_prefix_gate_init is not None:
        env['GRAPH_QWEN_PREFIX_GATE_INIT'] = str(qwen_prefix_gate_init)
    if qwen_prefix_gate_mode is not None:
        env['GRAPH_QWEN_PREFIX_GATE_MODE'] = str(qwen_prefix_gate_mode)
    if qwen_prefix_dynamic is not None:
        env['GRAPH_QWEN_PREFIX_DYNAMIC'] = '1' if qwen_prefix_dynamic else '0'
    if qwen_prefix_min_tokens is not None:
        env['GRAPH_QWEN_PREFIX_MIN_TOKENS'] = str(qwen_prefix_min_tokens)
    if qwen_prefix_tokens_per_log2 is not None:
        env['GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2'] = str(qwen_prefix_tokens_per_log2)
    if qwen_prefix_gate_override is not None:
        env['GRAPH_QWEN_PREFIX_GATE_OVERRIDE'] = str(qwen_prefix_gate_override)
    if qwen_prefix_rms_match is not None:
        env['GRAPH_QWEN_PREFIX_RMS_MATCH'] = '1' if qwen_prefix_rms_match else '0'
    if decoder_prompt_max_length is not None:
        env['GRAPH_DECODER_PROMPT_MAX_LENGTH'] = str(decoder_prompt_max_length)
    if prompt_fit_assembly is not None:
        env['GRAPH_PROMPT_FIT_ASSEMBLY'] = '1' if prompt_fit_assembly else '0'
    if auto_cfg is not None:
        env['GRAPH_AUTO_CFG'] = '1' if auto_cfg else '0'
    if prompt_assembly_mode is not None:
        env['GRAPH_PROMPT_ASSEMBLY_MODE'] = str(prompt_assembly_mode)
    if prompt_clean_asm is not None:
        env['GRAPH_PROMPT_CLEAN_ASM'] = '1' if prompt_clean_asm else '0'
    if max_block_instrs is not None:
        env['GRAPH_MAX_BLOCK_INSTRS'] = str(max_block_instrs)
    if dfg_mode is not None:
        env['GRAPH_DFG_MODE'] = str(dfg_mode)
    if edge_ablation is not None:
        env['GRAPH_EDGE_ABLATION'] = str(edge_ablation)
    if gnn_ablation is not None:
        env['GRAPH_GNN_ABLATION'] = str(gnn_ablation)
    if gnn_layers is not None:
        env['GRAPH_GNN_LAYERS'] = str(gnn_layers)
    if global_attention_ablation is not None:
        env['GRAPH_GLOBAL_ATTENTION_ABLATION'] = str(global_attention_ablation)
    if region_compression is not None:
        env['GRAPH_REGION_COMPRESSION'] = str(region_compression)
    if region_max_blocks is not None:
        env['GRAPH_REGION_MAX_BLOCKS'] = str(region_max_blocks)
    if block_pooling is not None:
        env['GRAPH_BLOCK_POOLING'] = str(block_pooling)
    if block_vectors_per_block is not None:
        env['GRAPH_BLOCK_VECTORS_PER_BLOCK'] = str(block_vectors_per_block)
    if add_reverse_edges is not None:
        env['GRAPH_ADD_REVERSE_EDGES'] = '1' if add_reverse_edges else '0'
    if block_position_mode is not None:
        env['GRAPH_BLOCK_POSITION_MODE'] = str(block_position_mode)
    if position_scheme is not None:
        env['GRAPH_POSITION_SCHEME'] = str(position_scheme)
    if causal_position_ids is not None:
        env['GRAPH_CAUSAL_POSITION_IDS'] = str(causal_position_ids)
    if grpo_train_graph_glue is not None:
        env['GRPO_TRAIN_GRAPH_GLUE'] = '1' if grpo_train_graph_glue else '0'
    if use_reasoning is not None:
        env['GRAPH_USE_REASONING'] = '1' if use_reasoning else '0'
    if attn_implementation is not None:
        env['GRAPH_ATTN_IMPLEMENTATION'] = str(attn_implementation)
    if gradient_checkpointing is not None:
        env['GRAPH_GRADIENT_CHECKPOINTING'] = '1' if gradient_checkpointing else '0'
    if hf_repo:
        env['GRAPH_HF_REPO'] = str(hf_repo)
        if hf_token:
            env['GRAPH_HF_TOKEN'] = str(hf_token)
        env['GRAPH_HF_PRIVATE'] = '1' if hf_private else '0'
        env['GRAPH_HF_UPLOAD_CHECKPOINTS'] = '1' if hf_upload_checkpoints else '0'
        env['GRAPH_HF_PATH_PREFIX'] = f"artifacts/{cfg['name']}{'_grpo' if use_grpo else ''}"
        if hf_upload_checkpoints and save_strategy is None:
            save_strategy = "epoch"
    if save_strategy is not None:
        env['GRAPH_SAVE_STRATEGY'] = str(save_strategy)
    if save_steps is not None:
        env['GRAPH_SAVE_STEPS'] = str(save_steps)
    if save_total_limit is not None:
        env['GRAPH_SAVE_TOTAL_LIMIT'] = str(save_total_limit)
    
    # Dynamic batch size and gradient accumulation based on local vs remote
    dec_lower = cfg['decoder'].lower()
    if use_local:
        # RTX 3060 6GB VRAM limits:
        if '9b' in dec_lower or '9b' in cfg['name']:
            batch_size = 1
            grad_accum = 64
        elif '8b' in dec_lower or '8b' in cfg['name']:
            batch_size = 1
            grad_accum = 64
        elif '4b' in dec_lower or '4b' in cfg['name']:
            batch_size = 1
            grad_accum = 32
        elif '2b' in dec_lower or '2b' in cfg['name']:
            batch_size = 1
            grad_accum = 8
        elif '770m' in dec_lower or '0.8b' in dec_lower:
            batch_size = 2
            grad_accum = 4
        elif 'base' in dec_lower:
            batch_size = 4
            grad_accum = 2
        else:
            batch_size = 8
            grad_accum = 1
    else:
        # Remote defaults (H100 80GB): full bf16, no 4-bit quantization, and
        # large batches to fully utilize the GPU.
        env['GRAPH_LOAD_4BIT'] = '0'
        # Reduce allocator fragmentation; helps recover the last GB or two that
        # would otherwise be lost to fragmentation on near-full GPUs.
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        grad_accum = 1
        if '9b' in dec_lower or '9b' in cfg['name']:
            batch_size = 1
            grad_accum = 64
        elif '8b' in dec_lower or '8b' in cfg['name']:
            batch_size = 1
            grad_accum = 64
        elif '4b' in dec_lower or '4b' in cfg['name']:
            batch_size = 1
            grad_accum = 32
        elif '2b' in dec_lower or '2b' in cfg['name']:
            # 2b activations are very large; batch_size 8 OOMs even on an 80GB
            # H100 (full bf16, no quantization). Use a small per-device batch
            # and high gradient accumulation to keep the same effective batch
            # of 32 while fitting activations in memory.
            batch_size = 2
            grad_accum = 16
        elif '770m' in dec_lower or '0.8b' in dec_lower:
            batch_size = 32
        elif 'base' in dec_lower:
            batch_size = 48
        else:
            batch_size = 64  # Salesforce/codet5-small

    if dry_run:
        batch_size = 2
        grad_accum = 1
        env['GRAPH_MAX_STEPS'] = '2'
        limit_eval = 2
        num_samples = 2
        pass_num_samples = 2
    else:
        if train_batch_size is not None:
            batch_size = train_batch_size
        if grad_accum_override is not None:
            grad_accum = grad_accum_override

    if load_4bit is not None:
        env['GRAPH_LOAD_4BIT'] = '1' if load_4bit else '0'

    env['GRAPH_BATCH_SIZE'] = str(batch_size)
    env['GRAPH_GRAD_ACCUM'] = str(grad_accum)
    env['GRAPH_LORA_R'] = cfg.get('lora_r', '16')
    env['GRAPH_LORA_ALPHA'] = cfg.get('lora_alpha', '32')
    print(
        f"Dynamic Batch Size: {batch_size} | Grad Accum: {grad_accum} | "
        f"LoRA R: {env['GRAPH_LORA_R']} | Alpha: {env['GRAPH_LORA_ALPHA']} | "
        f"Qwen prefix tokens: {env.get('GRAPH_QWEN_PREFIX_TOKENS', '16')} | "
        f"Qwen gate init: {env.get('GRAPH_QWEN_PREFIX_GATE_INIT', '0.2')} | "
        f"Qwen gate mode: {env.get('GRAPH_QWEN_PREFIX_GATE_MODE', 'scalar')} | "
        f"Dynamic prefix: {env.get('GRAPH_QWEN_PREFIX_DYNAMIC', '0')} "
        f"[{env.get('GRAPH_QWEN_PREFIX_MIN_TOKENS', '4')}, "
        f"{env.get('GRAPH_QWEN_PREFIX_TOKENS_PER_LOG2', '4')}/log2] | "
        f"Prefix RMS match: {env.get('GRAPH_QWEN_PREFIX_RMS_MATCH', '0')} | "
        f"Prompt tokens: {env.get('GRAPH_DECODER_PROMPT_MAX_LENGTH', '768')} | "
        f"Reasoning: {env.get('GRAPH_USE_REASONING', '0')} | "
        f"Attention: {env.get('GRAPH_ATTN_IMPLEMENTATION', 'auto')} | "
        f"Grad ckpt: {env.get('GRAPH_GRADIENT_CHECKPOINTING', '0')}"
    )
    print(
        f"Graph channel: dfg_mode={env.get('GRAPH_DFG_MODE', 'legacy')} | "
        f"edge_ablation={env.get('GRAPH_EDGE_ABLATION', 'full')} | "
        f"gnn_ablation={env.get('GRAPH_GNN_ABLATION', 'full')} | "
        f"gnn_layers={env.get('GRAPH_GNN_LAYERS', '4')} | "
        f"global_attention={env.get('GRAPH_GLOBAL_ATTENTION_ABLATION', 'full')} | "
        f"region_compression={env.get('GRAPH_REGION_COMPRESSION', 'off')} | "
        f"region_max_blocks={env.get('GRAPH_REGION_MAX_BLOCKS', '8')} | "
        f"block_pooling={env.get('GRAPH_BLOCK_POOLING', 'cls')} | "
        f"block_vectors={env.get('GRAPH_BLOCK_VECTORS_PER_BLOCK', '4')} | "
        f"reverse_edges={env.get('GRAPH_ADD_REVERSE_EDGES', '0')} | "
        f"block_positions={env.get('GRAPH_BLOCK_POSITION_MODE', 'off')} | "
        f"position_scheme={env.get('GRAPH_POSITION_SCHEME', 'legacy')} | "
        f"causal_position_ids={env.get('GRAPH_CAUSAL_POSITION_IDS', 'cumsum')} | "
        f"grpo_train_graph_glue={env.get('GRPO_TRAIN_GRAPH_GLUE', '1')} | "
        f"auto_cfg={env.get('GRAPH_AUTO_CFG', '0')}"
    )
        
    if use_grpo:
        output_dir = f"artifacts/{cfg['name']}_grpo"
        compile_predictions_file = str(msc_code_root / 'results' / f"{cfg['name']}_grpo_compile_predictions.json")
        pass_predictions_file = str(msc_code_root / 'results' / f"{cfg['name']}_grpo_pass_predictions.json")
        checkpoint_file = str(msc_code_root / 'artifacts' / f"{cfg['name']}_grpo" / 'pytorch_model.bin')

        # GRPO wall-clock is dominated by (a) autoregressive generation of G
        # completions/prompt and (b) one cold `dart run` per assertion to score
        # them. These are CLI-configurable so runs do not need shell env vars.
        env['GRPO_GROUP_SIZE'] = str(grpo_group_size if grpo_group_size is not None else 4)
        env['GRPO_EPOCHS'] = str(grpo_epochs if grpo_epochs is not None else 1)
        # 3s was too tight: with 32-64 parallel reward workers, cold dart
        # compile/run regularly exceeds it and fine candidates get scored as
        # failures, which is pure reward noise.
        env['GRPO_TEST_TIMEOUT'] = str(grpo_test_timeout if grpo_test_timeout is not None else 8)
        if grpo_lr is not None:
            env['GRPO_LR'] = str(grpo_lr)
        if grpo_perfect_bonus is not None:
            env['GRPO_PERFECT_BONUS'] = str(grpo_perfect_bonus)
        if grpo_max_new_tokens is not None:
            env['GRPO_MAX_NEW_TOKENS'] = str(grpo_max_new_tokens)
        # Parallelize the per-assertion `dart run` calls across host CPU cores.
        import multiprocessing
        default_workers = str(max(1, min(32, multiprocessing.cpu_count() - 1)))
        env['GRPO_REWARD_WORKERS'] = str(grpo_reward_workers if grpo_reward_workers is not None else default_workers)
        print(
            f"GRPO speed knobs: group_size={env['GRPO_GROUP_SIZE']} "
            f"epochs={env['GRPO_EPOCHS']} test_timeout={env['GRPO_TEST_TIMEOUT']}s "
            f"lr={env.get('GRPO_LR', '5e-6')} "
            f"perfect_bonus={env.get('GRPO_PERFECT_BONUS', '2.0')} "
            f"entropy_coef={env.get('GRPO_ENTROPY_COEF', '0.0')} "
            f"reward_workers={env['GRPO_REWARD_WORKERS']} "
            f"max_new_tokens={env.get('GRPO_MAX_NEW_TOKENS', '256')}"
        )

        if grpo_checkpoint:
            resume_checkpoint = str(Path(grpo_checkpoint).expanduser())
            if not os.path.exists(resume_checkpoint):
                raise SystemExit(
                    f"--grpo_checkpoint={resume_checkpoint!r} does not exist. Fix the path, or "
                    f"omit --grpo_checkpoint to auto-load the SFT artifact / base model."
                )
            env['GRAPH_CHECKPOINT'] = resume_checkpoint
            print(f"GRPO stage: Continuing from checkpoint: {resume_checkpoint}")
            if skip_training:
                checkpoint_file = resume_checkpoint
        else:
            sft_checkpoint = str(msc_code_root / 'artifacts' / cfg['name'] / 'pytorch_model.bin')
            if os.path.exists(sft_checkpoint):
                env['GRAPH_CHECKPOINT'] = sft_checkpoint
                print(f"GRPO stage: Found SFT checkpoint at {sft_checkpoint}. Loading it for GRPO.")
            else:
                print(f"GRPO stage: SFT checkpoint not found at {sft_checkpoint}. Running GRPO from base pre-trained model.")
    else:
        output_dir = f"artifacts/{cfg['name']}"
        compile_predictions_file = str(msc_code_root / 'results' / f"{cfg['name']}_compile_predictions.json")
        pass_predictions_file = str(msc_code_root / 'results' / f"{cfg['name']}_pass_predictions.json")
        checkpoint_file = str(msc_code_root / 'artifacts' / cfg['name'] / 'pytorch_model.bin')
        if sft_checkpoint:
            resume_checkpoint = str(Path(sft_checkpoint).expanduser())
            if not os.path.exists(resume_checkpoint):
                raise SystemExit(
                    f"--sft_checkpoint={resume_checkpoint!r} does not exist. Fix the path, or "
                    f"omit --sft_checkpoint to evaluate/train from the base model."
                )
            env['GRAPH_CHECKPOINT'] = resume_checkpoint
            print(f"SFT stage: Continuing from checkpoint: {resume_checkpoint}")
            if skip_training:
                checkpoint_file = resume_checkpoint
        
    env['GRAPH_OUTPUT_DIR'] = output_dir
    eval_metric_workers = str(metric_workers) if metric_workers is not None else os.environ.get("EVAL_DART_WORKERS", "1")

    # 1. Run training pipeline
    train_module = 'scripts.training.graph_grpo_decompiler_antigravity' if use_grpo else 'scripts.training.graph_encoder_decoder_decompiler_v2_antigravity'
    if skip_training:
        print(f"Skipping training; evaluating existing checkpoint: {checkpoint_file}")
    else:
        if num_gpus and num_gpus > 1 and not use_grpo:
            # Data-parallel SFT via DDP. torch.distributed.run spawns one process
            # per GPU and sets LOCAL_RANK/WORLD_SIZE/RANK, which the HF Trainer
            # picks up automatically. --standalone handles single-node rendezvous.
            print(f"[multi-gpu] Launching SFT on {num_gpus} GPUs via torch.distributed.run (DDP).")
            print(
                f"[multi-gpu] NOTE: effective batch = per_device({batch_size}) x "
                f"gpus({num_gpus}) x grad_accum({grad_accum}) = "
                f"{batch_size * num_gpus * grad_accum}. Halve --grad_accum to keep it constant."
            )
            train_cmd = [
                sys.executable, '-m', 'torch.distributed.run',
                '--standalone', '--nproc_per_node', str(num_gpus),
                '--module', train_module,
            ]
        else:
            if num_gpus and num_gpus > 1 and use_grpo:
                print(
                    f"[multi-gpu] GRPO is not DDP-enabled yet; running on a single GPU "
                    f"(--num_gpus={num_gpus} applies to SFT only)."
                )
            train_cmd = [sys.executable, '-m', train_module]
        subprocess.run(train_cmd, cwd=str(msc_code_root), env=env, check=True)
        if use_grpo and grpo_reward_preflight_batches and grpo_reward_preflight_batches > 0:
            print("GRPO reward preflight complete; no training checkpoint or evaluation was requested.")
            return
        if hf_repo:
            artifact_path = msc_code_root / output_dir
            upload_to_huggingface(
                hf_repo,
                artifact_path,
                f"artifacts/{cfg['name']}{'_grpo' if use_grpo else ''}",
                token=hf_token,
                private=hf_private,
                commit_message=f"Upload {'GRPO' if use_grpo else 'SFT'} artifact {cfg['name']}",
            )

    inference_env = env
    inference_checkpoint = checkpoint_file
    if checkpoint_file and os.path.exists(checkpoint_file):
        pass
    elif skip_training and not (sft_checkpoint or grpo_checkpoint):
        inference_checkpoint = ""
        inference_env = env.copy()
        inference_env['GRAPH_ENCODER_PEFT'] = 'none'
        inference_env['GRAPH_DECODER_PEFT'] = 'none'
        print(
            "No checkpoint found at "
            f"{checkpoint_file}; evaluating the base decoder/encoder without PEFT adapters."
        )
    else:
        raise SystemExit(
            f"Expected checkpoint not found: {checkpoint_file}. "
            "Use --sft_checkpoint/--grpo_checkpoint, remove --skip_training to train it, "
            "or omit the checkpoint flags only for an intentional raw base-model eval."
        )

    # When both metrics use the same dataset and sample count, generate one
    # candidate pool. Besides saving GPU time, this guarantees that compile@k
    # and pass@k describe exactly the same stochastic samples.
    compile_dataset_path = (msc_code_root / compile_dataset).resolve()
    pass_dataset_path = (msc_code_root / pass_dataset).resolve()
    shared_eval_pool = (
        compile_dataset_path == pass_dataset_path
        and int(num_samples) == int(pass_num_samples)
    )

    # 2./3. Multi-sample inference, or reuse existing prediction files.
    # --skip_inference exists so a crash in the metrics stage (e.g. a broken
    # tree-sitter discovered AFTER training+inference) costs minutes to
    # resume, not a fresh round of GPU generation.
    if skip_inference:
        if shared_eval_pool:
            if os.path.exists(compile_predictions_file) and not os.path.exists(pass_predictions_file):
                copy_prediction_pool(compile_predictions_file, pass_predictions_file)
            elif os.path.exists(pass_predictions_file) and not os.path.exists(compile_predictions_file):
                copy_prediction_pool(pass_predictions_file, compile_predictions_file)
        missing = [p for p in (compile_predictions_file, pass_predictions_file) if not os.path.exists(p)]
        if missing:
            raise SystemExit(
                "--skip_inference was set but prediction files are missing:\n  "
                + "\n  ".join(missing)
            )
        print("Skipping inference; reusing existing prediction files:")
        print(f"  {compile_predictions_file}")
        print(f"  {pass_predictions_file}")
    else:
        print("\nRunning multi-sample inference for legacy compile/CodeBLEU tasks...")
        compile_infer_cmd = [
            sys.executable, str(msc_code_root / 'scripts/evaluation/graph_inference_antigravity.py'),
            '--dataset', str(msc_code_root / compile_dataset),
            '--decoder_model', cfg['decoder'],
            '--output', compile_predictions_file,
            '--limit', str(limit_eval),
            '--num_samples', str(num_samples),
            '--generation_batch_size', str(generation_batch_size),
            '--max_new_tokens', str(eval_max_new_tokens),
            '--decoder_prompt_max_length', str(decoder_prompt_max_length or env.get('GRAPH_DECODER_PROMPT_MAX_LENGTH', '768')),
            '--decoder_revision', str(decoder_revision or ''),
            '--encoder_revision', str(encoder_revision or ''),
            '--seed', str(seed),
            '--graph_input_ablation', str(eval_graph_input_ablation or 'none'),
            '--graph_ablation_seed', str(
                seed if eval_graph_ablation_seed is None else eval_graph_ablation_seed
            ),
        ]
        if inference_checkpoint:
            compile_infer_cmd.extend(['--checkpoint', inference_checkpoint])
        subprocess.run(compile_infer_cmd, env=inference_env, check=True)

        if shared_eval_pool:
            print("\nCompile and pass evaluation settings match; reusing the same candidate pool.")
            copy_prediction_pool(compile_predictions_file, pass_predictions_file)
        else:
            print("\nRunning multi-sample inference for legacy pass@k tasks...")
            pass_infer_cmd = [
                sys.executable, str(msc_code_root / 'scripts/evaluation/graph_inference_antigravity.py'),
                '--dataset', str(msc_code_root / pass_dataset),
                '--decoder_model', cfg['decoder'],
                '--output', pass_predictions_file,
                '--limit', str(limit_eval),
                '--num_samples', str(pass_num_samples),
                '--generation_batch_size', str(generation_batch_size),
                '--max_new_tokens', str(eval_max_new_tokens),
                '--decoder_prompt_max_length', str(decoder_prompt_max_length or env.get('GRAPH_DECODER_PROMPT_MAX_LENGTH', '768')),
                '--decoder_revision', str(decoder_revision or ''),
                '--encoder_revision', str(encoder_revision or ''),
                '--seed', str(seed),
                '--graph_input_ablation', str(eval_graph_input_ablation or 'none'),
                '--graph_ablation_seed', str(
                    seed if eval_graph_ablation_seed is None else eval_graph_ablation_seed
                ),
            ]
            if inference_checkpoint:
                pass_infer_cmd.extend(['--checkpoint', inference_checkpoint])
            subprocess.run(pass_infer_cmd, env=inference_env, check=True)

    # 4. Calculate metrics
    print("\nCalculating metrics...")
    
    # CodeBLEU on the compile task set. The raw score is comparable to the
    # legacy raw CodeBLEU files; compiled_only mirrors compile-test.py output.
    codebleu_out = subprocess.check_output([
        sys.executable, str(msc_code_root / 'scripts/evaluation/graph_codebleu_antigravity.py'),
        '--predictions', compile_predictions_file
    ], env=env, text=True)
    print("CodeBLEU Results:")
    print(codebleu_out)

    codebleu_compiled_out = subprocess.check_output([
        sys.executable, str(msc_code_root / 'scripts/evaluation/graph_codebleu_antigravity.py'),
        '--predictions', compile_predictions_file,
        '--compiled_only',
        '--workers', eval_metric_workers,
    ], env=env, text=True)
    print("Compiled-only CodeBLEU Results:")
    print(codebleu_compiled_out)
    
    # Compile@k
    compile_out = subprocess.check_output([
        sys.executable, str(msc_code_root / 'scripts/evaluation/graph_compile_at_k_antigravity.py'),
        '--predictions', compile_predictions_file,
        '--k_values', '1,3' if dry_run else '1,5',
        '--workers', eval_metric_workers,
        '--compile_mode', compile_mode,
    ], env=env, text=True)
    print("Compile@k Results:")
    print(compile_out)

    # Pass@k. Capture stdout only: stderr stays live so the evaluator's
    # progress bar is visible during the multi-minute Dart run.
    pass_out = subprocess.run([
        sys.executable, str(msc_code_root / 'scripts/evaluation/graph_pass_at_k_antigravity.py'),
        '--predictions', pass_predictions_file,
        '--k_values', '1,2' if dry_run else '1,5,10',
        '--workers', eval_metric_workers,
    ], env=env, stdout=subprocess.PIPE, text=True, check=True)
    print("Pass@k Results:")
    print(pass_out.stdout)

    # Save summary results
    results_dir = msc_code_root / "results" / "sweeps_antigravity"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'experiment': cfg['name'],
        'config': cfg,
        'datasets': {
            'compile_codebleu': compile_dataset,
            'pass_at_k': pass_dataset,
        },
        'num_samples': {
            'compile_codebleu': num_samples,
            'pass_at_k': pass_num_samples,
            'generation_batch_size': generation_batch_size,
            'eval_max_new_tokens': eval_max_new_tokens,
            'compile_mode': compile_mode,
        },
        'predictions': {
            'compile_codebleu': compile_predictions_file,
            'pass_at_k': pass_predictions_file,
        },
        'provenance': {
            'compile_codebleu': compile_predictions_file + '.provenance.json',
            'pass_at_k': pass_predictions_file + '.provenance.json',
            'decoder_revision': decoder_revision or None,
            'encoder_revision': encoder_revision or None,
            'seed': seed,
        },
        'codebleu': extract_json(codebleu_out),
        'codebleu_compiled_only': extract_json(codebleu_compiled_out),
        'compile_at_k': extract_json(compile_out),
        'pass_at_k': extract_json(pass_out.stdout)
    }
    
    suffix = "_grpo" if use_grpo else ""
    with open(results_dir / f"{cfg['name']}{suffix}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    summary_file = results_dir / f"{cfg['name']}{suffix}.json"
    print(f"Results summary saved to {summary_file}")

    # Compile statistical candidate-level metrics to CSV
    csv_output_file = results_dir / f"{cfg['name']}{suffix}_compile_stats.csv"
    print(f"\nCompiling candidate-level metrics to CSV: {csv_output_file} ...")
    subprocess.run([
        sys.executable, str(msc_code_root / 'scripts/evaluation/compile_statistical_results_antigravity.py'),
        '--predictions', compile_predictions_file,
        '--output', str(csv_output_file),
        '--workers', eval_metric_workers,
        '--compile_mode', compile_mode,
    ], env=env, check=True)

    pass_csv_output_file = results_dir / f"{cfg['name']}{suffix}_pass_stats.csv"
    if shared_eval_pool:
        print(f"\nReusing aligned candidate-level statistics: {pass_csv_output_file}")
        shutil.copy2(csv_output_file, pass_csv_output_file)
    else:
        print(f"\nCompiling pass-task candidate-level metrics to CSV: {pass_csv_output_file} ...")
        subprocess.run([
            sys.executable, str(msc_code_root / 'scripts/evaluation/compile_statistical_results_antigravity.py'),
            '--predictions', pass_predictions_file,
            '--output', str(pass_csv_output_file),
            '--workers', eval_metric_workers,
            '--compile_mode', compile_mode,
        ], env=env, check=True)

    if hf_repo:
        result_files = [
            (compile_predictions_file, f"results/{Path(compile_predictions_file).name}"),
            (pass_predictions_file, f"results/{Path(pass_predictions_file).name}"),
            (summary_file, f"results/sweeps_antigravity/{summary_file.name}"),
            (csv_output_file, f"results/sweeps_antigravity/{csv_output_file.name}"),
            (pass_csv_output_file, f"results/sweeps_antigravity/{pass_csv_output_file.name}"),
        ]
        for local_file, repo_path in result_files:
            upload_to_huggingface(
                hf_repo,
                local_file,
                repo_path,
                token=hf_token,
                private=hf_private,
                commit_message=f"Upload results for {cfg['name']}{suffix}",
            )


def run_stages_for_cfg(cfg, args):
    """Run the requested training stage(s) for a single experiment config.

    With --chain, runs SFT first then GRPO (the GRPO stage auto-loads the SFT
    checkpoint from artifacts/{name}/pytorch_model.bin). Otherwise runs a single
    stage (GRPO if --use_grpo, else SFT). Completed stages are skipped based on
    their results summary file.
    """
    if args.chain:
        stages = [False, True]  # SFT, then GRPO
    else:
        stages = [args.use_grpo]

    workspace_root = Path(__file__).resolve().parent.parent
    for use_grpo in stages:
        suffix = "_grpo" if use_grpo else ""
        results_file = workspace_root / "results" / "sweeps_antigravity" / f"{cfg['name']}{suffix}.json"
        if results_file.exists() and not args.force_rerun:
            print(f"Skipping completed stage: {cfg['name']}{suffix}")
            continue
        stage_name = "GRPO" if use_grpo else "SFT"
        print(f"\n>>> Stage: {stage_name} for {cfg['name']}")
        run_experiment(
            cfg,
            limit_eval=args.limit_eval,
            num_samples=args.num_samples,
            pass_num_samples=args.pass_num_samples,
            generation_batch_size=args.generation_batch_size,
            eval_max_new_tokens=args.eval_max_new_tokens,
            dry_run=args.dry_run,
            use_grpo=use_grpo,
            use_local=args.local,
            num_gpus=args.num_gpus,
            train_file=args.train_file,
            eval_file=args.eval_file,
            compile_dataset=args.compile_dataset,
            pass_dataset=args.pass_dataset,
            grpo_train_file=args.grpo_train_file,
            train_batch_size=args.train_batch_size,
            grad_accum_override=args.grad_accum,
            load_4bit=args.load_4bit,
            grpo_group_size=args.grpo_group_size,
            grpo_epochs=args.grpo_epochs,
            grpo_test_timeout=args.grpo_test_timeout,
            grpo_max_new_tokens=args.grpo_max_new_tokens,
            grpo_reward_workers=args.grpo_reward_workers,
            metric_workers=args.metric_workers,
            compile_mode=args.compile_mode,
            sft_checkpoint=args.sft_checkpoint,
            grpo_checkpoint=args.grpo_checkpoint,
            sft_lr=args.sft_lr,
            grpo_lr=args.grpo_lr,
            grpo_perfect_bonus=args.grpo_perfect_bonus,
            grpo_reward_mode=args.grpo_reward_mode,
            grpo_verpo_alpha=args.grpo_verpo_alpha,
            grpo_verpo_anchor_weight=args.grpo_verpo_anchor_weight,
            grpo_verpo_density_norm=args.grpo_verpo_density_norm,
            grpo_binary_fail_reward=args.grpo_binary_fail_reward,
            grpo_no_compile_penalty=args.grpo_no_compile_penalty,
            grpo_compile_reward=args.grpo_compile_reward,
            grpo_partial_reward_cap=args.grpo_partial_reward_cap,
            grpo_perfect_base_reward=args.grpo_perfect_base_reward,
            grpo_overlap_weight=args.grpo_overlap_weight,
            grpo_unique_test_bonus=args.grpo_unique_test_bonus,
            grpo_duplicate_penalty=args.grpo_duplicate_penalty,
            grpo_kl_coef=args.grpo_kl_coef,
            grpo_clip_eps=args.grpo_clip_eps,
            grpo_entropy_coef=args.grpo_entropy_coef,
            grpo_gen_temperature=args.grpo_gen_temperature,
            grpo_gen_top_p=args.grpo_gen_top_p,
            grpo_adv_norm=args.grpo_adv_norm,
            grpo_min_reward_range=args.grpo_min_reward_range,
            grpo_passk_k=args.grpo_passk_k,
            grpo_score_chunk_size=args.grpo_score_chunk_size,
            grpo_loss_pooling=args.grpo_loss_pooling,
            grpo_simko_k=args.grpo_simko_k,
            grpo_overlong_filter=args.grpo_overlong_filter,
            grpo_clip_eps_high=args.grpo_clip_eps_high,
            grpo_reward_preflight_batches=args.grpo_reward_preflight_batches,
            qwen_prefix_tokens=args.qwen_prefix_tokens,
            qwen_prefix_gate_init=args.qwen_prefix_gate_init,
            qwen_prefix_gate_mode=args.qwen_prefix_gate_mode,
            qwen_prefix_dynamic=args.qwen_prefix_dynamic,
            qwen_prefix_min_tokens=args.qwen_prefix_min_tokens,
            qwen_prefix_tokens_per_log2=args.qwen_prefix_tokens_per_log2,
            qwen_prefix_gate_override=args.qwen_prefix_gate_override,
            qwen_prefix_rms_match=args.qwen_prefix_rms_match,
            decoder_prompt_max_length=args.decoder_prompt_max_length,
            prompt_fit_assembly=args.prompt_fit_assembly,
            auto_cfg=args.auto_cfg,
            prompt_assembly_mode=args.prompt_assembly_mode,
            prompt_clean_asm=args.prompt_clean_asm,
            max_block_instrs=args.max_block_instrs,
            dfg_mode=args.dfg_mode,
            edge_ablation=args.edge_ablation,
            gnn_ablation=args.gnn_ablation,
            gnn_layers=args.gnn_layers,
            global_attention_ablation=args.global_attention_ablation,
            region_compression=args.region_compression,
            region_max_blocks=args.region_max_blocks,
            block_pooling=args.block_pooling,
            block_vectors_per_block=args.block_vectors_per_block,
            add_reverse_edges=args.add_reverse_edges,
            block_position_mode=args.block_position_mode,
            position_scheme=args.position_scheme,
            causal_position_ids=args.causal_position_ids,
            grpo_train_graph_glue=args.grpo_train_graph_glue,
            use_reasoning=args.use_reasoning,
            attn_implementation=args.attn_implementation,
            gradient_checkpointing=args.gradient_checkpointing,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
            hf_private=bool(args.hf_private),
            hf_upload_checkpoints=bool(args.hf_upload_checkpoints),
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            skip_training=args.skip_training,
            skip_inference=args.skip_inference,
            quiet=bool(args.quiet),
            decoder_revision=args.decoder_revision,
            encoder_revision=args.encoder_revision,
            seed=args.seed,
            eval_graph_input_ablation=args.eval_graph_input_ablation,
            eval_graph_ablation_seed=args.eval_graph_ablation_seed,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_all', action='store_true', help="Run all sweeps matching filters")
    parser.add_argument('--experiment', default='', help="Run a specific experiment name from the generated list")
    parser.add_argument('--decoder', default='', help="Filter sweeps by decoder key (e.g. codet5-small, qwen-0.5b)")
    parser.add_argument('--mode', default='', help="Filter sweeps by tuning mode (e.g. lora_enc_dec, full_ft)")
    parser.add_argument('--max_risk', default='medium', choices=['low', 'medium', 'high', 'critical'], 
                        help="Filter sweeps by maximum allowed OOM risk (default: medium)")
    parser.add_argument('--limit_eval', type=int, default=0, help="Limit number of eval samples; 0 means all rows")
    parser.add_argument('--num_samples', type=int, default=5, help="Number of compile/CodeBLEU candidates per sample")
    parser.add_argument('--pass_num_samples', type=int, default=10, help="Number of pass@k candidates per sample")
    parser.add_argument('--generation_batch_size', type=int, default=5,
                        help="Generate candidates in chunks to reduce eval VRAM; total candidates still controlled by num_samples/pass_num_samples")
    parser.add_argument('--eval_max_new_tokens', type=int, default=768,
                        help="Max generated tokens per eval candidate; 768 matches SFT target length and avoids truncated uncompilable Dart")
    parser.add_argument('--decoder_revision', default='',
                        help="Immutable Hugging Face decoder revision/commit SHA. Required for confirmatory runs")
    parser.add_argument('--encoder_revision', default='',
                        help="Immutable Hugging Face encoder revision/commit SHA. Required for confirmatory runs")
    parser.add_argument('--seed', type=int, default=42,
                        help="Training and generation seed recorded in result provenance")
    parser.add_argument('--train_file', default=None,
                        help="SFT training JSONL. Comma-separated paths are allowed, e.g. data/datasets/dart_all.jsonl,data/testing/grpo_data.jsonl")
    parser.add_argument('--eval_file', default=None,
                        help="SFT eval JSONL. Comma-separated paths are allowed")
    parser.add_argument('--grpo_train_file', default=None,
                        help="GRPO training JSONL. Defaults to data/testing/grpo_data.jsonl inside the GRPO trainer")
    parser.add_argument('--compile_dataset', default='data/testing/compile-test2.jsonl',
                        help="Legacy compile/CodeBLEU task JSONL; old v1-v6 compile stats used 126 rows here")
    parser.add_argument('--pass_dataset', default='data/testing/grpo_data.jsonl',
                        help="Legacy pass@k task JSONL with unit tests; old pass@k used 154 rows here")
    parser.add_argument('--num_gpus', type=int, default=1,
                        help="GPUs for data-parallel SFT via DDP (torch.distributed.run spawns one "
                             "process per GPU; the HF Trainer all-reduces gradients). Effective batch "
                             "scales by num_gpus, so halve --grad_accum to keep it constant. GRPO is "
                             "single-GPU for now.")
    parser.add_argument('--hardware_profile', choices=['auto', 'h100', 'h200', 'rtx6000'], default='auto',
                        help="Convenience defaults for remote GPU class; explicit numeric flags still win")
    parser.add_argument('--train_batch_size', type=int, default=None,
                        help="Override GRAPH_BATCH_SIZE without setting env vars")
    parser.add_argument('--grad_accum', type=int, default=None,
                        help="Override GRAPH_GRAD_ACCUM without setting env vars")
    parser.add_argument('--load_4bit', type=int, choices=[0, 1], default=None,
                        help="Override GRAPH_LOAD_4BIT; 0 recommended on H100/H200")
    parser.add_argument('--grpo_group_size', type=int, default=None,
                        help="Override GRPO_GROUP_SIZE without setting env vars")
    parser.add_argument('--grpo_epochs', type=int, default=None,
                        help="Override GRPO_EPOCHS without setting env vars")
    parser.add_argument('--grpo_test_timeout', type=int, default=None,
                        help="Override GRPO_TEST_TIMEOUT seconds without setting env vars")
    parser.add_argument('--grpo_max_new_tokens', type=int, default=None,
                        help="Override GRPO_MAX_NEW_TOKENS without setting env vars")
    parser.add_argument('--grpo_reward_workers', type=int, default=None,
                        help="Override GRPO_REWARD_WORKERS without setting env vars")
    parser.add_argument('--metric_workers', type=int, default=None,
                        help="Parallel Dart workers for compile@k and per-candidate stats CSV metrics")
    parser.add_argument('--compile_mode', choices=['legacy', 'tests', 'jit_tests'], default='legacy',
                        help="Compile@k harness. Use jit_tests for pass-aligned functional evaluation; legacy is retained only for old 126-row comparisons")
    parser.add_argument('--sft_checkpoint', default='',
                        help="Checkpoint to load for SFT continuation/evaluation")
    parser.add_argument('--sft_lr', type=float, default=None,
                        help="Override GRAPH_LR for SFT without editing the generated sweep")
    parser.add_argument('--grpo_lr', type=float, default=None,
                        help="Override GRPO_LR without setting env vars")
    parser.add_argument('--grpo_perfect_bonus', type=float, default=None,
                        help="Override GRPO_PERFECT_BONUS without setting env vars")
    parser.add_argument('--grpo_reward_mode', choices=['shaped', 'binary', 'verpo'], default=None,
                        help="GRPO Dart reward mode. binary = full-pass reward only; shaped = static partial credit; "
                             "verpo = dense group-difficulty-weighted partial credit (breaks all-fail degenerate groups)")
    parser.add_argument('--grpo_verpo_alpha', type=float, default=None,
                        help="VeRPO test-difficulty coefficient alpha; paper default is 2.0")
    parser.add_argument('--grpo_verpo_anchor_weight', type=float, default=None,
                        help="VeRPO full-suite global-outcome anchor coefficient; paper default is 1.0")
    parser.add_argument('--grpo_verpo_density_norm', type=int, choices=[0, 1], default=None,
                        help="Enable VeRPO Gaussian-KDE test-density normalization; paper setting is 1")
    parser.add_argument('--grpo_binary_fail_reward', type=float, default=None,
                        help="Failure reward used when --grpo_reward_mode binary is active")
    parser.add_argument('--grpo_no_compile_penalty', type=float, default=None,
                        help="Reward for Dart candidates that fail to compile with the test harness")
    parser.add_argument('--grpo_compile_reward', type=float, default=None,
                        help="Reward for candidates that compile but pass zero tests")
    parser.add_argument('--grpo_partial_reward_cap', type=float, default=None,
                        help="Maximum additional shaped reward for partial pass_ratio before the full-pass bonus")
    parser.add_argument('--grpo_perfect_base_reward', type=float, default=None,
                        help="Base reward for fully passing all extracted tests before --grpo_perfect_bonus")
    parser.add_argument('--grpo_overlap_weight', type=float, default=None,
                        help="Optional token-overlap shaping weight for Dart GRPO rewards; default 0")
    parser.add_argument('--grpo_unique_test_bonus', type=float, default=None,
                        help="Group diversity bonus for tests passed uniquely by one sampled candidate")
    parser.add_argument('--grpo_duplicate_penalty', type=float, default=None,
                        help="Group diversity penalty for exact-normalized duplicate sampled candidates")
    parser.add_argument('--grpo_kl_coef', type=float, default=None,
                        help="Override GRPO_KL_COEF without setting env vars")
    parser.add_argument('--grpo_clip_eps', type=float, default=None,
                        help="Override GRPO_CLIP_EPS without setting env vars")
    parser.add_argument('--grpo_entropy_coef', type=float, default=None,
                        help="Optional entropy bonus coefficient for GRPO; default 0 because it adds full-vocab memory pressure")
    parser.add_argument('--grpo_gen_temperature', type=float, default=None,
                        help="GRPO rollout sampling temperature; scoring uses the same value. Default 0.7 matches eval")
    parser.add_argument('--grpo_gen_top_p', type=float, default=None,
                        help="GRPO rollout nucleus sampling top_p. Default 0.95 matches eval")
    parser.add_argument('--grpo_adv_norm', choices=['mean', 'std'], default=None,
                        help="GRPO advantage normalization. mean (default) avoids std blow-up of tiny reward gaps in small groups")
    parser.add_argument('--grpo_min_reward_range', type=float, default=None,
                        help="Skip GRPO groups whose reward max-min is at or below this; saves GPU time on no-signal batches")
    parser.add_argument('--grpo_passk_k', type=int, default=None,
                        help="If >1, GRPO optimizes pass@k: group advantages scaled by (1-p_hat)^(k-1) so reliably "
                             "solved prompts stop sharpening. Counteracts the Stage C pass@k collapse")
    parser.add_argument('--grpo_score_chunk_size', type=int, default=None,
                        help="Chunk the GRPO scoring forward+backward to this many samples (identical gradients, "
                             "memory bound by chunk not group size). Use 4 with --grpo_group_size 16+")
    parser.add_argument('--grpo_loss_pooling', choices=['token', 'seq'], default=None,
                        help="GRPO loss pooling: token (DAPO-style, historical) or seq (GSPO-style; short passes "
                             "no longer outweighed by long failures)")
    parser.add_argument('--grpo_simko_k', type=int, default=None,
                        help="Legacy experimental top-K smoothing, not faithful CaSP/SimKO. Keep 0 for confirmatory runs")
    parser.add_argument('--grpo_overlong_filter', type=int, choices=[0, 1], default=None,
                        help="Zero advantages of truncated (no-EOS) samples so truncation artifacts do not teach")
    parser.add_argument('--grpo_clip_eps_high', type=float, default=None,
                        help="DAPO clip-higher upper bound; inert in the single-update loop, plumbed for completeness")
    parser.add_argument('--grpo_reward_preflight_batches', type=int, default=None,
                        help="Generate and score N GRPO batches without optimizer updates, then exit")
    parser.add_argument('--qwen_prefix_tokens', type=int, default=None,
                        help="Override GRAPH_QWEN_PREFIX_TOKENS. Default is 16; use 0 to disable the upgraded Qwen glue")
    parser.add_argument('--qwen_prefix_gate_init', type=float, default=None,
                        help="Override GRAPH_QWEN_PREFIX_GATE_INIT. Default is 0.2")
    parser.add_argument('--qwen_prefix_gate_mode', choices=['scalar', 'token'], default=None,
                        help="Prefix gate granularity. token gives each learned prefix slot an independent gate; "
                             "scalar preserves historical checkpoints")
    parser.add_argument('--qwen_prefix_dynamic', type=int, choices=[0, 1], default=None,
                        help="Use a graph-size-dependent active prefix count, padded to --qwen_prefix_tokens")
    parser.add_argument('--qwen_prefix_min_tokens', type=int, default=None,
                        help="Minimum active slots when --qwen_prefix_dynamic=1 (default 4)")
    parser.add_argument('--qwen_prefix_tokens_per_log2', type=int, default=None,
                        help="Active slots per ceil(log2(block_count)) when dynamic (default 4)")
    parser.add_argument('--qwen_prefix_gate_override', type=float, default=None,
                        help="Force the learned Qwen graph prefix gate after loading a checkpoint. "
                             "Use for gate ablations on existing checkpoints; unlike gate_init, this "
                             "overrides the saved gate_logit")
    parser.add_argument('--qwen_prefix_rms_match', type=int, choices=[0, 1], default=None,
                        help="RMS-match prefix vectors to the decoder token-embedding scale (learnable "
                             "scalar) before the gate. REQUIRED for wide prefixes: 64/128 tokens "
                             "destabilized training without it (eval_loss rose after epoch 1). Train and "
                             "inference must use the same value. Sets GRAPH_QWEN_PREFIX_RMS_MATCH")
    parser.add_argument('--decoder_prompt_max_length', type=int, default=None,
                        help="Max decoder-side text prompt tokens for Qwen causal models. Default is 768")
    parser.add_argument('--prompt_fit_assembly', type=int, choices=[0, 1], default=None,
                        help="Trim the assembly middle so the prompt fits the token budget and keeps the 'Dart code:' cue. "
                             "Changes the prompt format: enable only for a NEW SFT run plus its downstream GRPO/eval")
    parser.add_argument('--auto_cfg', type=int, choices=[0, 1], default=None,
                        help="When a dataset row has no precomputed cfg/edges, extract a real basic-block CFG from the "
                             "assembly inline (GRAPH_AUTO_CFG). Default off keeps single-block behavior for reproducing "
                             "older checkpoints; prefer pointing at *_cfg.jsonl files instead")
    parser.add_argument('--prompt_assembly_mode', choices=['full', 'none', 'graph_only'], default=None,
                        help="Decoder prompt assembly channel: 'full' (default) includes raw assembly text; "
                             "'none'/'graph_only' withhold it so the decoder relies on the graph-prefix channel "
                             "(encoder-carries-assembly ablation). Sets GRAPH_PROMPT_ASSEMBLY_MODE")
    parser.add_argument('--prompt_clean_asm', type=int, choices=[0, 1], default=None,
                        help="Strip '<symbol+0xoffset>' annotations and ';'/'//' comments from the prompt assembly "
                             "(~25%% fewer tokens, zero info loss; graph channel unaffected). Sets GRAPH_PROMPT_CLEAN_ASM")
    parser.add_argument('--max_block_instrs', type=int, default=None,
                        help="Cap basic-block length (instructions) during inline CFG extraction so each block fits the "
                             "encoder's 512-token window; long blocks split into a fall-through chain. Sets GRAPH_MAX_BLOCK_INSTRS")
    parser.add_argument('--dfg_mode', choices=['legacy', 'off', 'edges'], default=None,
                        help="Dataflow channel (GRAPH_DFG_MODE). legacy (default): historical <unk> DFG token appendage "
                             "(dead channel, kept for old-checkpoint reproducibility). off: drop it and give the whole "
                             "512 window to assembly. edges: drop it AND add cross-block 'dataflow' edges (register "
                             "reaching-definitions, x86-64+ARM64) to the CFG graph for the GNN. Use 'edges' for new runs")
    parser.add_argument('--edge_ablation', choices=['full', 'none', 'cfg', 'dfg', 'shuffle'], default=None,
                        help="Controlled graph-topology ablation applied after graph construction. Keeps block encoder, "
                             "prefix adapter, token budget, and parameter count fixed while retaining all edges, no "
                             "edges, CFG only, DFG only, or a deterministic shuffled topology")
    parser.add_argument('--gnn_ablation', choices=['full', 'identity'], default=None,
                        help="Keep the same block encoder/prefix architecture but bypass GINE message passing. "
                             "Global block attention and learned prefix pooling remain active")
    parser.add_argument('--gnn_layers', type=int, default=None,
                        help="Number of GINE layers for newly trained controls (1-8, default 4). "
                             "Checkpoint construction must use the same value.")
    parser.add_argument('--global_attention_ablation', choices=['full', 'identity'], default=None,
                        help="Bypass global block self-attention while retaining its parameters, the block encoder, "
                             "projection, and learned prefix resampler.")
    parser.add_argument('--region_compression', choices=['off', 'linear_residual'], default=None,
                        help="Optionally attention-pool bounded straight-line CFG regions and fuse each summary "
                             "back into its member blocks before global attention.")
    parser.add_argument('--region_max_blocks', type=int, default=None,
                        help="Maximum blocks in one straight-line region (default 8). Checkpoint construction and "
                             "inference must use the same value.")
    parser.add_argument('--block_pooling', choices=['cls', 'multi_query'], default=None,
                        help="Local block representation: historical one-vector CLS pooling or learned multi-query "
                             "token pooling whose vectors survive into global attention.")
    parser.add_argument('--block_vectors_per_block', type=int, default=None,
                        help="Learned vectors retained per block in multi_query mode (2-8; default 4).")
    parser.add_argument('--add_reverse_edges', type=int, choices=[0, 1], default=None,
                        help="Add a separately typed reverse relation for every selected CFG/DFG edge so GINE can "
                             "propagate successor/use information backward. Sets GRAPH_ADD_REVERSE_EDGES")
    parser.add_argument('--block_position_mode', choices=['off', 'sinusoidal'], default=None,
                        help="Order signal before global block attention. sinusoidal prevents the graph-prefix "
                              "resampler from treating the canonical block sequence as an unordered set")
    parser.add_argument('--eval_graph_input_ablation',
                        choices=['none', 'cyclic_shift', 'shuffle_blocks'], default='none',
                        help="Inference-only graph-channel causal control. Does not alter the target prompt/tests.")
    parser.add_argument('--eval_graph_ablation_seed', type=int, default=None,
                        help="Deterministic seed for --eval_graph_input_ablation; defaults to --seed.")
    parser.add_argument('--position_scheme', choices=['legacy', 'roberta'], default=None,
                        help="GraphCodeBERT block position ids (GRAPH_POSITION_SCHEME). legacy: historical 0-based "
                             "(positions 0/1 are untrained in the pretrained encoder). roberta: pretraining-faithful "
                             "(code from 2, pads at 1). Use 'roberta' for new runs, mandatory with a frozen encoder")
    parser.add_argument('--causal_position_ids', choices=['cumsum', 'arange'], default=None,
                        help="SFT decoder RoPE positions (GRAPH_CAUSAL_POSITION_IDS). cumsum (default): match "
                             "generate()/GRPO's attention-mask-cumsum convention. arange: legacy training convention "
                             "of pre-fix checkpoints (target positions inflated past the padded prompt)")
    parser.add_argument('--grpo_train_graph_glue', type=int, choices=[0, 1], default=None,
                        help="1 (default): GRPO policy gradient also trains the graph glue (GNN, projection, prefix "
                             "adapter). 0: freeze the glue during GRPO (pre-fix behavior). Sets GRPO_TRAIN_GRAPH_GLUE")
    parser.add_argument('--use_reasoning', type=int, choices=[0, 1], default=None,
                        help="Include dataset reasoning in the decoder text prompt. Default is 0; keep it off for eval-matched training")
    parser.add_argument('--attn_implementation', default=None,
                        help="Decoder attention backend. Examples: auto, flash_attention_4, flash_attention_3, flash_attention_2, sdpa, flex_attention, eager, or a HF kernels id")
    parser.add_argument('--gradient_checkpointing', type=int, choices=[0, 1], default=None,
                        help="Enable decoder gradient checkpointing for lower VRAM at the cost of speed")
    parser.add_argument('--hf_repo', default='',
                        help="Optional Hugging Face repo id (username/repo) for automatic checkpoint/result uploads")
    parser.add_argument('--hf_token', default='',
                        help="Optional Hugging Face token. Prefer HF_TOKEN env var so the token is not stored in shell history")
    parser.add_argument('--hf_private', type=int, choices=[0, 1], default=1,
                        help="Create/use the HF repo as private by default")
    parser.add_argument('--hf_upload_checkpoints', type=int, choices=[0, 1], default=1,
                        help="When --hf_repo is set, upload saved training checkpoints from the trainer callback")
    parser.add_argument('--save_strategy', choices=['no', 'steps', 'epoch'], default=None,
                        help="Trainer save strategy. With --hf_repo this defaults to epoch unless set explicitly")
    parser.add_argument('--save_steps', type=int, default=None,
                        help="Save every N update steps when --save_strategy steps")
    parser.add_argument('--save_total_limit', type=int, default=2,
                        help="Limit local trainer checkpoints when saving periodically")
    parser.add_argument('--grpo_checkpoint', default='',
                        help="Checkpoint to load for GRPO. Use this to continue from a previous GRPO artifact instead of the SFT artifact")
    parser.add_argument('--quiet', type=int, choices=[0, 1], default=1,
                        help="1 (default): suppress framework warnings/progress bars in all child processes. "
                             "Errors always stay visible. 0: show everything")
    parser.add_argument('--force_rerun', action='store_true',
                        help="Run the selected stage even if its sweeps summary JSON already exists")
    parser.add_argument('--skip_training', action='store_true',
                        help="Skip training and only run inference/metrics for the existing checkpoint")
    parser.add_argument('--skip_inference', action='store_true',
                        help="Reuse existing *_predictions.json files and only recompute metrics/summary/stats. "
                             "Combine with --skip_training to resume after a metrics-stage crash")
    parser.add_argument('--dry_run', action='store_true', help="Run only 2 steps of training and 2 eval samples to verify correctness")
    parser.add_argument('--use_grpo', action='store_true', help="Use GRPO Reinforcement Learning instead of standard SFT")
    parser.add_argument('--chain', action='store_true', help="Chain SFT then GRPO for each experiment (SFT checkpoint is auto-loaded by the GRPO stage)")
    parser.add_argument('--local', action='store_true', help="Run sweep locally using RTX 3060 GPU settings, epochs=3, larger LRs")
    parser.add_argument('--epochs', type=int, default=None, help="Override training epochs for all selected sweeps")
    parser.add_argument('--lora_r', type=int, default=None, help="Override LoRA rank (r) for all selected sweeps")
    parser.add_argument('--lora_alpha', type=int, default=None, help="Override LoRA alpha for all selected sweeps")
    parser.add_argument('--name_suffix', default='',
                        help="Append a suffix to selected experiment names, e.g. _ut, to avoid overwriting artifacts")
    parser.add_argument('--encoder', default='', help="Encoder model: short key (gcb, clap) or full HF id (overrides GRAPH_ENCODER_MODEL)")
    args = parser.parse_args()

    if args.region_max_blocks is not None and args.region_max_blocks < 1:
        parser.error('--region_max_blocks must be a positive integer')
    if (
        args.block_vectors_per_block is not None
        and not 2 <= args.block_vectors_per_block <= 8
    ):
        parser.error('--block_vectors_per_block must be between 2 and 8')

    if args.hardware_profile in {'h200', 'rtx6000'}:
        if args.grpo_group_size is None:
            args.grpo_group_size = 8
        if args.load_4bit is None:
            args.load_4bit = 0
    elif args.hardware_profile == 'h100':
        if args.grpo_group_size is None:
            args.grpo_group_size = 4
        if args.load_4bit is None:
            args.load_4bit = 0

    # Resolve the encoder. --encoder takes precedence over GRAPH_ENCODER_MODEL.
    ENCODER_ALIASES = {
        'gcb': 'microsoft/graphcodebert-base',
        'graphcodebert': 'microsoft/graphcodebert-base',
        'clap': 'hustcw/clap-asm',
        'clap-asm': 'hustcw/clap-asm',
    }
    if args.encoder:
        encoder_model = ENCODER_ALIASES.get(args.encoder.lower(), args.encoder)
        os.environ['GRAPH_ENCODER_MODEL'] = encoder_model
    else:
        encoder_model = os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base")

    # Resolve the encoder suffix to differentiate results
    encoder_suffix = "_gcb"
    if "clap-asm" in encoder_model.lower():
        encoder_suffix = "_clap"
    
    selection_aliases = {id(cfg): {cfg['name']} for cfg in SWEEPS}
    for cfg in SWEEPS:
        cfg['name'] = cfg['name'] + encoder_suffix
        selection_aliases[id(cfg)].add(cfg['name'])

    # Map parameters if running locally
    if args.local:
        print("Local mode enabled. Overriding hyperparameters for local execution:")
        print("  - Mapped LRs: 1e-5 -> 1e-4, 5e-6 -> 5e-5")
        print("  - Mapped Epochs: 1 -> 3")
        print("  - Naming suffix: _local")
        for cfg in SWEEPS:
            if cfg['lr'] == '1e-5':
                cfg['lr'] = '1e-4'
            elif cfg['lr'] == '5e-6':
                cfg['lr'] = '5e-5'
            cfg['epochs'] = '3'
            cfg['name'] = cfg['name'].replace('1e5', '1e4').replace('5e6', '5e5') + "_local"

    # CLI hyperparameter overrides (take precedence over defaults and --local).
    if args.epochs is not None or args.lora_r is not None or args.lora_alpha is not None:
        overrides = []
        if args.epochs is not None:
            overrides.append(f"epochs={args.epochs}")
        if args.lora_r is not None:
            overrides.append(f"lora_r={args.lora_r}")
        if args.lora_alpha is not None:
            overrides.append(f"lora_alpha={args.lora_alpha}")
        print(f"CLI overrides applied to all sweeps: {', '.join(overrides)}")
        for cfg in SWEEPS:
            peft_named = any(tag in cfg['name'] for tag in ['lora', 'dora'])
            if args.epochs is not None:
                cfg['epochs'] = str(args.epochs)
            if args.lora_r is not None:
                cfg['lora_r'] = str(args.lora_r)
                if peft_named:
                    cfg['name'] = re.sub(r'_r\d+_', f"_r{args.lora_r}_", cfg['name'])
            if args.lora_alpha is not None:
                cfg['lora_alpha'] = str(args.lora_alpha)
                if peft_named and f"_a{args.lora_alpha}" not in cfg['name']:
                    cfg['name'] = f"{cfg['name']}_a{args.lora_alpha}"

    if args.name_suffix:
        for cfg in SWEEPS:
            cfg['name'] = f"{cfg['name']}{args.name_suffix}"

    # Apply filters
    filtered_sweeps = []
    for cfg in SWEEPS:
        if args.decoder and args.decoder not in cfg['name']:
            continue
        if args.mode and args.mode not in cfg['name']:
            continue
            
        # Risk ordering: low < medium < high < critical
        risk_levels = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        cfg_risk = risk_levels.get(cfg['oom_risk'], 0)
        max_allowed = risk_levels.get(args.max_risk, 1)
        
        if cfg_risk > max_allowed:
            continue
            
        filtered_sweeps.append(cfg)

    if args.run_all:
        if not filtered_sweeps:
            print("No experiments matched the specified filters and risk tolerance.")
            return
            
        print(f"Starting execution of {len(filtered_sweeps)} experiments...")
        for cfg in filtered_sweeps:
            try:
                run_stages_for_cfg(cfg, args)
            except Exception as e:
                print(f"Experiment {cfg['name']} failed with error: {e}")
                
    elif args.experiment:
        target_names = {args.experiment}
        if args.name_suffix and not args.experiment.endswith(args.name_suffix):
            target_names.add(f"{args.experiment}{args.name_suffix}")
        target = []
        for candidate in SWEEPS:
            aliases = set(selection_aliases[id(candidate)])
            aliases.add(candidate['name'])
            if args.name_suffix and candidate['name'].endswith(args.name_suffix):
                aliases.add(candidate['name'][:-len(args.name_suffix)])
            if aliases & target_names:
                target.append(candidate)
        if target:
            run_stages_for_cfg(target[0], args)
        else:
            print(f"Experiment config '{args.experiment}' not found in the list.")
            
    else:
        # Print list of matched experiments
        print(f"Available sweep experiments matching filter (max_risk={args.max_risk.upper()}):")
        for idx, cfg in enumerate(filtered_sweeps):
            print(f" {idx+1:2d}. {cfg['name']:50s} [Risk: {cfg['oom_risk'].upper()}]")
        print(f"\nMatched {len(filtered_sweeps)} of {len(SWEEPS)} total sweeps.")
        print("Run with --run_all to run these, or --experiment <name> to run a specific one.")
        print("Use --max_risk [low/medium/high/critical] to change VRAM safety margins.")
        print("Use --dry_run to test a configuration before launching a full training run.")


if __name__ == '__main__':
    main()
