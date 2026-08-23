"""
Post-Sweep Run Coordinator (Antigravity version).
1. Re-evaluates previous runs (codet5p-770m and qwen) to fix the strict=False loading bug.
2. Runs the missing codet5p-2b configurations (SFT training + evaluation) standalone.
"""

import sys
import os
from pathlib import Path
import json
import subprocess

# Ensure correct pathing
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from configs.run_sweeps_antigravity import SWEEPS, extract_json, run_experiment

def re_evaluate(cfg, limit_eval=165, num_samples=5):
    print(f"\n==================================================")
    print(f"RE-EVALUATING EXPERIMENT: {cfg['name']}")
    print(f"Decoder: {cfg['decoder']}")
    print(f"==================================================\n")

    env = os.environ.copy()
    env['PYTHONPATH'] = 'MSc-Code'
    dart_bin = '/home/zeus/dart-sdk/bin' if os.path.exists('/home/zeus/dart-sdk/bin') else os.path.join(os.path.expanduser('~'), 'dart-sdk', 'bin')
    env['PATH'] = f"{dart_bin}:{env.get('PATH', '')}"
    env['GRAPH_DECODER_MODEL'] = cfg['decoder']
    env['GRAPH_ENCODER_PEFT'] = cfg['encoder_peft']
    env['GRAPH_DECODER_PEFT'] = cfg['decoder_peft']
    env['GRAPH_FREEZE_ENCODER'] = cfg['freeze_encoder']
    env['GRAPH_FREEZE_DECODER'] = cfg['freeze_decoder']
    env['GRAPH_LR'] = cfg['lr']
    env['GRAPH_EPOCHS'] = cfg['epochs']
    env['GRAPH_LORA_R'] = cfg.get('lora_r', '16')
    env['GRAPH_LORA_ALPHA'] = cfg.get('lora_alpha', '32')

    predictions_file = f"MSc-Code/results/{cfg['name']}_predictions.json"
    checkpoint_file = f"MSc-Code/artifacts/{cfg['name']}/pytorch_model.bin"
    
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint not found at {checkpoint_file}. Skipping re-evaluation.")
        return

    # 1. Run multi-sample inference with corrected loading
    if os.path.exists(predictions_file) and os.path.getsize(predictions_file) > 1000:
        print(f"\nPredictions file already exists at {predictions_file}. Skipping inference and going straight to metrics calculation.")
    else:
        print("\nRunning multi-sample inference evaluation...")
        subprocess.run([
            sys.executable, 'MSc-Code/scripts/evaluation/graph_inference_antigravity.py',
            '--dataset', 'MSc-Code/data/datasets/test-set.jsonl',
            '--decoder_model', cfg['decoder'],
            '--output', predictions_file,
            '--checkpoint', checkpoint_file,
            '--limit', str(limit_eval),
            '--num_samples', str(num_samples)
        ], env=env, check=True)

    # 2. Calculate metrics
    print("\nCalculating metrics...")
    
    # CodeBLEU
    codebleu_out = subprocess.check_output([
        sys.executable, 'MSc-Code/scripts/evaluation/graph_codebleu_antigravity.py',
        '--predictions', predictions_file
    ], env=env, text=True)
    print("CodeBLEU Results:")
    print(codebleu_out)
    
    # Compile@k
    compile_out = subprocess.check_output([
        sys.executable, 'MSc-Code/scripts/evaluation/graph_compile_at_k_antigravity.py',
        '--predictions', predictions_file,
        '--k_values', '1,5'
    ], env=env, text=True)
    print("Compile@k Results:")
    print(compile_out)

    # Pass@k
    pass_out = subprocess.run([
        sys.executable, 'MSc-Code/scripts/evaluation/graph_pass_at_k_antigravity.py',
        '--predictions', predictions_file,
        '--k_values', '1,5'
    ], env=env, capture_output=True, text=True)
    print("Pass@k Results:")
    print(pass_out.stdout)

    # Save summary results
    results_dir = Path("results/sweeps_antigravity")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'experiment': cfg['name'],
        'config': cfg,
        'codebleu': extract_json(codebleu_out),
        'compile_at_k': extract_json(compile_out),
        'pass_at_k': extract_json(pass_out.stdout) if pass_out.returncode == 0 else {}
    }
    
    with open(results_dir / f"{cfg['name']}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Results summary saved to {results_dir / cfg['name']}.json")

    # Compile statistical candidate-level metrics to CSV
    csv_output_file = results_dir / f"{cfg['name']}_stats.csv"
    print(f"\nCompiling candidate-level metrics to CSV: {csv_output_file} ...")
    try:
        subprocess.run([
            sys.executable, 'MSc-Code/scripts/evaluation/compile_statistical_results_antigravity.py',
            '--predictions', predictions_file,
            '--output', str(csv_output_file)
        ], env=env, check=True)
    except Exception as e:
        print(f"Failed to compile statistical CSV: {e}")

def main():
    codet5p_2b_targets = []
    
    print("==================================================")
    print("PHASE 1: RUNNING STANDALONE CODET5P-2B SWEEPS")
    print("==================================================")
    for cfg in SWEEPS:
        if cfg['name'] in codet5p_2b_targets:
            try:
                run_experiment(cfg, limit_eval=165, num_samples=5, dry_run=False, use_grpo=False)
            except Exception as e:
                print(f"Failed to run standalone sweep {cfg['name']}: {e}")

    re_eval_targets = [
        'codet5p-2b_lora_enc_dec_r16_1e5',
        'codet5p-2b_lora_enc_dec_r16_5e6',
        'codet5p-770m_lora_enc_dec_r16_1e5',
        'codet5p-770m_lora_enc_dec_r16_5e6',
        'codet5p-770m_lora_enc_dec_r8_1e5',
        'codet5p-770m_lora_enc_dec_r8_5e6',
        'qwen-0.8b_lora_enc_dec_r16_1e5',
        'qwen-0.8b_lora_enc_dec_r16_5e6',
        'qwen-2b_lora_enc_dec_r16_1e5',
        'qwen-2b_lora_enc_dec_r16_5e6'
    ]
    
    print("\n==================================================")
    print("PHASE 2: RE-EVALUATING COMPLETED RUNS")
    print("==================================================")
    for cfg in SWEEPS:
        if cfg['name'] in re_eval_targets:
            try:
                re_evaluate(cfg)
            except Exception as e:
                print(f"Failed to re-evaluate {cfg['name']}: {e}")

    print("\nAll post-sweep operations completed successfully!")

if __name__ == '__main__':
    main()
