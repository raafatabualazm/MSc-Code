
import itertools
import json
import os
import subprocess
from pathlib import Path

DECODERS = [
    't5-small',
    'Salesforce/codet5-base',
    'Salesforce/codet5p-770m',
]

MODES = {
    'freeze_encoder': {'freeze_encoder': '1'},
    'full_ft': {'freeze_encoder': '0'},
    'freeze_decoder_encoder_ft': {'freeze_encoder': '0'},
}

LRS = ['1e-5', '5e-6', '1e-6']

results_dir = Path('results/full_systematic_sweep')
results_dir.mkdir(parents=True, exist_ok=True)

all_results = []

for decoder, (mode_name, mode_cfg), lr in itertools.product(DECODERS, MODES.items(), LRS):
    exp_name = f"{decoder.split('/')[-1]}_{mode_name}_{lr.replace('-', '')}"

    env = os.environ.copy()
    env['PYTHONPATH'] = 'MSc-Code'
    env['GRAPH_DECODER_MODEL'] = decoder
    env['GRAPH_FREEZE_ENCODER'] = mode_cfg['freeze_encoder']
    env['GRAPH_LR'] = lr
    env['GRAPH_EPOCHS'] = '1'
    env['GRAPH_OUTPUT_DIR'] = f'artifacts/{exp_name}'

    print(f'=== RUNNING {exp_name} ===')

    try:
        subprocess.run(
            ['python', '-m', 'scripts.training.graph_encoder_decoder_decompiler_v2'],
            cwd='MSc-Code',
            env=env,
            check=True,
        )

        prediction_file = f'MSc-Code/results/{exp_name}_predictions.json'

        subprocess.run([
            'python',
            'MSc-Code/scripts/evaluation/graph_inference.py',
            '--dataset',
            'MSc-Code/data/datasets/test-set.jsonl',
            '--output',
            prediction_file,
            '--checkpoint',
            f'MSc-Code/artifacts/{exp_name}/pytorch_model.bin'
        ], check=True)

        codebleu = subprocess.check_output([
            'python',
            'MSc-Code/scripts/evaluation/graph_codebleu.py',
            '--predictions',
            prediction_file,
        ], text=True)

        compile_score = subprocess.check_output([
            'python',
            'MSc-Code/scripts/evaluation/graph_compile_at_k.py',
            '--predictions',
            prediction_file,
        ], text=True)

        pass_score = subprocess.check_output([
            'python',
            'MSc-Code/scripts/evaluation/graph_pass_at_k.py',
            '--predictions',
            prediction_file,
        ], text=True)

        record = {
            'experiment': exp_name,
            'decoder': decoder,
            'mode': mode_name,
            'lr': lr,
            'codebleu_raw': codebleu,
            'compile_raw': compile_score,
            'pass_raw': pass_score,
        }

        all_results.append(record)

        with open(results_dir / f'{exp_name}.json', 'w', encoding='utf-8') as handle:
            json.dump(record, handle, indent=2)

    except Exception as exc:
        with open(results_dir / f'{exp_name}_FAILED.txt', 'w', encoding='utf-8') as handle:
            handle.write(str(exc))

with open(results_dir / 'summary.json', 'w', encoding='utf-8') as handle:
    json.dump(all_results, handle, indent=2)

print('ALL EXPERIMENTS FINISHED')
