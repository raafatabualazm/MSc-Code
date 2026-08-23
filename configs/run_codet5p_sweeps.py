
import os
import subprocess
from pathlib import Path

configs = [
    {
        'name': 'codet5p_770m_freeze_5e6',
        'decoder': 'Salesforce/codet5p-770m',
        'freeze': '1',
        'lr': '5e-6',
    },
    {
        'name': 'codet5p_770m_freeze_1e6',
        'decoder': 'Salesforce/codet5p-770m',
        'freeze': '1',
        'lr': '1e-6',
    },
]

base = Path('results/sweeps_codet5p')
base.mkdir(parents=True, exist_ok=True)

for cfg in configs:
    env = os.environ.copy()
    env['PYTHONPATH'] = 'MSc-Code'
    env['GRAPH_DECODER_MODEL'] = cfg['decoder']
    env['GRAPH_FREEZE_ENCODER'] = cfg['freeze']
    env['GRAPH_LR'] = cfg['lr']
    env['GRAPH_EPOCHS'] = '1'
    env['GRAPH_OUTPUT_DIR'] = f"artifacts/{cfg['name']}"

    subprocess.run([
        'python', '-m', 'scripts.training.graph_encoder_decoder_decompiler_v2'
    ], cwd='MSc-Code', env=env, check=True)

    pred = f"MSc-Code/results/{cfg['name']}_predictions.json"

    subprocess.run([
        'python', 'MSc-Code/scripts/evaluation/graph_inference.py',
        '--dataset', 'MSc-Code/data/datasets/test-set.jsonl',
        '--output', pred,
        '--checkpoint', f"MSc-Code/artifacts/{cfg['name']}/pytorch_model.bin"
    ], check=True)

    with open(base / f"{cfg['name']}.txt", 'w', encoding='utf-8') as handle:
        for cmd in [
            ['python', 'MSc-Code/scripts/evaluation/graph_codebleu.py', '--predictions', pred],
            ['python', 'MSc-Code/scripts/evaluation/graph_compile_at_k.py', '--predictions', pred],
            ['python', 'MSc-Code/scripts/evaluation/graph_pass_at_k.py', '--predictions', pred],
        ]:
            result = subprocess.check_output(cmd, text=True)
            handle.write(result)
            handle.write('\n')

print('done')
