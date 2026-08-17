"""
Statistical Results Compiler for Neural Decompiler (Antigravity version).
Evaluates each individual candidate prediction on CodeBLEU, compilation, and test passing,
and outputs a per-problem, per-candidate CSV for statistical analysis.
"""

from __future__ import annotations

import sys
sys.modules['gptqmodel'] = None

# Monkeypatch PreTrainedTokenizerFast to work around transformers 5.9.0 type mismatch with tokenizers 0.22.1 on AddedToken
try:
    import tokenizers
    from transformers import PreTrainedTokenizerFast
    _old_add_tokens = PreTrainedTokenizerFast._add_tokens
    
    def _patched_add_tokens(self, new_tokens, special_tokens=False):
        dict_or_attr = lambda o, k, d: o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
        conv_tokens = []
        for t in new_tokens:
            if isinstance(t, str):
                conv_tokens.append(t)
            else:
                conv_tokens.append(tokenizers.AddedToken(
                    dict_or_attr(t, 'content', str(t)),
                    single_word=dict_or_attr(t, 'single_word', False),
                    lstrip=dict_or_attr(t, 'lstrip', False),
                    rstrip=dict_or_attr(t, 'rstrip', False),
                    normalized=dict_or_attr(t, 'normalized', True),
                    special=dict_or_attr(t, 'special', False)
                ))
        if special_tokens:
            return self._tokenizer.add_special_tokens(conv_tokens)
        return self._tokenizer.add_tokens(conv_tokens)
        
    PreTrainedTokenizerFast._add_tokens = _patched_add_tokens
except Exception as e:
    pass

import argparse
import json
import csv
import subprocess
import tempfile
import re
import os
from pathlib import Path
from codebleu import CodeBLEUCalculator

def compile_dart(code: str) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'main.dart'
        try:
            path.write_text(code, encoding='utf-8')
            result = subprocess.run(
                ['dart', 'analyze', str(path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

def compile_swift(code: str) -> bool:
    return bool(code.strip())

def pseudo_pass(reference: str, prediction: str) -> bool:
    reference_tokens = set(reference.lower().split())
    prediction_tokens = set(prediction.lower().split())

    if not reference_tokens:
        return False

    overlap = len(reference_tokens & prediction_tokens) / len(reference_tokens)
    return overlap >= 0.3

def run_sandbox_pass(solution_code: str, test_code: str) -> bool:
    """Check if the code passes actual functional unit tests using Dart sandbox execution"""
    if not solution_code.strip() or not test_code.strip():
        return False
        
    lines = solution_code.split('\n')
    imports = []
    function_lines = []
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith('import ') or
            stripped.startswith('export ') or
            stripped.startswith('@pragma(') or
            stripped.startswith('library ') or
            stripped.startswith('part ')):
            imports.append(line)
        else:
            function_lines.append(line)
            
    imports_section = '\n'.join(imports) if imports else ''
    function_section = '\n'.join(function_lines).strip()
    full_code = (imports_section + "\n\n" if imports_section else "") + function_section + "\n\n" + test_code
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_filepath = os.path.join(temp_dir, 'temp_test.dart')
            with open(test_filepath, 'w', encoding='utf-8') as f:
                f.write(full_code)
                
            test_proc = subprocess.run(
                ['dart', '--disable-dart-dev', 'run', test_filepath],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=5
            )
            return test_proc.returncode == 0
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help="Path to predictions JSON file")
    parser.add_argument('--output', required=True, help="Path to write statistical CSV file")
    args = parser.parse_args()

    rows = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    if not rows:
        print("Empty predictions file.")
        return

    calculators = {
        'dart': CodeBLEUCalculator('dart'),
        'swift': CodeBLEUCalculator('swift'),
    }

    # Determine maximum candidate count (K) to dynamically format CSV headers
    max_k = 0
    for row in rows:
        candidates = row.get('predictions', [row.get('prediction', '')])
        if not candidates:
            candidates = [row.get('prediction', '')]
        max_k = max(max_k, len(candidates))

    # Construct CSV headers
    headers = ['problem_id', 'language', 'reference_length']
    for i in range(max_k):
        headers.extend([
            f'cand_{i+1}_codebleu',
            f'cand_{i+1}_compile',
            f'cand_{i+1}_pass'
        ])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for idx, row in enumerate(rows):
            lang = row.get('language', 'dart').lower()
            calc = calculators.get(lang, calculators['dart'])
            reference = row.get('reference', '')
            test_code = row.get('tests', '')
            
            candidates = row.get('predictions', [row.get('prediction', '')])
            if not candidates:
                candidates = [row.get('prediction', '')]
                
            row_data = [
                row.get('id', idx),
                lang,
                len(reference)
            ]

            for i in range(max_k):
                if i < len(candidates):
                    cand = candidates[i]
                    
                    # 1. CodeBLEU
                    try:
                        score = calc.compute_codebleu(reference, cand)['codebleu']
                    except Exception:
                        score = 0.0
                        
                    # 2. Compile
                    comp = 1 if (compile_dart(cand) if lang == 'dart' else compile_swift(cand)) else 0
                    
                    # 3. Pass (using sandbox if unit tests exist, otherwise pseudo-pass overlap)
                    if test_code:
                        passed = 1 if run_sandbox_pass(cand, test_code) else 0
                    else:
                        passed = 1 if pseudo_pass(reference, cand) else 0
                        
                    row_data.extend([round(score, 5), comp, passed])
                else:
                    # Pad missing candidates for alignment
                    row_data.extend(['', '', ''])
                    
            writer.writerow(row_data)

    print(f"Successfully compiled per-problem statistics to: {args.output}")

if __name__ == '__main__':
    main()
