"""
Graph-aware GRPO (Group Relative Policy Optimization) training script for neural decompilation (Antigravity version).
Optimized for low VRAM using PEFT adapter-disabling context to get reference model logits without parameter duplication.
"""

from __future__ import annotations

import sys
sys.modules['gptqmodel'] = None

# Quiet mode (default on): suppress framework warning spam so the per-step
# GRPO log stays readable. Errors are NOT suppressed on purpose - silent
# failures already produced fake zero-metric runs (FULL_HANDOFF Section 11,
# Problem 1) and a swallowed step error would save a no-op checkpoint and
# then burn eval GPU time on it. Set GRAPH_QUIET=0 to see everything.
import os
if os.environ.get("GRAPH_QUIET", "1") != "0":
    import warnings
    for _cat in (FutureWarning, UserWarning, DeprecationWarning):
        warnings.filterwarnings("ignore", category=_cat)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

import os
import re
import json
import math
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# How many `dart run` subprocesses to run concurrently when scoring rewards.
# The reward is dominated by cold-starting `dart run` per assertion; running
# them in a thread pool across the host's CPU cores turns a multi-day GRPO run
# into a few-hour one. Defaults to (cpu_count - 1), capped, override via env.
GRPO_REWARD_WORKERS = int(os.environ.get(
    "GRPO_REWARD_WORKERS",
    str(max(1, min(32, (os.cpu_count() or 4) - 1)))
))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from torch_geometric.data import Batch

try:
    from tqdm.auto import tqdm
except Exception:  # tqdm ships with transformers; a progress bar is never load-bearing
    tqdm = None
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    set_seed,
)

# Insert root to system path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.pyg_cfg_dataset import cfg_to_pyg
from models.hierarchical_graph_encoder_antigravity import LocalBlockEncoder, GraphPoolingEncoder
from scripts.data.dfg_extractor import LightweightDFGExtractor
from models.graphcodebert_tensor_builder import GraphCodeBERTTensorBuilder
from scripts.training.graph_positions import cumsum_position_ids  # shared with the SFT forward
from scripts.training.graph_encoder_decoder_decompiler_v2_antigravity import (
    GraphDecompilerConfig,
    GraphCodeBERTT5Seq2Seq,
    PROMPT_SCHEMA_VERSION,
    canonicalize_source,
    load_jsonl_many,
    maybe_override_qwen_prefix_gate,
    tokenize_dataset,
)
from scripts.evaluation.graph_compile_at_k_antigravity import evaluate_dart_jit_tests_detail
from scripts.training.hybrid_data_controls import assert_training_approved, verified_origin
from scripts.training.checkpoint_contract import validate_trainable_checkpoint_load
from scripts.provenance_antigravity import (
    file_record,
    git_state,
    graph_environment,
    model_commit,
    runtime_record,
    write_json,
)

ENCODER_MODEL = os.environ.get("GRAPH_ENCODER_MODEL", "microsoft/graphcodebert-base")


def compute_overlap_reward(reference: str, completion: str) -> float:
    ref_tokens = set(reference.lower().split())
    comp_tokens = set(completion.lower().split())
    if not ref_tokens:
        return 0.0
    overlap = len(ref_tokens & comp_tokens) / len(ref_tokens)
    return overlap

class TruePerTestReward:
    """Executable Dart reward with exact-binary and shaped-per-test modes.

    Binary mode runs the complete stored harness and is the confirmatory path.
    Shaped mode can execute individual assertions for diagnostics/partial
    credit. Both modes remove an emitted main(), resolve the harness candidate
    alias, and fail closed on missing or unextractable tests.
    """

    def __init__(self,
                 base_reward: float = -1.0,
                 pass_ratio_reward: float = 8.0,
                 perfect_bonus: float = 2.0,
                 enable_asserts: bool = False,
                 main_violation_penalty: float = -5.0,
                 empty_code_penalty: float = -3.0,
                 no_compile_penalty: float = -2.0,
                 compile_reward: float = -0.25,
                 partial_reward_cap: float = 2.0,
                 perfect_base_reward: float = 4.0,
                 binary_fail_reward: float = -1.0,
                 timeout: int = int(os.environ.get("GRPO_TEST_TIMEOUT", "5"))):
        self.base_reward = base_reward
        self.pass_ratio_reward = pass_ratio_reward
        self.perfect_bonus = float(os.environ.get("GRPO_PERFECT_BONUS", perfect_bonus))
        self.enable_asserts = enable_asserts
        self.main_violation_penalty = main_violation_penalty
        self.empty_code_penalty = empty_code_penalty
        self.no_compile_penalty = float(os.environ.get("GRPO_NO_COMPILE_PENALTY", no_compile_penalty))
        self.compile_reward = float(os.environ.get("GRPO_COMPILE_REWARD", compile_reward))
        self.partial_reward_cap = float(os.environ.get("GRPO_PARTIAL_REWARD_CAP", partial_reward_cap))
        self.perfect_base_reward = float(os.environ.get("GRPO_PERFECT_BASE_REWARD", perfect_base_reward))
        self.binary_fail_reward = float(os.environ.get("GRPO_BINARY_FAIL_REWARD", binary_fail_reward))
        self.reward_mode = os.environ.get("GRPO_REWARD_MODE", "shaped").strip().lower()
        if self.reward_mode not in {"shaped", "binary", "verpo"}:
            raise ValueError(f"Unsupported GRPO_REWARD_MODE={self.reward_mode!r}")
        # VeRPO (arxiv 2601.03525): group-relative test difficulty, Gaussian
        # density correction, and a binary full-suite anchor. In this single-turn
        # harness, centering their weighted sum is algebraically identical to
        # adding the paper's turn-level and trajectory-level advantages.
        self.verpo_alpha = float(os.environ.get("GRPO_VERPO_ALPHA", "2.0"))
        self.verpo_anchor_weight = float(
            os.environ.get("GRPO_VERPO_ANCHOR_WEIGHT", "1.0")
        )
        self.verpo_density_norm = os.environ.get("GRPO_VERPO_DENSITY_NORM", "1") == "1"
        if self.verpo_alpha <= 0.0:
            raise ValueError("GRPO_VERPO_ALPHA must be positive")
        if self.verpo_anchor_weight < 0.0:
            raise ValueError("GRPO_VERPO_ANCHOR_WEIGHT must be non-negative")
        self.timeout = timeout

    def _failure_reward(self) -> float:
        if self.reward_mode == "binary":
            return self.binary_fail_reward
        return self.no_compile_penalty

    def _evaluate_binary_harness(
        self, solution_code: str, test_code: str
    ) -> tuple[bool, bool, str]:
        """Score binary reward through the exact pass-aligned evaluator.

        Keeping a second extraction and execution path here caused subtle
        train/eval drift: for example, evaluation strips leading prose before
        Dart code while the old reward path did not. The shared evaluator is
        now the single authority for candidate extraction, source assembly,
        JIT/static-error classification, and execution.
        """
        compiled, passed, diagnostic, _source = evaluate_dart_jit_tests_detail(
            solution_code,
            test_code,
            "grpo_reward",
            timeout=self.timeout,
        )
        return bool(compiled), bool(passed), str(diagnostic or "")

    # --- structural guards -------------------------------------------------
    def _strip_strings_and_comments(self, s: str) -> str:
        s = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', '""', s, flags=re.S)
        s = re.sub(r'//.*$', '', s, flags=re.M)
        s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
        return s

    def _has_main_function(self, code: str) -> bool:
        clean = self._strip_strings_and_comments(code)
        pat = re.compile(
            r'^\s*(?:\w+(?:<[^>]*>)?\s+)?'          # optional type e.g. void / Future<void>
            r'main\s*\([^)]*\)\s*(?:async\s*)?\{',  # main(...) [async] {
            re.MULTILINE
        )
        return bool(pat.search(clean))

    def _extract_code_candidate(self, code: str) -> str:
        m = re.search(r"```(?:dart)?\s*(.*?)```", code, flags=re.S | re.I)
        if m:
            return m.group(1).strip()
        return code.strip()

    def _remove_main_function(self, code: str) -> str:
        """Remove top-level main() blocks before appending the unit-test main()."""
        code = self._extract_code_candidate(code)
        while True:
            main_match = re.search(
                r"^\s*(?:\w+(?:<[^>]*>)?\s+)?main\s*\([^)]*\)\s*(?:async\s*)?\{",
                code,
                flags=re.MULTILINE,
            )
            if not main_match:
                return code.strip()

            start = main_match.start()
            depth = 0
            i = main_match.end() - 1
            end = None
            while i < len(code):
                ch = code[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                i += 1

            if end is None:
                return code[:start].strip()
            code = (code[:start] + code[end:]).strip()

    def _prepare_solution_for_tests(self, code: str) -> str:
        code = self._remove_main_function(code)
        code = re.sub(r"^@pragma\(.*\)\s*$", "", code, flags=re.MULTILINE)
        return code.strip()

    def _dart_subprocess_env(self, workdir: str) -> Dict[str, str]:
        """Give Dart a writable, per-candidate home on ephemeral workers."""
        env = os.environ.copy()
        home = os.path.join(workdir, ".dart_home")
        appdata = os.path.join(home, "AppData", "Roaming")
        localappdata = os.path.join(home, "AppData", "Local")
        pub_cache = os.path.join(home, ".pub-cache")
        for path in (home, appdata, localappdata, pub_cache):
            os.makedirs(path, exist_ok=True)
        env.update({
            "HOME": home,
            "USERPROFILE": home,
            "APPDATA": appdata,
            "LOCALAPPDATA": localappdata,
            "PUB_CACHE": pub_cache,
            "CI": "true",
            "DART_SUPPRESS_ANALYTICS": "1",
        })
        return env

    # --- public entry point ------------------------------------------------
    def compute_reward_details(self, solution_code: str, test_code: str) -> Dict[str, Any]:
        if not shutil.which("dart"):
            raise RuntimeError("Dart is required for GRPO reward scoring but was not found on PATH")
        if not test_code:
            raise ValueError("GRPO reward row has no test harness")

        test_cases = self._extract_expect_calls_single_line(test_code)
        total_tests = len(test_cases)
        if total_tests == 0:
            raise ValueError(
                "GRPO reward extracted zero expect(candidate(...)) assertions; "
                "canonicalize or reject this training row"
            )

        # Binary reward must consume the RAW model completion through the same
        # extractor and JIT harness as evaluation. Do not pre-strip or rebuild
        # it through the shaped-reward helpers first.
        if self.reward_mode == "binary":
            compiled, full_pass, diagnostic = self._evaluate_binary_harness(
                solution_code, test_code
            )
            return {
                "reward": (
                    self.perfect_base_reward + self.perfect_bonus
                    if full_pass else self.binary_fail_reward
                ),
                "pass_ratio": 1.0 if full_pass else 0.0,
                "passed": total_tests if full_pass else 0,
                "total": total_tests,
                "compiled": compiled,
                "test_passes": [full_pass] * total_tests,
                "full_pass": full_pass,
                "status": "full_harness_pass" if full_pass else "full_harness_fail",
                "diagnostic": diagnostic,
            }

        solution_code = self._prepare_solution_for_tests(solution_code)
        if not solution_code.strip() or len(solution_code.strip()) < 10:
            return {
                "reward": self.binary_fail_reward if self.reward_mode == "binary" else self.empty_code_penalty,
                "pass_ratio": 0.0,
                "passed": 0,
                "total": total_tests,
                "compiled": False,
                "test_passes": [],
                "full_pass": False,
                "status": "empty",
            }
        # The harness provides its own main(); any surviving main is malformed.
        if self._has_main_function(solution_code):
            return {
                "reward": self.binary_fail_reward if self.reward_mode == "binary" else self.main_violation_penalty,
                "pass_ratio": 0.0,
                "passed": 0,
                "total": total_tests,
                "compiled": False,
                "test_passes": [],
                "full_pass": False,
                "status": "main_violation",
            }

        compiled = self._compile_candidate_with_tests(solution_code, test_code, test_cases)
        if not compiled:
            return {
                "reward": self._failure_reward(),
                "pass_ratio": 0.0,
                "passed": 0,
                "total": total_tests,
                "compiled": False,
                "test_passes": [False] * total_tests,
                "full_pass": False,
                "status": "compile_failed",
            }

        details = self._parse_and_run_individual_tests(solution_code, test_code)
        if details['total'] != total_tests:
            raise RuntimeError(
                f"Per-test reward count drift: expected {total_tests}, got {details['total']}"
            )

        pass_ratio = details['passed'] / details['total']
        test_passes = [bool(x.get('passed')) for x in details.get('individual_results', [])]
        full_pass = details['passed'] == details['total'] and self._run_full_test_harness(
            solution_code, test_code
        )
        if full_pass:
            # Make a fully passing solution the only high-reward outcome.
            # Earlier linear shaping (-1 + 8 * pass_ratio) over-rewarded
            # candidates that almost passed and moved eval pass@k downward.
            total = self.perfect_base_reward + self.perfect_bonus
        elif details['passed'] > 0:
            # Partial tests are useful as a direction signal, but keep them
            # far below a full pass so GRPO does not optimize "almost right".
            total = self.compile_reward + self.partial_reward_cap * pass_ratio
        else:
            total = self.compile_reward

        return {
            "reward": max(-5.0, min(10.0, total)),
            "pass_ratio": pass_ratio,
            "passed": details['passed'],
            "total": details['total'],
            "compiled": compiled,
            "test_passes": test_passes,
            "full_pass": full_pass,
            "status": "tested_full_pass" if full_pass else "tested_partial",
        }

    def compute_reward(self, solution_code: str, test_code: str) -> float:
        return float(self.compute_reward_details(solution_code, test_code)["reward"])

    # --- test extraction / execution --------------------------------------
    def _extract_expect_calls_single_line(self, test_code: str) -> List[str]:
        cases = []
        for ln in test_code.splitlines():
            s = ln.strip()
            if s.startswith('expect(') and 'candidate(' in s:
                cases.append(s if s.endswith(';') else s + ';')
        return cases

    def _resolve_candidate_name(self, solution_code: str, full_test_code: str) -> Optional[str]:
        # Preferred: alias declared in the test file.
        m = re.search(r"final\s+candidate\s*=\s*(\w+)\s*;", full_test_code)
        if m:
            return m.group(1)
        m2 = re.search(r"^[\w<>\[\]\?,\s]+?\s+(\w+)\s*\(", solution_code, re.MULTILINE)
        if m2:
            return m2.group(1)
        return None

    def _build_test_program(self, solution_code: str, test_cases: List[str], full_test_code: str) -> Optional[str]:
        all_imports = self._combine_imports(
            self._extract_imports(full_test_code),
            self._extract_imports(solution_code),
        )
        helpers = self._extract_helper_functions_safe(full_test_code)
        func_body = self._extract_function_body(solution_code)
        actual_func_name = self._resolve_candidate_name(solution_code, full_test_code)
        if not actual_func_name:
            return None

        test_body = "\n  ".join(test_cases)
        return f"""{all_imports}

{func_body}

{helpers}

void main() {{
  final candidate = {actual_func_name};
  {test_body}
}}
"""

    def _build_full_test_program(self, solution_code: str, full_test_code: str) -> str:
        all_imports = self._combine_imports(
            self._extract_imports(full_test_code),
            self._extract_imports(solution_code),
        )
        func_body = self._extract_function_body(solution_code)
        test_body = "\n".join(
            line
            for line in full_test_code.splitlines()
            if not line.strip().startswith(("import ", "export ", "library ", "part "))
        )
        return f"{all_imports}\n\n{func_body}\n\n{test_body}\n"

    def _compile_candidate_with_tests(self, solution_code: str, full_test_code: str, test_cases: List[str]) -> bool:
        del test_cases
        program = self._build_full_test_program(solution_code, full_test_code)
        # With 32-64 reward workers hammering the CPU, a cold `dart compile`
        # can blow a small timeout spuriously; branding that candidate
        # "does not compile" injects pure noise into the reward. Retry once
        # with double the budget before giving up.
        for attempt in range(2):
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    src = os.path.join(tmp, 'compile_check.dart')
                    out = os.path.join(tmp, 'compile_check.dill')
                    with open(src, 'w', encoding='utf-8') as f:
                        f.write(program)
                    proc = subprocess.run(
                        ['dart', '--disable-dart-dev', 'compile', 'kernel', src, '-o', out],
                        cwd=tmp,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout * (attempt + 1),
                        env=self._dart_subprocess_env(tmp),
                    )
                    return proc.returncode == 0
            except subprocess.TimeoutExpired:
                continue
            except Exception:
                return False
        return False

    def _run_full_test_harness(self, solution_code: str, full_test_code: str) -> bool:
        program = self._build_full_test_program(solution_code, full_test_code)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "full_test.dart")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(program)
                proc = subprocess.run(
                    ["dart", "--disable-dart-dev", "run", path],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=self._dart_subprocess_env(tmp),
                )
                return proc.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _parse_and_run_individual_tests(self, solution_code: str, test_code: str) -> dict:
        solution_code = self._prepare_solution_for_tests(solution_code)
        test_cases = self._extract_expect_calls_single_line(test_code)
        results = {'total': len(test_cases), 'passed': 0, 'failed': 0, 'individual_results': []}
        if not test_cases:
            return results
        # Each assertion runs an independent `dart run` in its own temp dir, so
        # they are safe to execute concurrently. The threads block on the
        # subprocess (releasing the GIL), giving near-linear speedup with cores.
        workers = max(1, min(GRPO_REWARD_WORKERS, len(test_cases)))
        if workers == 1:
            ordered = [self._run_single_expect_test(solution_code, tc, test_code)
                       for tc in test_cases]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                ordered = list(pool.map(
                    lambda tc: self._run_single_expect_test(solution_code, tc, test_code),
                    test_cases,
                ))
        for ok in ordered:
            results['individual_results'].append({'passed': ok})
            results['passed'] += int(ok)
        results['failed'] = results['total'] - results['passed']
        return results

    def _run_single_expect_test(self, solution_code: str, test_case: str, full_test_code: str) -> bool:
        try:
            program = self._build_test_program(solution_code, [test_case], full_test_code)
            if program is None:
                return False
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, 'single_test.dart')
                with open(p, 'w', encoding='utf-8') as f:
                    f.write(program)
                cmd = ['dart', '--disable-dart-dev', 'run', p]
                if self.enable_asserts:
                    cmd.insert(1, '--enable-asserts')
                proc = subprocess.run(
                    cmd,
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=self._dart_subprocess_env(tmp),
                )
                return proc.returncode == 0
        except Exception:
            return False

    def _combine_imports(self, a: str, b: str) -> str:
        seen, out = set(), []
        for src in (a, b):
            for ln in src.splitlines():
                key = ln.strip()
                if key and key not in seen:
                    seen.add(key)
                    out.append(ln)
        return '\n'.join(out)

    def _extract_imports(self, code: str) -> str:
        return '\n'.join([ln for ln in code.splitlines() if ln.strip().startswith(('import ', 'export '))])

    def _extract_helper_functions_safe(self, test_code: str) -> str:
        # Only capture top-level helper *definitions* (e.g. `void expect(...) {`),
        # never call statements like `expect(candidate(...), true);` that appear
        # inside the harness's own main(). A definition signature requires a
        # preceding return type and an opening `{` for the body (possibly on a
        # following line), and must not be a `;`-terminated call.
        lines = test_code.splitlines()
        i, out = 0, []
        while i < len(lines):
            s = lines[i].strip()
            is_def = (
                re.match(r'^[\w<>\[\]\?,\s]+\s+expect\w*\s*\(', s) is not None
                and not s.endswith(';')
                and '{' in s
            )
            if is_def:
                brace, j = 0, i
                while j < len(lines):
                    cur = lines[j]
                    out.append(cur)
                    brace += cur.count('{') - cur.count('}')
                    if brace == 0 and cur.strip().endswith('}'):
                        i = j
                        break
                    j += 1
            i += 1
        return '\n'.join(out)

    def _extract_function_body(self, code: str) -> str:
        code = self._prepare_solution_for_tests(code)
        body = []
        for ln in code.splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith(('import ', 'export ', '@pragma(', 'library ', 'part ', '//', '/*')):
                continue
            body.append(ln)
        return '\n'.join(body)


_dart_per_test_reward = TruePerTestReward()


def compile_swift(code: str) -> bool:
    # The GRPO reward set only ships Dart unit tests, so Swift candidates are
    # rewarded on a syntax/type-check basis (swiftc -typecheck) rather than tests.
    if not shutil.which("swiftc"):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'main.swift'
        try:
            path.write_text(code, encoding='utf-8')
            result = subprocess.run(
                ['swiftc', '-typecheck', str(path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=15
            )
            return result.returncode == 0
        except Exception:
            return False


def _completion_diversity_key(text: str) -> str:
    """Cheap exact-ish duplicate key for group-level diversity shaping."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def verpo_group_rewards(
    group_details: List[Dict[str, Any]],
    group_rewards: List[float],
    *,
    alpha: float,
    anchor_weight: float,
    density_norm: bool = True,
    epsilon: float = 1e-8,
) -> List[float]:
    """VeRPO reward for one single-turn rollout group (arxiv 2601.03525).

    The local reward follows equations 2-6: estimate each test's group pass
    rate, apply inverse-pass-rate difficulty weighting, then divide by a
    Gaussian-kernel estimate of difficulty density. The global full-suite
    outcome is added as the equation 10 anchor. Because this project has one
    generation turn per trajectory and uses mean centering (Fnorm=1), centering
    this combined reward is equivalent to the paper's dual-level advantage.

      rho_j = fraction of the group's candidates that passed test j
      w_j = exp(-alpha * rho_j)
      density_j = sum_k exp(-(rho_j-rho_k)^2 / (2*sigma^2))
      w'_j = w_j / (density_j + epsilon)
      reward_i = sum_j w'_j * passed_ij + anchor_weight * full_pass_i

    When a group has no per-test evidence (for example, a non-Dart fallback),
    the original rewards pass through unchanged.
    """
    if len(group_details) != len(group_rewards):
        raise ValueError("VeRPO details/reward lengths differ")
    if alpha <= 0.0:
        raise ValueError("VeRPO alpha must be positive")
    if anchor_weight < 0.0:
        raise ValueError("VeRPO anchor weight must be non-negative")
    if epsilon <= 0.0:
        raise ValueError("VeRPO epsilon must be positive")

    group_size = len(group_details)
    n_tests = max((len(d.get("test_passes") or []) for d in group_details), default=0)
    if group_size == 0 or n_tests == 0:
        return list(group_rewards)

    # G x T pass matrix; missing entries (shorter/absent vectors) count as fails.
    matrix: List[List[bool]] = []
    for detail in group_details:
        passes = [bool(x) for x in (detail.get("test_passes") or [])[:n_tests]]
        passes += [False] * (n_tests - len(passes))
        matrix.append(passes)

    rho = [sum(matrix[g][j] for g in range(group_size)) / group_size for j in range(n_tests)]
    weights = [math.exp(-alpha * rho[j]) for j in range(n_tests)]

    if density_norm:
        rho_mean = sum(rho) / n_tests
        rho_variance = sum((value - rho_mean) ** 2 for value in rho) / n_tests
        sigma = math.sqrt(rho_variance) / math.sqrt(2.0)
        if sigma <= epsilon:
            densities = [float(n_tests)] * n_tests
        else:
            denominator = 2.0 * sigma * sigma
            densities = [
                sum(
                    math.exp(-((rho[j] - rho[k]) ** 2) / denominator)
                    for k in range(n_tests)
                )
                for j in range(n_tests)
            ]
        weights = [
            weight / (density + epsilon)
            for weight, density in zip(weights, densities)
        ]

    out: List[float] = []
    for g, detail in enumerate(group_details):
        local_reward = sum(weights[j] for j in range(n_tests) if matrix[g][j])
        global_reward = 1.0 if bool(detail.get("full_pass")) else 0.0
        out.append(local_reward + anchor_weight * global_reward)
    return out


def reward_configuration(reward: TruePerTestReward) -> Dict[str, Any]:
    if reward.reward_mode == "verpo":
        return {
            "difficulty_alpha": reward.verpo_alpha,
            "anchor_weight": reward.verpo_anchor_weight,
            "density_normalization": "gaussian_kde" if reward.verpo_density_norm else "off",
            "advantage_normalization": "mean",
        }
    return {
        "perfect": reward.perfect_base_reward + reward.perfect_bonus,
        "binary_fail": reward.binary_fail_reward,
    }


def group_advantages(
    rewards: torch.Tensor,
    batch_size: int,
    group_size: int,
    norm: str = "mean",
    min_reward_range: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-relative advantages with a no-signal guard.

    Groups whose reward range is at or below `min_reward_range` get zero
    advantage so the caller can skip them: with 4-8 samples, std-normalizing
    noise-level reward differences (e.g. a lone duplicate penalty on otherwise
    identical failures) would blow them up into full-size +-1.5 advantages.
    `norm="mean"` (Dr.GRPO-style, reward minus group mean) keeps advantage
    magnitude proportional to the actual reward gap; `norm="std"` restores the
    original GRPO normalization.

    Returns (advantages flat (B*G,), signal_mask (B,) bool).
    """
    grouped = rewards.view(batch_size, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    reward_range = grouped.max(dim=1).values - grouped.min(dim=1).values
    signal_mask = reward_range > min_reward_range
    if norm == "std":
        std = grouped.std(dim=1, keepdim=True).clamp(min=1e-5)
        advantages = (grouped - mean) / std
    else:
        advantages = grouped - mean
    advantages = advantages * signal_mask.unsqueeze(1).to(advantages.dtype)
    return advantages.reshape(-1), signal_mask


def passk_advantage_weights(
    perfect_flags: torch.Tensor,
    batch_size: int,
    group_size: int,
    k: int,
) -> torch.Tensor:
    """Per-sample weights implementing the pass@k policy gradient.

    For a prompt with per-sample solve probability p, the pass@k objective
    1-(1-p)^k has gradient k*(1-p)^(k-1) * grad(p): the pass@1 gradient
    scaled by a PROMPT-LEVEL factor that vanishes as the prompt becomes
    reliably solved. Estimating p with the group's perfect-pass rate and
    folding the constant k into the learning rate, each group's advantages
    get multiplied by (1 - p_hat)^(k-1).

    Effect: already-solved prompts (p_hat -> 1) stop contributing gradient,
    so the policy cannot keep sharpening on them; rarely-solved prompts keep
    full weight. This is the coverage-preserving counterweight to the
    mode-concentration that plain per-sample reward maximization causes
    (observed in Stage C: pass@1 up, pass@5/10 down).

    perfect_flags: (batch_size*group_size,) float tensor of 0/1 ALL-tests-pass
    flags, grouped contiguously like rewards. Returns weights of the same shape.
    """
    if k <= 1:
        return torch.ones_like(perfect_flags)
    flags = perfect_flags.reshape(batch_size, group_size)
    p_hat = flags.mean(dim=1, keepdim=True)
    weights = (1.0 - p_hat).clamp(min=0.0) ** (k - 1)
    return weights.expand(batch_size, group_size).reshape(-1)


def effective_token_log_probs(
    policy_log_probs: torch.Tensor,
    token_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    simko_k: int,
) -> torch.Tensor:
    """Legacy experimental top-K positive smoothing.

    This is NOT a faithful implementation of the current CaSP method formerly
    discussed as SimKO: it has no sampled-token/top-N mixture coefficient,
    entropy-quantile gate, or stronger negative rank-1 term. It is retained
    only so historical commands remain loadable and must stay disabled in the
    confirmatory run. simko_k <= 1 is the identity.

    policy_log_probs: (N, T, V); token_log_probs: (N, T); advantages: (N,).
    """
    if simko_k <= 1:
        return token_log_probs
    topk_mean = policy_log_probs.topk(simko_k, dim=-1).values.mean(dim=-1)
    positive = (advantages > 0).to(token_log_probs.dtype).unsqueeze(-1)
    return positive * topk_mean + (1.0 - positive) * token_log_probs


def selected_token_log_probs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Return sampled-token log probabilities without a full log-softmax tensor.

    Qwen3-8B has a large vocabulary. Materializing ``log_softmax(logits.float())``
    for every generated token adds several GiB even for a small scoring chunk.
    Cross entropy computes the same selected-token quantity through the fused
    log-softmax/NLL path and keeps the full-vocabulary tensor in its model dtype.
    """
    vocab_size = logits.size(-1)
    losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    )
    return -losses.reshape_as(target_ids).float()


def pooled_surrogate_loss(
    loss_matrix: torch.Tensor,
    valid_mask: torch.Tensor,
    pooling: str,
    token_denom: torch.Tensor | float,
    sample_denom: float,
) -> torch.Tensor:
    """Pool the per-token surrogate loss.

    token: sum over all valid tokens / GLOBAL valid-token count (DAPO-style
           token-level loss; long sequences carry proportionally more
           gradient - this has always been this trainer's behavior).
    seq:   per-sample length-normalized mean, then / GLOBAL sample count
           (GSPO-style sequence-level pooling; every sample counts equally,
           so short passing completions are no longer outweighed by long
           rambling failures).

    Both denominators are GLOBAL so chunked scoring sums to the identical
    full-batch value (covered in grpo_selfcheck.py).
    """
    masked = loss_matrix * valid_mask
    if pooling == "seq":
        per_seq = masked.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        return per_seq.sum() / sample_denom
    return masked.sum() / token_denom


def calculate_rewards(
    completions: List[str],
    references: List[str],
    languages: List[str],
    tests: Optional[List[str]] = None,
    return_stats: bool = False,
    group_size: Optional[int] = None,
    unique_test_bonus: float = 0.0,
    duplicate_penalty: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
    if tests is None:
        tests = [None] * len(completions)
    lengths = {len(completions), len(references), len(languages), len(tests)}
    if len(lengths) != 1:
        raise ValueError(
            "Reward inputs are misaligned: "
            f"completions={len(completions)}, references={len(references)}, "
            f"languages={len(languages)}, tests={len(tests)}"
        )
    reward_inputs = list(zip(completions, references, languages, tests))

    # Exact binary reward is one evaluator-aligned full-harness run per completion. The
    # old worker knob only affected assertions inside shaped/VeRPO mode, leaving G
    # binary candidates serial. Score independent candidates concurrently;
    # every invocation uses its own temporary directory and Dart home.
    dart_jobs = [
        (index, comp, test_code)
        for index, (comp, _ref, lang, test_code) in enumerate(reward_inputs)
        if str(lang).lower() == 'dart' and test_code
    ]
    dart_details: Dict[int, Dict[str, Any]] = {}

    def score_dart(job):
        index, comp, test_code = job
        return index, _dart_per_test_reward.compute_reward_details(comp, test_code)

    can_parallelize = isinstance(_dart_per_test_reward, TruePerTestReward)
    workers = max(1, min(GRPO_REWARD_WORKERS, len(dart_jobs))) if dart_jobs else 1
    if dart_jobs and can_parallelize and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, details in pool.map(score_dart, dart_jobs):
                dart_details[index] = details
    else:
        for job in dart_jobs:
            index, details = score_dart(job)
            dart_details[index] = details

    rewards = []
    pass_ratios = []
    perfect_flags: List[float] = []  # 1.0 iff ALL unit tests passed; aligned 1:1 with rewards
    compiled_flags = []
    reward_details: List[Dict[str, Any]] = []
    tested_count = 0
    overlap_weight = float(os.environ.get("GRPO_OVERLAP_WEIGHT", "0.0"))
    for input_index, (comp, ref, lang, test_code) in enumerate(reward_inputs):
        overlap = compute_overlap_reward(ref, comp)
        reward_score = overlap
        detail_for_stats: Dict[str, Any] = {}

        # Dart uses the exact full harness in binary mode and per-assertion
        # diagnostics in shaped/VeRPO mode. Swift has no tests in this corpus
        # and falls back to `swiftc -typecheck`.
        lang_lower = str(lang).lower()
        if lang_lower == 'dart' and test_code:
            # The per-test scorer already encodes empty/main violations, partial
            # pass credit, and a perfect-pass bonus. Use it as the reward.
            # Token-overlap shaping defaults to 0 because it improved CodeBLEU
            # while failing to improve pass@k in the first GRPO runs.
            details = dart_details[input_index]
            functional = float(details["reward"])
            reward_score = functional + overlap_weight * overlap
            rewards.append(reward_score)
            detail_for_stats = details
            if details.get("compiled") is not None:
                compiled_flags.append(bool(details["compiled"]))
            if details["pass_ratio"] is not None:
                pass_ratios.append(float(details["pass_ratio"]))
                tested_count += 1
            perfect_flags.append(1.0 if bool(details.get("full_pass")) else 0.0)
            reward_details.append(detail_for_stats)
            continue

        comp_strip = comp.strip()
        # Heavily penalize empty or tiny outputs
        if not comp_strip or len(comp_strip) < 10:
            rewards.append(-1.0)
            perfect_flags.append(0.0)
            reward_details.append({"test_passes": [], "compiled": False})
            continue
        elif lang_lower == 'swift':
            is_compilable = compile_swift(comp)
        else:
            is_compilable = None

        if is_compilable is True:
            reward_score += 0.5  # Compile bonus!
        elif is_compilable is False:
            reward_score -= 0.2  # Compile penalty!

        rewards.append(reward_score)
        perfect_flags.append(0.0)  # no unit-test evidence => never "perfect"
        reward_details.append({"test_passes": [], "compiled": bool(is_compilable) if is_compilable is not None else False})

    # VeRPO dense reward: recompute each group's per-candidate reward from the
    # group's per-test pass matrix BEFORE any bonus/penalty shaping is layered
    # on. This is what breaks the all-fail degenerate group (60-70% -> <25%).
    reward_mode = getattr(_dart_per_test_reward, "reward_mode", "shaped")
    if reward_mode == "verpo":
        if not group_size or group_size <= 1:
            raise ValueError("VeRPO requires group_size > 1")
        if len(rewards) % group_size != 0:
            raise ValueError(
                "VeRPO reward count must be divisible by group_size: "
                f"rewards={len(rewards)} group_size={group_size}"
            )
        for start in range(0, len(rewards), group_size):
            end = start + group_size
            new_group = verpo_group_rewards(
                reward_details[start:end],
                rewards[start:end],
                alpha=getattr(_dart_per_test_reward, "verpo_alpha", 2.0),
                anchor_weight=getattr(
                    _dart_per_test_reward, "verpo_anchor_weight", 1.0
                ),
                density_norm=getattr(
                    _dart_per_test_reward, "verpo_density_norm", True
                ),
            )
            rewards[start:end] = new_group

    unique_bonus_values = [0.0 for _ in rewards]
    duplicate_penalty_values = [0.0 for _ in rewards]
    if group_size and group_size > 1 and (unique_test_bonus > 0.0 or duplicate_penalty > 0.0):
        for start in range(0, len(rewards), group_size):
            end = min(start + group_size, len(rewards))
            group_details = reward_details[start:end]
            group_completions = completions[start:end]

            if unique_test_bonus > 0.0:
                max_tests = max((len(d.get("test_passes", [])) for d in group_details), default=0)
                if max_tests > 0:
                    pass_counts = [0] * max_tests
                    for d in group_details:
                        passes = d.get("test_passes", [])
                        for idx, ok in enumerate(passes[:max_tests]):
                            pass_counts[idx] += int(bool(ok))
                    for offset, d in enumerate(group_details):
                        passes = d.get("test_passes", [])
                        unique_count = sum(
                            1 for idx, ok in enumerate(passes[:max_tests])
                            if bool(ok) and pass_counts[idx] == 1
                        )
                        bonus = unique_test_bonus * (unique_count / max_tests)
                        unique_bonus_values[start + offset] = bonus
                        rewards[start + offset] += bonus

            if duplicate_penalty > 0.0:
                keys = [_completion_diversity_key(c) for c in group_completions]
                counts: Dict[str, int] = {}
                for key in keys:
                    counts[key] = counts.get(key, 0) + 1
                denom = max(1, end - start - 1)
                for offset, key in enumerate(keys):
                    duplicates = max(0, counts.get(key, 0) - 1)
                    penalty = duplicate_penalty * (duplicates / denom)
                    duplicate_penalty_values[start + offset] = penalty
                    rewards[start + offset] -= penalty

    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    if not return_stats:
        return reward_tensor

    if pass_ratios:
        n = len(pass_ratios)
        perfect = sum(1 for r in pass_ratios if r >= 1.0)
        near_perfect = sum(1 for r in pass_ratios if 0.8 <= r < 1.0)
        high_partial = sum(1 for r in pass_ratios if 0.5 <= r < 0.8)
        zero_pass = sum(1 for r in pass_ratios if r <= 0.0)
        stats = {
            "pass_ratio_mean": sum(pass_ratios) / n,
            "pass_ratio_max": max(pass_ratios),
            "perfect_rate": perfect / n,
            "near_perfect_rate": near_perfect / n,
            "high_partial_rate": high_partial / n,
            "zero_pass_rate": zero_pass / n,
            "tested_reward_count": float(tested_count),
            "compiled_rate": (sum(compiled_flags) / len(compiled_flags)) if compiled_flags else 0.0,
            "compiled_known_fraction": len(compiled_flags) / max(1, len(rewards)),
            "unique_bonus_mean": sum(unique_bonus_values) / len(unique_bonus_values) if unique_bonus_values else 0.0,
            "duplicate_penalty_mean": sum(duplicate_penalty_values) / len(duplicate_penalty_values) if duplicate_penalty_values else 0.0,
            # per-sample (not a scalar): consumed by pass@k advantage weighting
            "perfect_flags": perfect_flags,
        }
    else:
        stats = {
            "pass_ratio_mean": 0.0,
            "pass_ratio_max": 0.0,
            "perfect_rate": 0.0,
            "near_perfect_rate": 0.0,
            "high_partial_rate": 0.0,
            "zero_pass_rate": 0.0,
            "tested_reward_count": 0.0,
            "compiled_rate": (sum(compiled_flags) / len(compiled_flags)) if compiled_flags else 0.0,
            "compiled_known_fraction": len(compiled_flags) / max(1, len(rewards)),
            "unique_bonus_mean": sum(unique_bonus_values) / len(unique_bonus_values) if unique_bonus_values else 0.0,
            "duplicate_penalty_mean": sum(duplicate_penalty_values) / len(duplicate_penalty_values) if duplicate_penalty_values else 0.0,
            "perfect_flags": perfect_flags,
        }
    return reward_tensor, stats

class GRPOTrainer:
    def __init__(
        self,
        model: GraphCodeBERTT5Seq2Seq,
        tokenizer: PreTrainedTokenizerBase,
        args: argparse.Namespace,
        verified_anchor_loader: DataLoader | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.grad_accum_steps = max(1, int(getattr(args, "grad_accum", 1)))
        self._grad_accum_counter = 0
        # Generation budget per completion. 128 truncated many valid Dart
        # solutions (SFT target length is 768) -> they failed every test and
        # produced no learning signal. 256 covers the vast majority.
        self.max_new_tokens = getattr(args, "max_new_tokens", 256)
        # Sampling params are shared between rollout generation and log-prob
        # scoring: logits are divided by gen_temperature before log_softmax so
        # the scored distribution is the one candidates were drawn from
        # (top-p truncation remains the usual accepted approximation).
        self.gen_temperature = float(getattr(args, "gen_temperature", 0.7))
        self.gen_top_p = float(getattr(args, "gen_top_p", 0.95))
        self.strict_graph = os.environ.get("GRPO_STRICT_GRAPH", "1") == "1"
        self.sft_anchor_coef = float(getattr(args, "sft_anchor_coef", 0.0))
        self.sft_anchor_on_no_signal = bool(
            int(getattr(args, "sft_anchor_on_no_signal", 0))
        )
        self.verified_anchor_loader = verified_anchor_loader
        self._verified_anchor_iterator = iter(verified_anchor_loader) if verified_anchor_loader is not None else None
        if self.sft_anchor_coef < 0.0:
            raise ValueError("GRPO SFT anchor coefficient must be non-negative")
        if self.sft_anchor_coef > 0.0 and self.verified_anchor_loader is None:
            raise ValueError(
                "GRPO_SFT_ANCHOR_COEF is nonzero but no independently verified anchor loader was supplied"
            )
        print(
            "GRPO verified-only SFT anchor: "
            f"coef={self.sft_anchor_coef} "
            f"on_no_signal={self.sft_anchor_on_no_signal} "
            f"source={'separate verified loader' if verified_anchor_loader is not None else 'disabled'}"
        )

        # Train the graph glue (GNN + projection + prefix adapter) with the
        # policy gradient. Default on: the earlier implementation ran the whole
        # graph branch under no_grad, so GRPO could never adapt the prefix the
        # decoder conditions on. GRPO_TRAIN_GRAPH_GLUE=0 restores the frozen
        # behavior. The local block encoder is always frozen during GRPO
        # (its SFT/LoRA state is kept; backprop through hundreds of
        # GraphCodeBERT block forwards is not worth the memory).
        self.train_graph_glue = os.environ.get("GRPO_TRAIN_GRAPH_GLUE", "1") == "1"
        print(f"GRPO graph glue training: {'ON' if self.train_graph_glue else 'OFF (frozen prefix)'}")

        # The local block encoder is intentionally frozen during GRPO. Mark it
        # frozen explicitly so it cannot enter AdamW merely because the SFT
        # checkpoint left encoder LoRA parameters with requires_grad=True.
        for parameter in self.model.local_encoder.parameters():
            parameter.requires_grad_(False)

        # Set up optimizer only on trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
        self.optimizer.zero_grad(set_to_none=True)

    def _next_verified_anchor_batch(self) -> Dict[str, Any]:
        if self.verified_anchor_loader is None:
            raise RuntimeError("verified anchor loader is unavailable")
        if self._verified_anchor_iterator is None:
            self._verified_anchor_iterator = iter(self.verified_anchor_loader)
        try:
            return next(self._verified_anchor_iterator)
        except StopIteration:
            self._verified_anchor_iterator = iter(self.verified_anchor_loader)
            return next(self._verified_anchor_iterator)

    def _backward_sft_anchor(self, device: str) -> float:
        """Backpropagate CE only on an independently replayed alternative target.

        The anchor batch comes from ``GRPO_VERIFIED_ANCHOR_FILE``; ordinary RL-row references are never
        used as anchor labels. This prevents all-zero groups from silently falling
        back to ordinary CE on the gold reference corpus.
        """
        if self.sft_anchor_coef <= 0.0:
            return 0.0
        batch = self._next_verified_anchor_batch()
        labels = batch.get("labels")
        if labels is None:
            raise ValueError("verified GRPO anchor requires labels")
        labels = labels.to(device)
        prompt_ids = batch.get("decoder_prompt_input_ids")
        prompt_mask = batch.get("decoder_prompt_attention_mask")
        if prompt_ids is not None:
            prompt_ids = prompt_ids.to(device)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(device)

        outputs = self.model(
            labels=labels,
            decoder_prompt_input_ids=prompt_ids,
            decoder_prompt_attention_mask=prompt_mask,
            cfg=batch.get("cfg"),
            edges=batch.get("edges"),
            block_inputs=batch.get("block_inputs"),
        )
        anchor_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if anchor_loss is None or not torch.isfinite(anchor_loss):
            raise RuntimeError(f"non-finite verified GRPO anchor loss: {anchor_loss}")
        (self.sft_anchor_coef * anchor_loss / self.grad_accum_steps).backward()
        return float(anchor_loss.detach().cpu().item())

    def _optimizer_step_if_needed(self, force: bool = False) -> bool:
        if self._grad_accum_counter <= 0:
            return False
        if not force and self._grad_accum_counter % self.grad_accum_steps != 0:
            return False
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if force and self._grad_accum_counter < self.grad_accum_steps:
            # Every microbatch backward is divided by grad_accum_steps. A
            # sparse-reward epoch can end with only one or two signal-bearing
            # microbatches; rescale that partial window to their actual mean
            # instead of silently shrinking the final/only optimizer update.
            correction = self.grad_accum_steps / self._grad_accum_counter
            for param in trainable_params:
                if param.grad is not None:
                    param.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._grad_accum_counter = 0
        return True

    def _shift_right_seq2seq(self, target_ids: torch.Tensor) -> torch.Tensor:
        """Build decoder inputs for seq2seq GRPO log-prob recomputation.

        Some HF seq2seq models expose `_shift_right`, but Salesforce CodeT5+
        does not. Mirror the SFT fallback so GRPO works for both T5 and CodeT5+.
        """
        if hasattr(self.model.base_decoder_model, "_shift_right"):
            return self.model.base_decoder_model._shift_right(target_ids)

        cfg = self.model.base_decoder_model.config
        decoder_start_token_id = getattr(cfg, "decoder_start_token_id", None)
        if decoder_start_token_id is None and hasattr(cfg, "decoder"):
            decoder_start_token_id = getattr(cfg.decoder, "decoder_start_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = getattr(cfg, "pad_token_id", None)
        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.pad_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.tokenizer.eos_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = 0

        pad_token_id = getattr(cfg, "pad_token_id", None)
        if pad_token_id is None and hasattr(cfg, "decoder"):
            pad_token_id = getattr(cfg.decoder, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        shifted = target_ids.new_full(target_ids.shape, pad_token_id)
        shifted[..., 0] = decoder_start_token_id
        shifted[..., 1:] = target_ids[..., :-1].clone()
        shifted.masked_fill_(shifted == -100, pad_token_id)
        return shifted

    def _compute_graph_context(self, cfg, edges, block_inputs, labels, device):
        """Graph channel forward for one batch: (encoder_hidden_states, mask).

        The local block encoder always runs under no_grad (frozen for GRPO);
        the graph glue (GNN/projection/prefix adapter) is differentiable when
        train_graph_glue is on so the policy gradient reaches the prefix.

        Text-only mode (causal + GRAPH_QWEN_PREFIX_TOKENS=0): returns a
        zero-width prefix and skips all graph compute; the downstream
        cat/mask/slice arithmetic handles width 0 unchanged.
        """
        if self.model.is_causal and self.model.qwen_prefix_tokens == 0:
            batch_size = len(block_inputs) if block_inputs is not None else labels.size(0)
            return (
                torch.zeros((batch_size, 0, self.model.decoder_dim), device=device),
                torch.zeros((batch_size, 0), device=device),
            )

        if self.strict_graph and (cfg is None or edges is None or block_inputs is None):
            raise ValueError("Strict GRPO graph mode requires cfg, edges, and block_inputs")
        if not (len(cfg) == len(edges) == len(block_inputs)):
            raise ValueError(
                "GRPO graph batch is misaligned: "
                f"cfg={len(cfg)} edges={len(edges)} block_inputs={len(block_inputs)}"
            )

        with torch.no_grad():
            block_embeddings_batch = []
            list_of_B_i = []

            for block_group in block_inputs:
                if not block_group:
                    if self.strict_graph:
                        raise ValueError("Strict GRPO graph mode encountered an empty block group")
                    list_of_B_i.append(1)
                    block_embeddings_batch.append(torch.zeros((1, self.model.encoder_dim), device=device))
                    continue

                group_input_ids = torch.stack([
                    torch.tensor(b['input_ids'], dtype=torch.long, device=device)
                    for b in block_group
                ])
                group_attention_mask = torch.stack([
                    torch.tensor(b['attention_mask'], dtype=torch.float, device=device).squeeze(0)
                    for b in block_group
                ])
                group_position_ids = torch.stack([
                    torch.tensor(b['position_ids'], dtype=torch.long, device=device)
                    for b in block_group
                ])
                group_token_type_ids = torch.stack([
                    torch.tensor(b['token_type_ids'], dtype=torch.long, device=device)
                    for b in block_group
                ])

                block_embeddings = self.model.local_encoder(
                    group_input_ids,
                    group_attention_mask,
                    group_position_ids,
                    group_token_type_ids,
                )
                block_embeddings_batch.append(block_embeddings)
                list_of_B_i.append(block_embeddings.size(0))

            edge_index = None
            edge_attr = None
            region_ids = None
            if cfg is not None and edges is not None:
                try:
                    batch_graphs = []
                    for batch_index in range(len(cfg)):
                        node_embeddings = block_embeddings_batch[batch_index]
                        if len(cfg[batch_index]) != node_embeddings.size(0):
                            raise ValueError(
                                "GRPO CFG/block tensor count mismatch for sample "
                                f"{batch_index}: cfg={len(cfg[batch_index])} "
                                f"embeddings={node_embeddings.size(0)}"
                            )
                        graph_record = {'cfg': cfg[batch_index], 'edges': edges[batch_index]}
                        graph_node_embeddings = (
                            node_embeddings.mean(dim=1)
                            if node_embeddings.dim() == 3
                            else node_embeddings
                        )
                        batch_graphs.append(
                            cfg_to_pyg(graph_record, graph_node_embeddings)
                        )
                    pyg_batch = Batch.from_data_list(batch_graphs)
                    if pyg_batch.ptr.diff().tolist() != list_of_B_i:
                        raise ValueError(
                            f"GRPO PyG pointers {pyg_batch.ptr.diff().tolist()} "
                            f"do not match block counts {list_of_B_i}"
                        )
                    if pyg_batch.edge_index.size(1) != pyg_batch.edge_attr.size(0):
                        raise ValueError("GRPO PyG edge_index/edge_attr cardinality mismatch")
                    edge_index = pyg_batch.edge_index
                    edge_attr = pyg_batch.edge_attr
                    region_ids = pyg_batch.region_id
                except Exception as exc:
                    self._graph_batch_failures = getattr(self, "_graph_batch_failures", 0) + 1
                    message = (
                        f"PyG graph construction failed for GRPO batch "
                        f"#{self._graph_batch_failures}: {exc!r}"
                    )
                    if self.strict_graph:
                        raise RuntimeError(message) from exc
                    print(f"[graph] WARNING: {message}; continuing without edges", file=sys.stderr)

        # Graph glue: differentiable when training it, no_grad when frozen.
        from contextlib import nullcontext
        glue_context = nullcontext() if self.train_graph_glue else torch.no_grad()
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
        with glue_context, torch.amp.autocast(device_type=autocast_device, enabled=False):
            graph_inputs = torch.cat(block_embeddings_batch, dim=0).float()
            encoder_hidden_states, encoder_attention_mask = self.model.graph_encoder(
                graph_inputs,
                edge_index=edge_index,
                edge_attr=edge_attr,
                list_of_B_i=list_of_B_i,
                region_ids=region_ids,
            )
            encoder_hidden_states = self.model.projection(encoder_hidden_states)
            encoder_hidden_states, encoder_attention_mask = self.model.prepare_decoder_context(
                encoder_hidden_states,
                encoder_attention_mask,
            )
        return encoder_hidden_states, encoder_attention_mask

    def train_step(
        self,
        batch: Dict[str, Any],
        device: str,
        update: bool = True,
        allow_no_signal_anchor: bool = True,
    ) -> Dict[str, float]:
        self.model.eval() # Generation uses eval mode

        cfg = batch.get('cfg')
        edges = batch.get('edges')
        block_inputs = batch.get('block_inputs')
        labels = batch.get('labels')
        decoder_prompt_input_ids = batch.get('decoder_prompt_input_ids')
        decoder_prompt_attention_mask = batch.get('decoder_prompt_attention_mask')
        if decoder_prompt_input_ids is not None:
            decoder_prompt_input_ids = decoder_prompt_input_ids.to(device)
        if decoder_prompt_attention_mask is not None:
            decoder_prompt_attention_mask = decoder_prompt_attention_mask.to(device)

        # 1. Forward pass through Graph Encoder
        encoder_hidden_states, encoder_attention_mask = self._compute_graph_context(
            cfg, edges, block_inputs, labels, device
        )

        # Repeat the hidden states G times for Group Sampling
        B, max_blocks, d_model = encoder_hidden_states.shape
        G = self.args.group_size

        # Reshape to group dimension
        # (B * G, max_blocks, d_model)
        grp_hidden_states = encoder_hidden_states.repeat_interleave(G, dim=0)
        grp_attention_mask = encoder_attention_mask.repeat_interleave(G, dim=0)
        grp_prompt_input_ids = decoder_prompt_input_ids.repeat_interleave(G, dim=0) if decoder_prompt_input_ids is not None else None
        grp_prompt_attention_mask = decoder_prompt_attention_mask.repeat_interleave(G, dim=0) if decoder_prompt_attention_mask is not None else None

        # 2. Sample completions from the policy model
        base_model = self.model.base_decoder_model

        if self.model.is_causal:
            start_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id or 0
            input_ids = torch.tensor([[start_token_id]], device=device).expand(B * G, -1)

            # Rollout generation never needs gradients; use a DETACHED prefix so
            # the glue autograd graph is neither recorded nor held here. The
            # differentiable grp_hidden_states is used by the scoring forward.
            with torch.no_grad():
                input_embeds = base_model.get_input_embeddings()(input_ids)
                embed_parts = [grp_hidden_states.detach()]
                if grp_prompt_input_ids is not None:
                    prompt_embeds = base_model.get_input_embeddings()(grp_prompt_input_ids)
                    embed_parts.append(prompt_embeds)
                embed_parts.append(input_embeds)
                inputs_embeds = torch.cat(embed_parts, dim=1)
                inputs_embeds = inputs_embeds.to(dtype=base_model.dtype)

                mask_parts = [grp_attention_mask]
                if grp_prompt_attention_mask is not None:
                    mask_parts.append(grp_prompt_attention_mask.to(dtype=grp_attention_mask.dtype))
                mask_parts.append(torch.ones((grp_attention_mask.size(0), 1), dtype=grp_attention_mask.dtype, device=device))
                combined_mask = torch.cat(mask_parts, dim=1)

                outputs = self.model.t5_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=combined_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.gen_temperature,
                    top_p=self.gen_top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
        else:
            # Seq2Seq
            start_token_id = self.model.t5_model.config.decoder_start_token_id or self.tokenizer.pad_token_id
            input_ids = torch.tensor([[start_token_id]], device=device).expand(B * G, -1)

            from transformers.modeling_outputs import BaseModelOutput
            with torch.no_grad():
                outputs = self.model.t5_model.generate(
                    decoder_input_ids=input_ids,
                    encoder_outputs=BaseModelOutput(last_hidden_state=grp_hidden_states.detach()),
                    attention_mask=grp_attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.gen_temperature,
                    top_p=self.gen_top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

        # 3. Calculate rewards for the generated sequences
        # Decode reference texts
        ref_texts = []
        languages = []
        test_harnesses = []
        batch_langs = batch.get('language', ['dart'] * B)
        batch_tests = batch.get('tests', [''] * B)
        for i in range(B):
            ref_id = labels[i]
            # Replace pad/ignore index
            ref_id_clean = [tok for tok in ref_id if tok != -100]
            ref_texts.append(self.tokenizer.decode(ref_id_clean, skip_special_tokens=True))
            languages.append(batch_langs[i])
            test_harnesses.append(batch_tests[i])

        completions = []
        for out_seq in outputs:
            completions.append(self.tokenizer.decode(out_seq, skip_special_tokens=True))

        # Replicate references G times to match completions batching
        repeated_refs = [ref for ref in ref_texts for _ in range(G)]
        repeated_langs = [lang for lang in languages for _ in range(G)]
        repeated_tests = [t for t in test_harnesses for _ in range(G)]

        rewards, reward_stats = calculate_rewards(
            completions,
            repeated_refs,
            repeated_langs,
            repeated_tests,
            return_stats=True,
            group_size=G,
            unique_test_bonus=float(getattr(self.args, "unique_test_bonus", 0.0)),
            duplicate_penalty=float(getattr(self.args, "duplicate_penalty", 0.0)),
        )
        rewards = rewards.to(device)

        # 4. Group-relative advantages (with a no-signal guard).
        adv_norm = str(getattr(self.args, "adv_norm", "mean")).strip().lower()
        min_reward_range = float(getattr(self.args, "min_reward_range", 0.05))
        advantages, signal_mask = group_advantages(
            rewards, B, G, norm=adv_norm, min_reward_range=min_reward_range
        )
        groups_with_signal = signal_mask.float().mean()

        # Optional pass@k objective: scale each group's advantages by
        # (1 - p_hat)^(k-1). Reliably-solved prompts contribute ~no gradient,
        # which blocks the pass@k-destroying sharpening seen with the plain
        # per-sample objective. Off when passk_k <= 1.
        passk_k = int(getattr(self.args, "passk_k", 0))
        passk_weight_mean = 1.0
        if passk_k > 1:
            perfect_flags = torch.tensor(
                reward_stats["perfect_flags"], dtype=advantages.dtype, device=advantages.device
            )
            passk_weights = passk_advantage_weights(perfect_flags, B, G, passk_k)
            advantages = advantages * passk_weights
            passk_weight_mean = float(passk_weights.mean().detach().cpu().item())

        # DAPO-style overlong filtering: samples that ran to max_new_tokens
        # without EOS are truncation artifacts (they fail compile for a
        # length reason, not a policy reason). Zero their advantage so the
        # policy is not taught from them. Off by default.
        overlong_rate = 0.0
        if int(getattr(self.args, "overlong_filter", 0)) == 1:
            eos_id_for_filter = self.tokenizer.eos_token_id
            has_eos = (outputs == eos_id_for_filter).any(dim=1)
            overlong_rate = float((~has_eos).float().mean().detach().cpu().item())
            advantages = advantages * has_eos.to(advantages.dtype)

        base_stats = {
            "loss": 0.0,
            "rl_loss": 0.0,
            "sft_anchor_loss": 0.0,
            "optimizer_stepped": 0.0,
            "grad_accum_counter": float(self._grad_accum_counter),
            "skipped_no_signal": 0.0,
            "reward_mean": float(rewards.mean().detach().cpu().item()),
            "reward_min": float(rewards.min().detach().cpu().item()),
            "reward_max": float(rewards.max().detach().cpu().item()),
            "reward_std": float(rewards.std(unbiased=False).detach().cpu().item()),
            "groups_with_signal": float(groups_with_signal.detach().cpu().item()),
            "advantage_abs_mean": float(advantages.abs().mean().detach().cpu().item()),
            "pass_ratio_mean": float(reward_stats["pass_ratio_mean"]),
            "pass_ratio_max": float(reward_stats["pass_ratio_max"]),
            "perfect_rate": float(reward_stats["perfect_rate"]),
            "near_perfect_rate": float(reward_stats["near_perfect_rate"]),
            "high_partial_rate": float(reward_stats["high_partial_rate"]),
            "zero_pass_rate": float(reward_stats["zero_pass_rate"]),
            "compiled_rate": float(reward_stats["compiled_rate"]),
            "compiled_known_fraction": float(reward_stats["compiled_known_fraction"]),
            "unique_bonus_mean": float(reward_stats["unique_bonus_mean"]),
            "duplicate_penalty_mean": float(reward_stats["duplicate_penalty_mean"]),
            "passk_weight_mean": passk_weight_mean,
            "overlong_rate": overlong_rate,
            "entropy": 0.0,
        }

        if not update:
            return {**base_stats, "skipped_no_signal": float(not bool(signal_mask.any()))}

        if not bool(signal_mask.any()) or not bool((advantages != 0).any()):
            # RL has no relative signal. With a separate verified-only SFT anchor this
            # micro-batch still supplies direction instead of disappearing from
            # training; without an anchor, retain the original cheap skip.
            if (
                allow_no_signal_anchor
                and self.sft_anchor_coef > 0.0
                and self.sft_anchor_on_no_signal
            ):
                anchor_loss = self._backward_sft_anchor(device)
                self._grad_accum_counter += 1
                optimizer_stepped = self._optimizer_step_if_needed(force=False)
                return {
                    **base_stats,
                    "loss": self.sft_anchor_coef * anchor_loss,
                    "sft_anchor_loss": anchor_loss,
                    "optimizer_stepped": float(optimizer_stepped),
                    "grad_accum_counter": float(self._grad_accum_counter),
                    "skipped_no_signal": 0.0,
                    "rl_no_signal": 1.0,
                }
            return {**base_stats, "skipped_no_signal": 1.0, "rl_no_signal": 1.0}

        # 5. Recompute log-probs of the sampled tokens.
        # The model intentionally STAYS in eval mode: gradients flow either
        # way, and LoRA dropout must stay off so the scored log-probs describe
        # the same deterministic policy that generated the samples.
        from peft import PeftModel

        kl_coef = float(self.args.kl_coef)
        use_ref = kl_coef > 0.0 and isinstance(self.model.t5_model, PeftModel)

        if self.model.is_causal:
            # Score every generated token, including the first one: the
            # scoring context replays generation exactly as
            # [graph prefix | prompt | start token | t0 .. t_{n-2}], so the
            # logit at the start token predicts t0, and so on.
            target_ids = outputs
            prefix_len = grp_hidden_states.size(1)
            prompt_len = grp_prompt_input_ids.size(1) if grp_prompt_input_ids is not None else 0

            start_col = torch.full(
                (outputs.size(0), 1), start_token_id, dtype=outputs.dtype, device=device
            )
            input_ids = torch.cat([start_col, outputs[:, :-1]], dim=1)
            target_embeds = base_model.get_input_embeddings()(input_ids)
            embed_parts = [grp_hidden_states]
            if grp_prompt_input_ids is not None:
                prompt_embeds = base_model.get_input_embeddings()(grp_prompt_input_ids)
                embed_parts.append(prompt_embeds)
            embed_parts.append(target_embeds)
            combined_inputs_embeds = torch.cat(embed_parts, dim=1)
            combined_inputs_embeds = combined_inputs_embeds.to(dtype=base_model.dtype)

            mask_parts = [grp_attention_mask]
            if grp_prompt_attention_mask is not None:
                mask_parts.append(grp_prompt_attention_mask.to(dtype=grp_attention_mask.dtype))
            mask_parts.append(torch.ones((grp_attention_mask.size(0), input_ids.size(1)), dtype=grp_attention_mask.dtype, device=device))
            forward_mask = torch.cat(mask_parts, dim=1)
            # The prompt is right-padded to a fixed length, so the combined
            # sequence has masked holes in the middle; replay generate()'s
            # cumsum position convention or the scored log-probs belong to a
            # different positional layout than the sampled tokens.
            position_ids = cumsum_position_ids(forward_mask)

            # Memory-bounded scoring for large groups: forward+backward in
            # sample chunks. Identical gradients (sum-decomposable loss with
            # one global denominator); peak memory set by chunk size, not G.
            score_chunk = int(getattr(self.args, "score_chunk_size", 0))
            if score_chunk > 0:
                return self._score_chunked_causal(
                    base_stats, combined_inputs_embeds, forward_mask,
                    position_ids, target_ids, prefix_len, prompt_len,
                    advantages, use_ref, kl_coef, device, score_chunk, batch,
                )

            policy_outputs = self.model.t5_model(
                inputs_embeds=combined_inputs_embeds,
                attention_mask=forward_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            policy_logits = policy_outputs.logits[:, prefix_len + prompt_len:, :]

            ref_logits = None
            if use_ref:
                # NOTE: adapter-disabled is the PRE-SFT base model, not the
                # GRPO starting policy. As a KL anchor it pulls the policy
                # away from the SFT solution, which is why kl_coef now
                # defaults to 0.0; only enable it knowing this caveat.
                with torch.no_grad(), self.model.t5_model.disable_adapter():
                    ref_outputs = self.model.t5_model(
                        inputs_embeds=combined_inputs_embeds,
                        attention_mask=forward_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    )
                    ref_logits = ref_outputs.logits[:, prefix_len + prompt_len:, :]
        else:
            # Seq2Seq
            target_ids = outputs
            decoder_input_ids = self._shift_right_seq2seq(target_ids)

            policy_outputs = self.model.t5_model(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=(grp_hidden_states,),
                attention_mask=grp_attention_mask,
            )
            policy_logits = policy_outputs.logits

            ref_logits = None
            if use_ref:
                with torch.no_grad(), self.model.t5_model.disable_adapter():
                    ref_outputs = self.model.t5_model(
                        decoder_input_ids=decoder_input_ids,
                        encoder_outputs=(grp_hidden_states,),
                        attention_mask=grp_attention_mask,
                    )
                    ref_logits = ref_outputs.logits

        # Score under the sampling temperature so the log-probs belong to the
        # distribution candidates were actually drawn from (top-p truncation
        # remains the usual accepted approximation).
        if self.gen_temperature > 0.0 and self.gen_temperature != 1.0:
            policy_logits = policy_logits / self.gen_temperature
            if ref_logits is not None:
                ref_logits = ref_logits / self.gen_temperature

        simko_k = int(getattr(self.args, "simko_k", 0))
        need_full_distribution = simko_k > 1 or self.args.entropy_coef > 0.0
        policy_log_probs = (
            torch.log_softmax(policy_logits.float(), dim=-1)
            if need_full_distribution
            else None
        )
        token_log_probs = (
            torch.gather(policy_log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            if policy_log_probs is not None
            else selected_token_log_probs(policy_logits, target_ids)
        )

        # Create mask for valid tokens (up to first EOS)
        eos_token_id = self.tokenizer.eos_token_id
        is_eos = (target_ids == eos_token_id)
        cum_eos = torch.cumsum(is_eos.to(torch.int32), dim=1)
        valid_mask = (cum_eos == 0) | ((cum_eos == 1) & is_eos)
        valid_mask = valid_mask.to(device)

        # 6. GRPO loss. With one gradient update per sampled batch, the
        # behavior policy IS the current policy, so the importance ratio is
        # taken against the detached current log-probs: its value is exactly 1
        # (clipping inert, as in single-iteration GRPO) and its gradient is
        # the REINFORCE policy gradient. The previous implementation took the
        # ratio against the adapter-disabled BASE model, which after SFT is
        # several nats away per token: positive-advantage tokens saturated the
        # clip (zero gradient) while negative-advantage tokens got
        # ratio-amplified gradients, so GRPO could suppress behaviors but
        # barely reinforce full-pass candidates.
        # Optional DAPO/GSPO/SimKO-style variants (handoff Section 23.16).
        # NOTE: with single-update-per-batch the ratio is identically 1, so
        # the decoupled clip (clip_eps_high) is inert unless the loop ever
        # moves to multi-update; it is plumbed for completeness.
        scored_log_probs = (
            effective_token_log_probs(policy_log_probs, token_log_probs, advantages, simko_k)
            if simko_k > 1
            else token_log_probs
        )
        eps_low = self.args.clip_eps
        eps_high = self.args.clip_eps_high if self.args.clip_eps_high is not None else eps_low
        old_token_log_probs = scored_log_probs.detach()
        ratio = torch.exp(scored_log_probs - old_token_log_probs)
        surr1 = ratio * advantages.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * advantages.unsqueeze(-1)
        policy_loss = -torch.min(surr1, surr2)

        loss_matrix = policy_loss
        if ref_logits is not None:
            token_ref_log_probs = selected_token_log_probs(ref_logits, target_ids)
            # k3 estimator of KL(policy || ref)
            kl = torch.exp(token_ref_log_probs - token_log_probs) - (token_ref_log_probs - token_log_probs) - 1.0
            loss_matrix = loss_matrix + kl_coef * kl

        pooling = str(getattr(self.args, "loss_pooling", "token")).strip().lower()
        loss = pooled_surrogate_loss(
            loss_matrix,
            valid_mask.float(),
            pooling,
            valid_mask.float().sum().clamp(min=1.0),
            float(target_ids.size(0)),
        )

        entropy_value = torch.tensor(0.0, device=device)
        if self.args.entropy_coef > 0.0:
            # Optional diversity pressure. This is off by default because the
            # full-vocab entropy term adds memory/time pressure on large Qwen
            # decoders; enable only for controlled anti-collapse GRPO trials.
            assert policy_log_probs is not None
            token_entropy = -(policy_log_probs.exp() * policy_log_probs).sum(dim=-1)
            entropy_value = (token_entropy * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1.0)
            loss = loss - self.args.entropy_coef * entropy_value

        # 7. Backward pass. Run RL first so its large decoder graph is freed,
        # then run the independent verified-only CE anchor. Both count as one
        # optimizer micro-batch.
        rl_loss_value = float(loss.detach().cpu().item())
        (loss / self.grad_accum_steps).backward()
        anchor_loss_value = self._backward_sft_anchor(device)
        self._grad_accum_counter += 1
        optimizer_stepped = self._optimizer_step_if_needed(force=False)

        return {
            **base_stats,
            "loss": rl_loss_value + self.sft_anchor_coef * anchor_loss_value,
            "rl_loss": rl_loss_value,
            "sft_anchor_loss": anchor_loss_value,
            "optimizer_stepped": float(optimizer_stepped),
            "grad_accum_counter": float(self._grad_accum_counter),
            "entropy": float(entropy_value.detach().cpu().item()),
            "rl_no_signal": 0.0,
        }

    def _score_chunked_causal(
        self,
        base_stats,
        combined_inputs_embeds,
        forward_mask,
        position_ids,
        target_ids,
        prefix_len,
        prompt_len,
        advantages,
        use_ref,
        kl_coef,
        device,
        chunk_size,
        batch,
    ):
        """Score/backward the generated samples in chunks of `chunk_size`.

        The GRPO loss is a sum over samples divided by ONE global valid-token
        count, so per-chunk backwards against that shared denominator
        accumulate to the same gradient as the single-pass path (chunked
        backward equivalence is covered in grpo_selfcheck.py). Peak activation
        and logits memory are set by chunk_size instead of group size, which
        is what makes group sizes of 16-32 safe on a single GPU.

        When train_graph_glue is on, gradients must reach the SHARED glue graph
        (GNN -> projection -> prefix adapter) that produced the prefix inside
        combined_inputs_embeds. The old approach kept the graph alive with
        retain_graph across chunks -> EVERY chunk's decoder activations stayed
        resident, so peak memory scaled with GROUP size, not chunk size (the
        G32-on-178GB OOM). Instead we detach a scoring leaf: each chunk's
        decoder graph frees right after its own backward (no retain_graph), the
        gradient w.r.t. the prefix accumulates on the leaf, and ONE backward
        through the glue runs after the loop. Peak memory is then truly bounded
        by chunk_size. With the glue frozen the leaf carries no grad and the
        final glue backward is skipped.
        """
        eos_token_id = self.tokenizer.eos_token_id
        is_eos = (target_ids == eos_token_id)
        cum_eos = torch.cumsum(is_eos.to(torch.int32), dim=1)
        valid_mask = ((cum_eos == 0) | ((cum_eos == 1) & is_eos)).to(device)
        denom = valid_mask.float().sum().clamp(min=1.0)

        total_samples = target_ids.size(0)
        total_loss = 0.0
        entropy_total = 0.0
        simko_k = int(getattr(self.args, "simko_k", 0))
        pooling = str(getattr(self.args, "loss_pooling", "token")).strip().lower()
        eps_low = self.args.clip_eps
        eps_high = self.args.clip_eps_high if self.args.clip_eps_high is not None else eps_low

        # Detach the scoring input into a leaf so each chunk's decoder graph
        # frees right after its backward (no retain_graph). The gradient w.r.t.
        # the prefix accumulates on this leaf; a single glue backward after the
        # loop pushes it through the GNN/projection/prefix adapter. This is what
        # actually bounds peak memory by chunk_size (see docstring).
        train_glue = self.train_graph_glue and combined_inputs_embeds.requires_grad
        scoring_embeds = combined_inputs_embeds.detach().requires_grad_(train_glue)

        for start in range(0, total_samples, chunk_size):
            end = min(start + chunk_size, total_samples)
            is_last = end >= total_samples

            policy_outputs = self.model.t5_model(
                inputs_embeds=scoring_embeds[start:end],
                attention_mask=forward_mask[start:end],
                position_ids=position_ids[start:end],
                use_cache=False,
            )
            policy_logits = policy_outputs.logits[:, prefix_len + prompt_len:, :]

            ref_logits = None
            if use_ref:
                with torch.no_grad(), self.model.t5_model.disable_adapter():
                    ref_outputs = self.model.t5_model(
                        inputs_embeds=scoring_embeds[start:end],
                        attention_mask=forward_mask[start:end],
                        position_ids=position_ids[start:end],
                        use_cache=False,
                    )
                    ref_logits = ref_outputs.logits[:, prefix_len + prompt_len:, :]

            if self.gen_temperature > 0.0 and self.gen_temperature != 1.0:
                policy_logits = policy_logits / self.gen_temperature
                if ref_logits is not None:
                    ref_logits = ref_logits / self.gen_temperature

            chunk_targets = target_ids[start:end]
            need_full_distribution = simko_k > 1 or self.args.entropy_coef > 0.0
            policy_log_probs = (
                torch.log_softmax(policy_logits.float(), dim=-1)
                if need_full_distribution
                else None
            )
            token_log_probs = (
                torch.gather(
                    policy_log_probs, dim=-1, index=chunk_targets.unsqueeze(-1)
                ).squeeze(-1)
                if policy_log_probs is not None
                else selected_token_log_probs(policy_logits, chunk_targets)
            )

            scored_log_probs = (
                effective_token_log_probs(
                    policy_log_probs, token_log_probs, advantages[start:end], simko_k
                )
                if simko_k > 1
                else token_log_probs
            )
            old_token_log_probs = scored_log_probs.detach()
            ratio = torch.exp(scored_log_probs - old_token_log_probs)
            chunk_adv = advantages[start:end].unsqueeze(-1)
            surr1 = ratio * chunk_adv
            surr2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * chunk_adv
            loss_matrix = -torch.min(surr1, surr2)

            if ref_logits is not None:
                token_ref_log_probs = selected_token_log_probs(ref_logits, chunk_targets)
                kl = torch.exp(token_ref_log_probs - token_log_probs) - (token_ref_log_probs - token_log_probs) - 1.0
                loss_matrix = loss_matrix + kl_coef * kl

            chunk_valid = valid_mask[start:end].float()
            chunk_loss = pooled_surrogate_loss(
                loss_matrix, chunk_valid, pooling, denom, float(total_samples)
            )

            if self.args.entropy_coef > 0.0:
                assert policy_log_probs is not None
                token_entropy = -(policy_log_probs.exp() * policy_log_probs).sum(dim=-1)
                entropy_contrib = (token_entropy * chunk_valid).sum() / denom
                chunk_loss = chunk_loss - self.args.entropy_coef * entropy_contrib
                entropy_total += float(entropy_contrib.detach().cpu().item())

            # No retain_graph: this chunk's decoder graph is independent and
            # frees now; grad accumulates on the scoring_embeds leaf.
            (chunk_loss / self.grad_accum_steps).backward()
            total_loss += float(chunk_loss.detach().cpu().item())

        # One backward through the shared glue with the accumulated prefix grad.
        # scoring_embeds.grad is already scaled by 1/grad_accum (each chunk's
        # loss was), so glue params get the same scaling as the decoder LoRA.
        if train_glue and scoring_embeds.grad is not None:
            combined_inputs_embeds.backward(scoring_embeds.grad)

        # The chunked RL graph is now released. Recompute a separate SFT anchor
        # graph before the shared optimizer step.
        anchor_loss_value = self._backward_sft_anchor(device)
        self._grad_accum_counter += 1
        optimizer_stepped = self._optimizer_step_if_needed(force=False)

        return {
            **base_stats,
            "loss": total_loss + self.sft_anchor_coef * anchor_loss_value,
            "rl_loss": total_loss,
            "sft_anchor_loss": anchor_loss_value,
            "optimizer_stepped": float(optimizer_stepped),
            "grad_accum_counter": float(self._grad_accum_counter),
            "entropy": entropy_total,
            "rl_no_signal": 0.0,
        }


def _resolve_hf_token() -> str:
    return (
        os.environ.get("GRAPH_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    )


def checkpoint_keys_for_save(model: GraphCodeBERTT5Seq2Seq) -> set[str]:
    """Return updated trainables plus every tensor loaded from the warm start.

    GRPO deliberately freezes the local GraphCodeBERT encoder after loading the
    SFT checkpoint. Saving only ``requires_grad`` tensors therefore used to
    discard its trained LoRA adapter, so evaluation silently fell back to the
    base encoder. Preserve every compatible warm-start tensor and overwrite the
    subset that GRPO updated with their current values.
    """
    model_keys = set(model.state_dict())
    trainable_keys = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    warm_start_keys = set(getattr(model, "_warm_start_checkpoint_keys", ()))
    return (trainable_keys | warm_start_keys) & model_keys


def save_trainable_checkpoint(model: GraphCodeBERTT5Seq2Seq, tokenizer, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    keys_to_save = checkpoint_keys_for_save(model)
    trainable_state_dict = {k: v for k, v in state_dict.items() if k in keys_to_save}
    torch.save(trainable_state_dict, output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
    print(
        f"Saved GRPO checkpoint with {len(trainable_state_dict)} tensors "
        f"({len(getattr(model, '_warm_start_checkpoint_keys', ()))} warm-start keys retained)."
    )
    return output_dir


def upload_checkpoint_to_hf(local_dir: str | Path) -> None:
    repo_id = os.environ.get("GRAPH_HF_REPO", "").strip()
    token = _resolve_hf_token()
    if not repo_id or not token or os.environ.get("GRAPH_HF_UPLOAD_CHECKPOINTS", "0") != "1":
        return

    local_dir = Path(local_dir)
    if not local_dir.exists():
        return

    path_prefix = os.environ.get("GRAPH_HF_PATH_PREFIX", "artifacts/checkpoints").strip("/")
    private = os.environ.get("GRAPH_HF_PRIVATE", "1") == "1"
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(local_dir),
            path_in_repo=f"{path_prefix}/{local_dir.name}",
            commit_message=f"Upload GRPO checkpoint {local_dir.name}",
        )
        print(f"HF checkpoint upload complete: {repo_id}/{path_prefix}/{local_dir.name}")
    except Exception as exc:
        print(f"HF checkpoint upload failed for {local_dir}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    # Larger groups give cleaner group-normalized advantages (G=4 is noisy).
    parser.add_argument('--group_size', type=int, default=int(os.environ.get("GRPO_GROUP_SIZE", "8")))
    # kl_coef defaults to 0: the only available KL reference (adapter-disabled)
    # is the PRE-SFT base model, and anchoring to it actively unlearns SFT.
    parser.add_argument('--kl_coef', type=float, default=float(os.environ.get("GRPO_KL_COEF", "0.0")))
    parser.add_argument(
        '--sft_anchor_coef',
        type=float,
        default=float(os.environ.get("GRPO_SFT_ANCHOR_COEF", "0.0")),
        help=(
            "Mix cross-entropy from a separate, independently replayed RS-SFT pool "
            "into signal-bearing GRPO updates. Requires GRPO_VERIFIED_ANCHOR_FILE."
        ),
    )
    parser.add_argument(
        '--sft_anchor_on_no_signal',
        type=int,
        choices=[0, 1],
        default=int(os.environ.get("GRPO_SFT_ANCHOR_ON_NO_SIGNAL", "0")),
        help="After dynamic resampling is exhausted, optionally run the verified-only anchor; default 0.",
    )
    parser.add_argument(
        '--dynamic_resample_attempts',
        type=int,
        default=int(os.environ.get("GRPO_DYNAMIC_RESAMPLE_ATTEMPTS", "2")),
        help="Resample the same prompt this many additional times when its group has zero reward variance.",
    )
    parser.add_argument(
        '--reward_test_field',
        choices=['tests', 'feedback_tests'],
        default=os.environ.get("GRPO_REWARD_TEST_FIELD", "feedback_tests"),
        help="Harness field exposed to online RL. Hidden acceptance tests are never tokenized into the RL loader.",
    )
    parser.add_argument('--clip_eps', type=float, default=float(os.environ.get("GRPO_CLIP_EPS", "0.2")))
    parser.add_argument('--gen_temperature', type=float, default=float(os.environ.get("GRPO_GEN_TEMPERATURE", "0.7")),
                        help="Rollout sampling temperature; log-prob scoring divides logits by the same value")
    parser.add_argument('--gen_top_p', type=float, default=float(os.environ.get("GRPO_GEN_TOP_P", "0.95")))
    parser.add_argument('--adv_norm', choices=['mean', 'std'], default=os.environ.get("GRPO_ADV_NORM", "mean"),
                        help="mean: reward minus group mean (Dr.GRPO-style). std: original GRPO normalization, "
                             "which inflates noise-level reward gaps in small groups")
    parser.add_argument('--min_reward_range', type=float, default=float(os.environ.get("GRPO_MIN_REWARD_RANGE", "0.05")),
                        help="Groups whose max-min reward is at or below this are skipped (no learning signal)")
    parser.add_argument('--passk_k', type=int, default=int(os.environ.get("GRPO_PASSK_K", "0")),
                        help="If >1, optimize pass@k instead of pass@1: scale each group's advantages by "
                             "(1 - p_hat)^(k-1), p_hat = group perfect-pass rate. Solved prompts stop "
                             "contributing gradient, preserving the diversity pass@k depends on. 0/1 = off")
    parser.add_argument('--score_chunk_size', type=int, default=int(os.environ.get("GRPO_SCORE_CHUNK_SIZE", "0")),
                        help="If >0, run the differentiable scoring forward+backward in chunks of this many "
                             "samples (identical gradients; peak memory bound by chunk size, not group size). "
                             "Required for --group_size 16+ on a single GPU. 0 = single-pass scoring")
    parser.add_argument('--loss_pooling', choices=['token', 'seq'], default=os.environ.get("GRPO_LOSS_POOLING", "token"),
                        help="token: DAPO-style global token-mean (long failures carry more gradient; historical "
                             "behavior). seq: GSPO-style per-sequence mean (every sample counts equally, so short "
                             "passing completions are not outweighed by long rambling failures)")
    parser.add_argument('--simko_k', type=int, default=int(os.environ.get("GRPO_SIMKO_K", "0")),
                        help="LEGACY EXPERIMENTAL top-K positive smoothing. This is not faithful CaSP/SimKO "
                             "(no entropy gate or negative rank-1 term); keep 0 for confirmatory runs")
    parser.add_argument('--overlong_filter', type=int, choices=[0, 1], default=int(os.environ.get("GRPO_OVERLONG_FILTER", "0")),
                        help="DAPO-style: zero the advantage of samples that hit max_new_tokens without EOS, so the "
                             "policy is not taught from truncation artifacts")
    parser.add_argument('--clip_eps_high', type=float, default=(float(os.environ["GRPO_CLIP_EPS_HIGH"]) if os.environ.get("GRPO_CLIP_EPS_HIGH") else None),
                        help="DAPO clip-higher: decoupled upper clip bound. INERT in the current single-update "
                             "loop (ratio is identically 1); plumbed for any future multi-update variant")
    parser.add_argument('--entropy_coef', type=float, default=float(os.environ.get("GRPO_ENTROPY_COEF", "0.0")))
    parser.add_argument('--learning_rate', type=float, default=float(os.environ.get("GRPO_LR", "5e-6")))
    parser.add_argument('--perfect_bonus', type=float, default=float(os.environ.get("GRPO_PERFECT_BONUS", "2.0")))
    parser.add_argument('--batch_size', type=int, default=int(os.environ.get("GRAPH_BATCH_SIZE", "4")))
    parser.add_argument('--grad_accum', type=int, default=int(os.environ.get("GRAPH_GRAD_ACCUM", "1")))
    parser.add_argument('--epochs', type=int, default=int(os.environ.get("GRPO_EPOCHS", "2")))
    parser.add_argument('--max_new_tokens', type=int, default=int(os.environ.get("GRPO_MAX_NEW_TOKENS", "512")))
    parser.add_argument('--unique_test_bonus', type=float, default=float(os.environ.get("GRPO_UNIQUE_TEST_BONUS", "0.0")))
    parser.add_argument('--duplicate_penalty', type=float, default=float(os.environ.get("GRPO_DUPLICATE_PENALTY", "0.0")))
    parser.add_argument('--max_steps', type=int, default=int(os.environ.get("GRAPH_MAX_STEPS", -1)))
    parser.add_argument('--reward_preflight_batches', type=int,
                        default=int(os.environ.get("GRPO_REWARD_PREFLIGHT_BATCHES", "0")),
                        help="Generate and score this many batches without updating or saving a checkpoint")
    parser.add_argument('--output_dir', default=os.environ.get("GRAPH_OUTPUT_DIR", "artifacts/qwen-grpo"))
    parser.add_argument('--checkpoint', default=os.environ.get("GRAPH_CHECKPOINT", ""))
    args = parser.parse_args()
    if args.sft_anchor_coef < 0.0:
        parser.error("--sft_anchor_coef must be non-negative")
    if args.dynamic_resample_attempts < 0:
        parser.error("--dynamic_resample_attempts must be non-negative")
    _dart_per_test_reward.perfect_bonus = args.perfect_bonus
    if _dart_per_test_reward.reward_mode == "verpo":
        if args.group_size <= 1:
            raise SystemExit("VeRPO requires --group_size greater than 1")
        if args.adv_norm != "mean":
            raise SystemExit(
                "VeRPO requires --adv_norm mean (the paper's Fnorm=1 setting)"
            )
        if args.unique_test_bonus != 0.0 or args.duplicate_penalty != 0.0:
            raise SystemExit(
                "VeRPO cannot be combined with unique-test or duplicate shaping; "
                "set both coefficients to 0"
            )
    if not shutil.which("dart"):
        raise SystemExit("GRPO requires Dart on PATH for verifiable reward scoring")
    if not args.checkpoint:
        raise SystemExit("GRPO requires --checkpoint/GRAPH_CHECKPOINT; training from an unverified base is disabled")
    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"GRPO checkpoint not found: {args.checkpoint}")

    config = GraphDecompilerConfig()
    seed = int(os.environ.get("GRAPH_SEED", "42"))
    set_seed(seed)
    # If output_dir was set in environment, prioritize it
    config.output_dir = args.output_dir
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    # GRPO needs records that carry unit tests so the reward can execute them.
    # Default to the test-bearing GRPO corpus; override with GRPO_TRAIN_FILE.
    config.train_file = os.environ.get("GRPO_TRAIN_FILE", "data/testing/grpo_data.jsonl")

    dfg_extractor = LightweightDFGExtractor()

    encoder_revision = os.environ.get("GRAPH_ENCODER_REVISION", "").strip() or None
    encoder_tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_MODEL,
        revision=encoder_revision,
        trust_remote_code=True,
    )
    decoder_model_name = os.environ.get("GRAPH_DECODER_MODEL", "t5-small")
    decoder_revision = os.environ.get("GRAPH_DECODER_REVISION", "").strip() or None
    decoder_tokenizer = AutoTokenizer.from_pretrained(
        decoder_model_name,
        revision=decoder_revision,
        trust_remote_code=True,
    )

    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.eos_token

    tensor_builder = GraphCodeBERTTensorBuilder(encoder_tokenizer, max_seq_len=config.max_input_length)

    print(f"Loading GRPO training dataset: {config.train_file}")
    raw_train_records = load_jsonl_many(config.train_file)
    if not raw_train_records:
        raise SystemExit(f"GRPO training dataset is empty: {config.train_file}")

    reward_field = args.reward_test_field
    train_records = []
    skipped_without_reward = 0
    for index, original in enumerate(raw_train_records):
        try:
            assert_training_approved(original)
        except Exception as exc:
            raise SystemExit(f"unsafe GRPO row {index}: {exc}") from exc
        reward_tests = str(original.get(reward_field) or "")
        if not reward_tests:
            # A one-assertion task may be Phase-0 safe for SFT but has no visible
            # feedback after the hidden split. It must not enter online RL.
            skipped_without_reward += 1
            continue
        row = dict(original)
        # Only the feedback harness is carried into tokenization/collation. The
        # hidden acceptance harness remains on disk for offline harvesting only.
        row["tests"] = reward_tests
        row.pop("acceptance_tests", None)
        train_records.append(row)
    if not train_records:
        raise SystemExit(f"no GRPO rows retain a nonempty {reward_field!r} harness")
    if skipped_without_reward:
        print(
            f"Skipped {skipped_without_reward} Phase-0 rows with no visible "
            f"{reward_field!r} harness; they remain SFT-only."
        )

    # Re-tokenize datasets with standard token types.
    print(f"Tokenizing GRPO rows with reward field {reward_field!r}...")
    train_dataset = tokenize_dataset(train_records, tensor_builder, dfg_extractor, decoder_tokenizer, config)

    def grpo_collate(batch_list):
        return {
            'labels': torch.stack([torch.tensor(x['labels'], dtype=torch.long) for x in batch_list]),
            'decoder_prompt_input_ids': torch.stack([torch.tensor(x['decoder_prompt_input_ids'], dtype=torch.long) for x in batch_list]),
            'decoder_prompt_attention_mask': torch.stack([torch.tensor(x['decoder_prompt_attention_mask'], dtype=torch.long) for x in batch_list]),
            'block_inputs': [x['block_inputs'] for x in batch_list],
            'cfg': [x['cfg'] for x in batch_list],
            'edges': [x['edges'] for x in batch_list],
            'language': [x.get('language', 'dart') for x in batch_list],
            'tests': [x.get('tests', '') for x in batch_list],
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=grpo_collate,
    )

    verified_anchor_loader = None
    verified_anchor_file = os.environ.get("GRPO_VERIFIED_ANCHOR_FILE", "").strip()
    if args.sft_anchor_coef > 0.0:
        if not verified_anchor_file:
            raise SystemExit(
                "GRPO_SFT_ANCHOR_COEF is nonzero but GRPO_VERIFIED_ANCHOR_FILE is unset"
            )
        anchor_records = load_jsonl_many(verified_anchor_file)
        invalid = []
        for index, row in enumerate(anchor_records):
            try:
                assert_training_approved(row)
            except Exception:
                invalid.append(index)
                continue
            if not verified_origin(row):
                invalid.append(index)
        if invalid:
            raise SystemExit(
                f"verified anchor file contains {len(invalid)} invalid rows; first={invalid[:8]}"
            )
        if not anchor_records:
            raise SystemExit("verified anchor file is empty")
        anchor_dataset = tokenize_dataset(
            anchor_records,
            tensor_builder,
            dfg_extractor,
            decoder_tokenizer,
            config,
        )
        verified_anchor_loader = DataLoader(
            anchor_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=grpo_collate,
        )
        print(
            f"Loaded {len(anchor_records)} independently verified anchor rows from "
            f"{verified_anchor_file}"
        )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GraphCodeBERTT5Seq2Seq().to(device)
    print(f"Loading checkpoint from: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    checkpoint_load_report = validate_trainable_checkpoint_load(
        model,
        state_dict,
        missing_keys=missing,
        unexpected_keys=unexpected,
        context="GRPO warm-start checkpoint",
    )
    # Keep the frozen SFT encoder adapter and every other compatible warm-start
    # tensor in all GRPO checkpoints. Their current values are read from the
    # model at save time, so parameters updated by GRPO are not overwritten.
    model._warm_start_checkpoint_keys = frozenset(
        set(state_dict) & set(model.state_dict())
    )
    print(
        "Loaded GRPO checkpoint under a validated architecture contract: "
        f"recognised={checkpoint_load_report['recognised_checkpoint_tensor_count']} "
        f"missing_frozen={checkpoint_load_report['missing_frozen_tensor_count']} "
        f"unexpected={checkpoint_load_report['unexpected_tensor_count']}"
    )

    maybe_override_qwen_prefix_gate(model)

    cast_trainables = 0
    for param in model.parameters():
        if param.requires_grad and param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.float()
            cast_trainables += 1
    non_fp32_trainables = [
        name for name, param in model.named_parameters()
        if param.requires_grad and param.is_floating_point() and param.dtype != torch.float32
    ]
    if non_fp32_trainables:
        raise RuntimeError(
            "GRPO requires FP32 trainable tensors at sub-1e-6 learning rates: "
            + ", ".join(non_fp32_trainables[:8])
        )

    first_trainable = next((p for p in model.parameters() if p.requires_grad), None)
    base_embed_dtype = model.base_decoder_model.get_input_embeddings().weight.dtype
    print(
        f"Sanity: base weights dtype={base_embed_dtype}, "
        f"trainable dtype={getattr(first_trainable, 'dtype', None)}, "
        f"cast_trainable_tensors_to_fp32={cast_trainables}"
    )

    trainer = GRPOTrainer(model, decoder_tokenizer, args, verified_anchor_loader)

    if args.reward_preflight_batches > 0:
        preflight_stats = []
        for batch_index, batch in enumerate(train_loader):
            if batch_index >= args.reward_preflight_batches:
                break
            batch['labels'] = batch['labels'].to(device)
            stats = trainer.train_step(batch, device, update=False)
            preflight_stats.append(stats)
            print(
                f"Reward preflight {batch_index + 1}/{args.reward_preflight_batches} | "
                f"SignalGroups={stats['groups_with_signal']:.3f} | "
                f"Perfect={stats['perfect_rate']:.4f} | "
                f"CompiledObserved={stats['compiled_rate']:.3f} "
                f"(known={stats['compiled_known_fraction']:.3f}) | "
                f"Reward=[{stats['reward_min']:.2f},{stats['reward_max']:.2f}]"
            )
        if not preflight_stats:
            raise RuntimeError("Reward preflight produced no batches")
        aggregate = {
            "schema_version": 1,
            "no_optimizer_update": True,
            "batches": len(preflight_stats),
            "seed": seed,
            "group_size": args.group_size,
            "reward_mode": _dart_per_test_reward.reward_mode,
            "reward": reward_configuration(_dart_per_test_reward),
            "generation": {
                "temperature": args.gen_temperature,
                "top_p": args.gen_top_p,
                "max_new_tokens": args.max_new_tokens,
            },
            "pass_stability_runs": int(os.environ.get("EVAL_PASS_STABILITY_RUNS", "1")),
            "checkpoint": file_record(args.checkpoint),
            "dataset": file_record(config.train_file),
            "reward_implementation": {
                "trainer": file_record(Path(__file__).resolve()),
                "jit_evaluator": file_record(
                    ROOT / "scripts" / "evaluation" / "graph_compile_at_k_antigravity.py"
                ),
            },
            "prompt_schema_version": PROMPT_SCHEMA_VERSION,
            "signal_group_rate_mean": sum(s['groups_with_signal'] for s in preflight_stats) / len(preflight_stats),
            "perfect_sample_rate_mean": sum(s['perfect_rate'] for s in preflight_stats) / len(preflight_stats),
            "compiled_sample_rate_mean": sum(s['compiled_rate'] for s in preflight_stats) / len(preflight_stats),
            "compiled_known_fraction_mean": sum(s['compiled_known_fraction'] for s in preflight_stats) / len(preflight_stats),
            "pass_ratio_mean": sum(s['pass_ratio_mean'] for s in preflight_stats) / len(preflight_stats),
        }
        output_path = Path(config.output_dir) / "reward_preflight.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(json.dumps(aggregate, indent=2))
        print(f"Reward preflight saved to {output_path}; no optimizer update was performed.")
        return

    max_steps = args.max_steps
    step_count = 0
    save_strategy = os.environ.get("GRAPH_SAVE_STRATEGY", "no")
    save_steps = int(os.environ.get("GRAPH_SAVE_STEPS", "0"))
    save_total_limit = int(os.environ.get("GRAPH_SAVE_TOTAL_LIMIT", "2"))
    print("Starting GRPO Reinforcement Learning loop...")
    print(
        f"GRPO optimizer: lr={args.learning_rate} grad_accum={args.grad_accum} "
        f"kl_coef={args.kl_coef} sft_anchor_coef={args.sft_anchor_coef} "
        f"sft_anchor_on_no_signal={args.sft_anchor_on_no_signal} "
        f"dynamic_resample_attempts={args.dynamic_resample_attempts} reward_test_field={args.reward_test_field} "
        f"clip_eps={args.clip_eps} entropy_coef={args.entropy_coef} "
        f"reward_mode={_dart_per_test_reward.reward_mode} "
        f"reward_config={json.dumps(reward_configuration(_dart_per_test_reward), sort_keys=True)} "
        f"unique_test_bonus={args.unique_test_bonus} duplicate_penalty={args.duplicate_penalty} "
        f"gen_temperature={args.gen_temperature} gen_top_p={args.gen_top_p} "
        f"adv_norm={args.adv_norm} min_reward_range={args.min_reward_range} "
        f"passk_k={args.passk_k} score_chunk_size={args.score_chunk_size} "
        f"loss_pooling={args.loss_pooling} legacy_topk_smoothing={args.simko_k} "
        f"overlong_filter={args.overlong_filter} clip_eps_high={args.clip_eps_high}"
    )

    consecutive_errors = 0
    for epoch in range(args.epochs):
        # Progress bar (with ETA) on stderr; the per-step stats line still
        # goes to stdout via progress.write so piped/tee'd logs stay clean.
        progress = None
        batch_iter = train_loader
        if tqdm is not None:
            progress = tqdm(
                train_loader,
                desc=f"GRPO epoch {epoch + 1}/{args.epochs}",
                unit="batch",
                dynamic_ncols=True,
            )
            batch_iter = progress
        log_line = progress.write if progress is not None else print
        for idx, batch in enumerate(batch_iter):
            # Move labels to device
            batch['labels'] = batch['labels'].to(device)

            try:
                attempts_used = 0
                while True:
                    # DAPO-style dynamic sampling: resample this same prompt until
                    # verifier outcomes have relative variance, or exhaust the
                    # bounded retry budget. No ordinary reference target is used.
                    stats = trainer.train_step(
                        batch,
                        device,
                        allow_no_signal_anchor=(
                            attempts_used >= args.dynamic_resample_attempts
                        ),
                    )
                    if not bool(stats.get("skipped_no_signal", 0.0)):
                        break
                    if attempts_used >= args.dynamic_resample_attempts:
                        break
                    attempts_used += 1
                stats["dynamic_resample_attempts_used"] = float(attempts_used)
                stats["resample_attempts"] = float(attempts_used)
                log_line(
                    f"Epoch {epoch+1} | Step {step_count+1} | Batch {idx+1}/{len(train_loader)} | "
                    f"Loss: {stats['loss']:.4f} "
                    f"(RL={stats.get('rl_loss', 0.0):.4f}, "
                    f"SFT={stats.get('sft_anchor_loss', 0.0):.4f}) | "
                    f"Reward: {stats['reward_mean']:.3f} "
                    f"[{stats['reward_min']:.3f}, {stats['reward_max']:.3f}] | "
                    f"RewardStd: {stats['reward_std']:.3f} | "
                    f"SignalGroups: {stats['groups_with_signal']:.2f} | "
                    f"AdvAbs: {stats['advantage_abs_mean']:.3f} | "
                    f"OptStep: {int(stats['optimizer_stepped'])} | "
                    f"Skip: {int(stats.get('skipped_no_signal', 0))} | "
                    f"Resamples: {int(stats.get('dynamic_resample_attempts_used', 0))} | "
                    f"Accum: {int(stats['grad_accum_counter'])}/{args.grad_accum} | "
                    f"PassRatio: {stats['pass_ratio_mean']:.3f}/{stats['pass_ratio_max']:.3f} | "
                    f"CompiledObserved: {stats['compiled_rate']:.2f} "
                    f"(known={stats['compiled_known_fraction']:.2f}) | "
                    f"UniqueBonus: {stats['unique_bonus_mean']:.3f} | "
                    f"DupPenalty: {stats['duplicate_penalty_mean']:.3f} | "
                    f"Entropy: {stats['entropy']:.3f} | "
                    f"Perfect: {stats['perfect_rate']:.2f} | "
                    f"NearPerfect: {stats['near_perfect_rate']:.2f} | "
                    f"HighPartial: {stats['high_partial_rate']:.2f} | "
                    f"ZeroPass: {stats['zero_pass_rate']:.2f} | "
                    f"PasskW: {stats.get('passk_weight_mean', 1.0):.2f}"
                )
                consecutive_errors = 0
            except Exception as e:
                import traceback
                log_line(f"Error at step {step_count+1}: {e}")
                traceback.print_exc()
                trainer.optimizer.zero_grad(set_to_none=True)
                trainer._grad_accum_counter = 0
                raise RuntimeError("GRPO step failed; aborting to avoid a stale or partial update") from e

            step_count += 1
            if save_strategy == "steps" and save_steps > 0 and step_count % save_steps == 0:
                ckpt_dir = Path(config.output_dir) / f"checkpoint-step-{step_count}"
                save_trainable_checkpoint(model, decoder_tokenizer, ckpt_dir)
                upload_checkpoint_to_hf(ckpt_dir)
                if save_total_limit > 0:
                    checkpoints = sorted(Path(config.output_dir).glob("checkpoint-step-*"), key=lambda p: p.stat().st_mtime)
                    for old_ckpt in checkpoints[:-save_total_limit]:
                        shutil.rmtree(old_ckpt, ignore_errors=True)
            if max_steps > 0 and step_count >= max_steps:
                break
        if progress is not None:
            progress.close()
        flushed = trainer._optimizer_step_if_needed(force=True)
        if flushed:
            print(f"Flushed remaining accumulated GRPO gradients at end of epoch {epoch + 1}.")
        if save_strategy == "epoch":
            ckpt_dir = Path(config.output_dir) / f"checkpoint-epoch-{epoch + 1}-step-{step_count}"
            save_trainable_checkpoint(model, decoder_tokenizer, ckpt_dir)
            upload_checkpoint_to_hf(ckpt_dir)
            if save_total_limit > 0:
                checkpoints = sorted(Path(config.output_dir).glob("checkpoint-epoch-*"), key=lambda p: p.stat().st_mtime)
                for old_ckpt in checkpoints[:-save_total_limit]:
                    shutil.rmtree(old_ckpt, ignore_errors=True)
        if max_steps > 0 and step_count >= max_steps:
            break

    print(f"Saving GRPO-tuned model to {config.output_dir}...")
    save_trainable_checkpoint(model, decoder_tokenizer, config.output_dir)
    provenance = {
        "schema_version": 1,
        "stage": "grpo",
        "prompt_schema_version": PROMPT_SCHEMA_VERSION,
        "scoring_tests_visible_to_policy": False,
        "seed": seed,
        "models": {
            "decoder": {
                "requested_id": decoder_model_name,
                "requested_revision": decoder_revision,
                "resolved_commit": model_commit(model.base_decoder_model.config),
            },
            "encoder": {
                "requested_id": ENCODER_MODEL,
                "requested_revision": encoder_revision,
                "resolved_commit": model_commit(model.local_encoder.encoder.config),
            },
        },
        "datasets": [file_record(config.train_file)],
        "verified_anchor_file": file_record(verified_anchor_file, required=False) if verified_anchor_file else None,
        "reward_test_field": args.reward_test_field,
        "dynamic_resample_attempts": args.dynamic_resample_attempts,
        "warm_start_checkpoint": file_record(args.checkpoint, required=False) if args.checkpoint else None,
        "checkpoint_load_contract": checkpoint_load_report,
        "saved_checkpoint": file_record(Path(config.output_dir) / "pytorch_model.bin"),
        "checkpoint_tensor_policy": {
            "warm_start_keys_retained": len(
                getattr(model, "_warm_start_checkpoint_keys", ())
            ),
            "saved_tensor_count": len(checkpoint_keys_for_save(model)),
            "local_encoder_frozen_during_grpo": True,
        },
        "source_files": [
            file_record(Path(__file__)),
            file_record(ROOT / "scripts/evaluation/graph_compile_at_k_antigravity.py"),
            file_record(ROOT / "scripts/training/graph_encoder_decoder_decompiler_v2_antigravity.py"),
            file_record(ROOT / "models/hierarchical_graph_encoder_antigravity.py"),
            file_record(ROOT / "models/graphcodebert_tensor_builder.py"),
            file_record(ROOT / "models/pyg_cfg_dataset.py"),
            file_record(ROOT / "scripts/data/cfg_extractor.py"),
            file_record(ROOT / "scripts/data/dfg_extractor.py"),
        ],
        "reward_configuration": reward_configuration(_dart_per_test_reward),
        "grpo_arguments": vars(args),
        "graph_environment": graph_environment(),
        "git": git_state(ROOT),
        "runtime": runtime_record(),
    }
    write_json(Path(config.output_dir) / "run_provenance.json", provenance)
    print("GRPO training completed successfully.")

if __name__ == '__main__':
    main()
