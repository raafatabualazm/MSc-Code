"""Local self-check for the GRPO trainer math (no GPU, no Dart, no HF models).

Imports the REAL graph_grpo_decompiler_antigravity module with its heavy
dependencies (transformers, torch_geometric, datasets, model glue) stubbed
out, then asserts:

  1. group_advantages: no-signal guard, mean vs std normalization, and the
     std blow-up case the mean default protects against.
  2. cumsum_position_ids matches HF generate()'s attention-mask convention.
  3. The detached-old importance ratio has the exact REINFORCE gradient and
     the clip is inert at ratio 1 (single-update GRPO).
  4. TruePerTestReward shaped/binary reward values (execution monkeypatched).
  5. calculate_rewards group-level unique-test bonus and duplicate penalty.
  6. passk_advantage_weights pass@k gradient scaling and the per-sample
     perfect_flags it consumes (alignment with the rewards tensor).
  7. Chunked-scoring equivalence: per-chunk backwards against one global
     denominator (retain_graph on the shared upstream graph) accumulate
     the same gradients as a single full-batch backward.
  8. DAPO/GSPO and legacy top-K smoothing options: token vs seq pooling
     arithmetic and chunk-decomposability; top-K credit applies only to
     positive-advantage samples.
  9. Fused selected-token log probabilities match explicit log-softmax/gather.
 10. GRPO checkpoints retain frozen tensors loaded from the SFT warm start.
 11. A partial gradient-accumulation window is rescaled to its actual mean
     before the forced epoch-end optimizer step.

Run:  python scripts/training/grpo_selfcheck.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


class _Dummy:  # placeholder for classes the self-check never instantiates
    def __init__(self, *args, **kwargs):
        pass


def _import_trainer():
    _stub_module("datasets", Dataset=_Dummy)
    tg = _stub_module("torch_geometric")
    tg_data = _stub_module("torch_geometric.data", Batch=_Dummy)
    tg.data = tg_data
    _stub_module(
        "transformers",
        AutoTokenizer=_Dummy,
        AutoModelForCausalLM=_Dummy,
        PreTrainedTokenizerBase=_Dummy,
        PreTrainedTokenizerFast=_Dummy,
        PreTrainedModel=_Dummy,
        set_seed=lambda seed: None,
    )
    _stub_module("models.pyg_cfg_dataset", cfg_to_pyg=lambda *a, **k: None)
    _stub_module(
        "models.hierarchical_graph_encoder_antigravity",
        LocalBlockEncoder=_Dummy,
        GraphPoolingEncoder=_Dummy,
    )
    _stub_module("scripts.data.dfg_extractor", LightweightDFGExtractor=_Dummy)
    _stub_module("models.graphcodebert_tensor_builder", GraphCodeBERTTensorBuilder=_Dummy)
    _stub_module(
        "scripts.training.graph_encoder_decoder_decompiler_v2_antigravity",
        GraphDecompilerConfig=_Dummy,
        GraphCodeBERTT5Seq2Seq=_Dummy,
        PROMPT_SCHEMA_VERSION="antigravity-v2-no-test-hints",
        canonicalize_source=lambda s: s,
        load_jsonl_many=lambda *paths: [],
        tokenize_dataset=lambda *a, **k: None,
        maybe_override_qwen_prefix_gate=lambda *a, **k: None,
    )
    # NOT stubbed: scripts.training.graph_positions (torch-only) — the
    # cumsum_position_ids check below must exercise the real implementation.

    import importlib.util

    path = ROOT / "scripts" / "training" / "graph_grpo_decompiler_antigravity.py"
    spec = importlib.util.spec_from_file_location("grpo_trainer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_group_advantages(m):
    # All-equal rewards: no signal, zero advantages.
    adv, mask = m.group_advantages(torch.tensor([1.0, 1.0, 1.0, 1.0]), 1, 4)
    assert not bool(mask.any())
    assert torch.allclose(adv, torch.zeros(4))

    # Mixed group, mean norm: advantage == reward - group mean.
    rewards = torch.tensor([4.5, -1.0, -1.0, -2.0])
    adv, mask = m.group_advantages(rewards, 1, 4, norm="mean")
    assert bool(mask.all())
    assert torch.allclose(adv, rewards - rewards.mean(), atol=1e-6)

    # Noise-level gap: std norm blows a 0.25 reward gap into a -1.5 advantage,
    # mean norm keeps it at noise scale. This is why mean is the default.
    noise = torch.tensor([-1.0, -1.0, -1.25, -1.0])
    adv_std, _ = m.group_advantages(noise, 1, 4, norm="std")
    adv_mean, _ = m.group_advantages(noise, 1, 4, norm="mean")
    assert adv_std.abs().max() > 1.4, adv_std
    assert adv_mean.abs().max() < 0.2, adv_mean

    # min_reward_range gates the same group out entirely.
    adv_gated, mask_gated = m.group_advantages(noise, 1, 4, min_reward_range=0.3)
    assert not bool(mask_gated.any())
    assert torch.allclose(adv_gated, torch.zeros(4))

    # Two groups: only the second has signal; the first is zeroed.
    two = torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0, -1.0, 0.0, 0.0])
    adv2, mask2 = m.group_advantages(two, 2, 4)
    assert mask2.tolist() == [False, True]
    assert torch.allclose(adv2[:4], torch.zeros(4))
    assert torch.allclose(adv2[4:], torch.tensor([1.0, -1.0, 0.0, 0.0]), atol=1e-6)
    print("group_advantages: OK")


def check_position_ids(m):
    mask = torch.tensor([[1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1]])
    pos = m.cumsum_position_ids(mask)
    # HF generate convention: cumsum-1 over the mask, pads -> 1.
    assert pos[0].tolist() == [0, 1, 1, 1, 2, 3], pos[0]
    assert pos[1].tolist() == [0, 1, 2, 3, 4, 5], pos[1]
    print("cumsum_position_ids: OK")


def check_ratio_trick():
    torch.manual_seed(0)
    logp_a = torch.randn(4, 7, requires_grad=True)
    logp_b = logp_a.detach().clone().requires_grad_(True)
    adv = torch.randn(4, 1)
    mask = torch.ones(4, 7)
    mask[:, 5:] = 0.0

    # Loss as implemented: clipped surrogate with ratio vs detached self.
    ratio = torch.exp(logp_a - logp_a.detach())
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
    loss_impl = (-torch.min(surr1, surr2) * mask).sum() / mask.sum()
    loss_impl.backward()

    # Plain REINFORCE policy gradient.
    loss_pg = (-(logp_b * adv) * mask).sum() / mask.sum()
    loss_pg.backward()

    assert torch.allclose(ratio.detach(), torch.ones_like(ratio))
    assert torch.allclose(logp_a.grad, logp_b.grad, atol=1e-6)
    print("ratio trick gradient == REINFORCE gradient: OK")


_TEST_CODE = """
final candidate = foo;
void main() {
  expect(candidate(1), 2);
  expect(candidate(2), 3);
  expect(candidate(3), 4);
  expect(candidate(4), 5);
}
void expect(dynamic a, dynamic b) { if (a != b) throw 'fail'; }
"""

_SOLUTION = "int foo(int x) {\n  return x + 1;\n}"


def _make_reward(m, mode="shaped"):
    r = m.TruePerTestReward()
    r.reward_mode = mode
    r.no_compile_penalty = -2.0
    r.compile_reward = 0.0
    r.partial_reward_cap = 3.0
    r.perfect_base_reward = 3.0
    r.perfect_bonus = 1.5
    r.binary_fail_reward = -1.0
    r.empty_code_penalty = -3.0
    return r


def _patch_execution(r, compiled=True, passed=4, total=4):
    r._compile_candidate_with_tests = lambda *a, **k: compiled
    r._run_full_test_harness = lambda *a, **k: compiled and passed == total
    r._evaluate_binary_harness = (
        lambda *a, **k: (compiled, compiled and passed == total, "selfcheck")
    )
    results = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "individual_results": [{"passed": i < passed} for i in range(total)],
    }
    r._parse_and_run_individual_tests = lambda *a, **k: results
    return r


def check_reward_parser_guards(m):
    import shutil as _shutil
    if not _shutil.which("dart"):
        m.shutil.which = lambda name: "dart" if name == "dart" else None

    reward = _make_reward(m, "binary")
    try:
        reward.compute_reward(_SOLUTION, "void main() {}")
    except ValueError as exc:
        assert "zero expect" in str(exc)
    else:
        raise AssertionError("zero-test reward row did not fail closed")

    test_with_set = _TEST_CODE + "\nvoid expectSet(Set a, Set b) { if (a.length != b.length) throw 'fail'; }\n"
    helpers = reward._extract_helper_functions_safe(test_with_set)
    assert "void expect(" in helpers
    assert "void expectSet(" in helpers
    print("reward parser fail-closed / expect* helper extraction: OK")


def check_reward_shapes(m):
    import shutil as _shutil
    if not _shutil.which("dart"):
        # compute_reward_details exits early without dart; fake its presence.
        m.shutil.which = lambda name: "dart" if name == "dart" else None

    r = _patch_execution(_make_reward(m, "shaped"), passed=4)
    assert r.compute_reward(_SOLUTION, _TEST_CODE) == 4.5  # 3.0 + 1.5 full pass

    r = _patch_execution(_make_reward(m, "shaped"), passed=2)
    assert r.compute_reward(_SOLUTION, _TEST_CODE) == 1.5  # 0.0 + 3.0 * 0.5

    r = _patch_execution(_make_reward(m, "shaped"), passed=0)
    assert r.compute_reward(_SOLUTION, _TEST_CODE) == 0.0  # compiles, passes none

    r = _patch_execution(_make_reward(m, "shaped"), compiled=False)
    assert r.compute_reward(_SOLUTION, _TEST_CODE) == -2.0  # no compile

    r = _make_reward(m, "shaped")
    assert r.compute_reward("x", _TEST_CODE) == -3.0  # empty/tiny

    r = _patch_execution(_make_reward(m, "binary"), passed=4)
    assert r.compute_reward(_SOLUTION, _TEST_CODE) == 4.5

    # Binary mode must execute the shared pass-aligned evaluator once and must
    # not reconstruct/run individual assertions. It must receive the raw
    # completion so evaluator-side code extraction stays identical.
    calls = {"compile": 0, "full": 0, "aligned": 0, "individual": 0}
    r = _make_reward(m, "binary")
    r._compile_candidate_with_tests = (
        lambda *a, **k: calls.__setitem__("compile", calls["compile"] + 1) or True
    )
    r._run_full_test_harness = (
        lambda *a, **k: calls.__setitem__("full", calls["full"] + 1) or True
    )
    raw_with_prose = "Here is the Dart implementation:\n" + _SOLUTION
    seen = {}
    def aligned(code, tests):
        calls["aligned"] += 1
        seen["code"] = code
        seen["tests"] = tests
        return True, True, ""
    r._evaluate_binary_harness = aligned
    r._parse_and_run_individual_tests = (
        lambda *a, **k: calls.__setitem__("individual", calls["individual"] + 1)
    )
    assert r.compute_reward(raw_with_prose, _TEST_CODE) == 4.5
    assert calls == {"compile": 0, "full": 0, "aligned": 1, "individual": 0}, calls
    assert seen == {"code": raw_with_prose, "tests": _TEST_CODE}

    r = _patch_execution(_make_reward(m, "binary"), passed=3)
    failed_details = r.compute_reward_details(_SOLUTION, _TEST_CODE)
    assert failed_details["compiled"] is True
    assert failed_details["status"] == "full_harness_fail"

    r = _patch_execution(_make_reward(m, "binary"), passed=3)
    assert r.compute_reward(_SOLUTION, _TEST_CODE) == -1.0  # binary fail

    r = _patch_execution(_make_reward(m, "binary"), compiled=False)
    assert r.compute_reward(_SOLUTION, _TEST_CODE) == -1.0
    print("TruePerTestReward shaped/binary values and exact-harness path: OK")


class _ScriptedReward:
    def __init__(self, details):
        self.details = list(details)
        self.calls = 0

    def compute_reward_details(self, comp, test_code):
        d = self.details[self.calls]
        self.calls += 1
        return d


def check_group_shaping(m):
    test_passes = [
        [True, False, False, False],   # c0
        [True, False, False, False],   # c1 (same test as c0 -> not unique)
        [False, True, True, False],    # c2 (two uniquely passed tests)
        [False, False, False, False],  # c3
    ]
    details = [
        {"reward": 0.0, "pass_ratio": sum(p) / 4, "passed": sum(p), "total": 4,
         "compiled": True, "test_passes": p, "status": "tested"}
        for p in test_passes
    ]
    saved = m._dart_per_test_reward
    m._dart_per_test_reward = _ScriptedReward(details)
    try:
        completions = ["aaa bbb", "aaa  bbb", "ccc", "ddd"]  # c0/c1 duplicates
        rewards = m.calculate_rewards(
            completions,
            references=[""] * 4,
            languages=["dart"] * 4,
            tests=[_TEST_CODE] * 4,
            group_size=4,
            unique_test_bonus=0.75,
            duplicate_penalty=0.25,
        )
    finally:
        m._dart_per_test_reward = saved

    # unique bonus: c2 passes tests 1 and 2 uniquely -> 0.75 * 2/4 = 0.375
    # duplicate penalty: c0 and c1 each have 1 duplicate -> 0.25 * 1/3
    expected = torch.tensor([-0.25 / 3, -0.25 / 3, 0.375, 0.0])
    assert torch.allclose(rewards, expected, atol=1e-6), rewards
    print("calculate_rewards unique-test bonus / duplicate penalty: OK")


def check_parallel_binary_reward_order(m):
    import time

    scorer = _make_reward(m, "binary")

    def details(comp, _tests):
        index = int(comp.rsplit("_", 1)[1])
        time.sleep((3 - index) * 0.005)
        passed = index % 2 == 0
        return {
            "reward": 1.0 if passed else -1.0,
            "pass_ratio": 1.0 if passed else 0.0,
            "passed": 4 if passed else 0,
            "total": 4,
            "compiled": passed,
            "test_passes": [passed] * 4,
            "status": "test",
        }

    scorer.compute_reward_details = details
    saved = m._dart_per_test_reward
    saved_workers = m.GRPO_REWARD_WORKERS
    m._dart_per_test_reward = scorer
    m.GRPO_REWARD_WORKERS = 4
    try:
        rewards = m.calculate_rewards(
            [f"candidate_{i}" for i in range(4)],
            references=[""] * 4,
            languages=["dart"] * 4,
            tests=[_TEST_CODE] * 4,
        )
        assert torch.allclose(rewards, torch.tensor([1.0, -1.0, 1.0, -1.0]))
        try:
            m.calculate_rewards(["candidate_0"], [""], ["dart", "dart"], [_TEST_CODE])
        except ValueError as exc:
            assert "misaligned" in str(exc)
        else:
            raise AssertionError("misaligned reward inputs did not fail closed")
    finally:
        m._dart_per_test_reward = saved
        m.GRPO_REWARD_WORKERS = saved_workers
    print("parallel binary reward ordering / alignment guard: OK")


def check_passk_weights(m):
    flags = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # k <= 1 means off: weights identically 1.
    assert torch.allclose(m.passk_advantage_weights(flags, 1, 4, 0), torch.ones(4))
    assert torch.allclose(m.passk_advantage_weights(flags, 1, 4, 1), torch.ones(4))

    # Fully solved group -> weight 0: no more sharpening on solved prompts.
    assert torch.allclose(m.passk_advantage_weights(torch.ones(4), 1, 4, 5), torch.zeros(4))
    # Never-solved group -> full weight 1.
    assert torch.allclose(m.passk_advantage_weights(torch.zeros(4), 1, 4, 5), torch.ones(4))

    # p_hat = 1/4, k = 5 -> (3/4)^4 for every sample in the group.
    w = m.passk_advantage_weights(flags, 1, 4, 5)
    assert torch.allclose(w, torch.full((4,), 0.75 ** 4)), w

    # Two groups get independent weights: p=0.5 -> 0.5^4, p=0 -> 1.
    two = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    w2 = m.passk_advantage_weights(two, 2, 4, 5)
    assert torch.allclose(w2, torch.tensor([0.0625] * 4 + [1.0] * 4)), w2
    print("passk_advantage_weights: OK")


def check_perfect_flags(m):
    details = [
        {"reward": 4.5, "pass_ratio": 1.0, "passed": 4, "total": 4,
         "compiled": True, "test_passes": [True] * 4, "full_pass": True,
         "status": "tested"},
        {"reward": 1.5, "pass_ratio": 0.5, "passed": 2, "total": 4,
         "compiled": True, "test_passes": [True, True, False, False],
         "full_pass": False, "status": "tested"},
        {"reward": -2.0, "pass_ratio": None, "passed": 0, "total": 4,
         "compiled": False, "test_passes": [], "full_pass": False,
         "status": "no_compile"},
    ]
    saved = m._dart_per_test_reward
    m._dart_per_test_reward = _ScriptedReward(details)
    try:
        rewards, stats = m.calculate_rewards(
            ["solution candidate one", "solution candidate two", "solution candidate three", "x"],
            references=[""] * 4,
            languages=["dart"] * 4,
            tests=[_TEST_CODE] * 3 + [None],  # last one: tiny-output path, no tests
            return_stats=True,
            group_size=4,
        )
    finally:
        m._dart_per_test_reward = saved

    # Only an explicit full-harness pass is perfect; None pass_ratio and the
    # tiny-output path must both produce 0.0, keeping 1:1 alignment.
    assert stats["perfect_flags"] == [1.0, 0.0, 0.0, 0.0], stats["perfect_flags"]
    assert len(stats["perfect_flags"]) == int(rewards.shape[0])
    print("calculate_rewards perfect_flags alignment: OK")


def check_chunked_backward_equivalence():
    """Mirror _score_chunked_causal's structure on a toy graph: a SHARED
    upstream tensor (like the graph-prefix embeddings) computed once, fresh
    per-chunk ops on row slices, per-chunk backward with retain_graph until
    the last chunk, one global denominator. Gradients must equal the
    single-pass backward exactly."""
    torch.manual_seed(2)
    n_samples, hidden, vocab = 8, 4, 6
    adv = torch.randn(n_samples, 1)
    base_shared = torch.randn(n_samples, hidden)
    base_w = torch.randn(hidden, vocab)
    targets = torch.randint(0, vocab, (n_samples, 1))

    def full_pass():
        shared = base_shared.clone().requires_grad_(True)
        w = base_w.clone().requires_grad_(True)
        logits = shared @ w  # shared upstream graph, computed once
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        tok = torch.gather(log_probs, dim=-1, index=targets)
        ratio = torch.exp(tok - tok.detach())
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
        loss = (-torch.min(surr1, surr2)).sum() / float(n_samples)
        loss.backward()
        return shared.grad.clone(), w.grad.clone(), float(loss.detach())

    def chunked_pass(chunk):
        shared = base_shared.clone().requires_grad_(True)
        w = base_w.clone().requires_grad_(True)
        logits = shared @ w  # computed ONCE, sliced per chunk
        total = 0.0
        for start in range(0, n_samples, chunk):
            end = min(start + chunk, n_samples)
            log_probs = torch.log_softmax(logits[start:end].float(), dim=-1)
            tok = torch.gather(log_probs, dim=-1, index=targets[start:end])
            ratio = torch.exp(tok - tok.detach())
            surr1 = ratio * adv[start:end]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * adv[start:end]
            chunk_loss = (-torch.min(surr1, surr2)).sum() / float(n_samples)
            chunk_loss.backward(retain_graph=end < n_samples)
            total += float(chunk_loss.detach())
        return shared.grad.clone(), w.grad.clone(), total

    g_shared_full, g_w_full, loss_full = full_pass()
    for chunk in (3, 4, 8):
        g_shared_c, g_w_c, loss_c = chunked_pass(chunk)
        assert torch.allclose(g_shared_c, g_shared_full, atol=1e-7), chunk
        assert torch.allclose(g_w_c, g_w_full, atol=1e-7), chunk
        assert abs(loss_c - loss_full) < 1e-6, (chunk, loss_c, loss_full)
    print("chunked scoring backward == single-pass backward: OK")


def check_dapo_simko_options(m):
    torch.manual_seed(3)
    # pooled_surrogate_loss: token vs seq arithmetic, chunk-decomposable both ways.
    lm = torch.randn(4, 5)
    vm = torch.ones(4, 5)
    vm[1, 3:] = 0.0
    vm[3, 1:] = 0.0
    token_denom = vm.sum()
    full_token = m.pooled_surrogate_loss(lm, vm, "token", token_denom, 4.0)
    assert torch.allclose(full_token, (lm * vm).sum() / token_denom)
    full_seq = m.pooled_surrogate_loss(lm, vm, "seq", token_denom, 4.0)
    expected_seq = ((lm * vm).sum(dim=1) / vm.sum(dim=1).clamp(min=1.0)).sum() / 4.0
    assert torch.allclose(full_seq, expected_seq)
    chunked_seq = (
        m.pooled_surrogate_loss(lm[:2], vm[:2], "seq", token_denom, 4.0)
        + m.pooled_surrogate_loss(lm[2:], vm[2:], "seq", token_denom, 4.0)
    )
    assert torch.allclose(chunked_seq, full_seq, atol=1e-6)
    chunked_token = (
        m.pooled_surrogate_loss(lm[:3], vm[:3], "token", token_denom, 4.0)
        + m.pooled_surrogate_loss(lm[3:], vm[3:], "token", token_denom, 4.0)
    )
    assert torch.allclose(chunked_token, full_token, atol=1e-6)

    # effective_token_log_probs: k<=1 identity; positive rows take the top-K
    # mean; negative and zero-advantage rows keep the sampled-token log-prob.
    lp = torch.log_softmax(torch.randn(3, 4, 7), dim=-1)
    tok = lp[..., 0]
    adv = torch.tensor([1.0, -1.0, 0.0])
    assert torch.allclose(m.effective_token_log_probs(lp, tok, adv, 0), tok)
    assert torch.allclose(m.effective_token_log_probs(lp, tok, adv, 1), tok)
    eff = m.effective_token_log_probs(lp, tok, adv, 3)
    topk_mean = lp.topk(3, dim=-1).values.mean(dim=-1)
    assert torch.allclose(eff[0], topk_mean[0])
    assert torch.allclose(eff[1], tok[1])
    assert torch.allclose(eff[2], tok[2])
    print("DAPO/GSPO/legacy top-K smoothing options: OK")


def check_selected_token_log_probs(m):
    torch.manual_seed(17)
    logits = torch.randn(2, 5, 11, requires_grad=True)
    targets = torch.randint(0, 11, (2, 5))
    expected = torch.gather(
        torch.log_softmax(logits.float(), dim=-1),
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)
    actual = m.selected_token_log_probs(logits, targets)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    print("memory-bounded selected-token log-probs: OK")


def check_partial_accumulation_flush(m):
    trainer = object.__new__(m.GRPOTrainer)
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.model.weight.data.zero_()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=1.0)
    trainer.grad_accum_steps = 8
    trainer._grad_accum_counter = 2

    # Two unit-gradient microbatches, each divided by 8, leave grad=0.25.
    # Forced flush must multiply by 8/2 so the update uses their mean (=1).
    trainer.model.weight.grad = torch.tensor([[0.25]])
    assert trainer._optimizer_step_if_needed(force=False) is False
    assert torch.allclose(trainer.model.weight, torch.zeros_like(trainer.model.weight))
    assert trainer._optimizer_step_if_needed(force=True) is True
    assert torch.allclose(trainer.model.weight, torch.tensor([[-1.0]]), atol=1e-6)
    assert trainer._grad_accum_counter == 0
    print("partial gradient-accumulation flush scaling: OK")


def check_warm_start_checkpoint_retention(m):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.updated = torch.nn.Parameter(torch.tensor([1.0]))
            self.frozen_from_sft = torch.nn.Parameter(
                torch.tensor([2.0]), requires_grad=False
            )
            self.unused_frozen_base = torch.nn.Parameter(
                torch.tensor([3.0]), requires_grad=False
            )

    model = TinyModel()
    model._warm_start_checkpoint_keys = frozenset({"frozen_from_sft"})
    keys = m.checkpoint_keys_for_save(model)
    assert keys == {"updated", "frozen_from_sft"}, keys
    print("warm-start checkpoint tensor retention: OK")


def main():
    m = _import_trainer()
    check_group_advantages(m)
    check_position_ids(m)
    check_ratio_trick()
    check_reward_shapes(m)
    check_reward_parser_guards(m)
    check_group_shaping(m)
    check_parallel_binary_reward_order(m)
    check_passk_weights(m)
    check_perfect_flags(m)
    check_chunked_backward_equivalence()
    check_dapo_simko_options(m)
    check_selected_token_log_probs(m)
    check_warm_start_checkpoint_retention(m)
    check_partial_accumulation_flush(m)
    print("\nALL GRPO SELF-CHECKS PASSED")


if __name__ == "__main__":
    main()
