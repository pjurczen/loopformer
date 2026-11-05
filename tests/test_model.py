from dataclasses import replace

import pytest
import torch
from support import (
    LOOP_MAX,
    TINY_CONFIG,
    VARIANTS,
    VOCAB_SIZE,
    after_training_steps,
    next_token_loss,
    random_tokens,
    tiny_model,
)

from checkpoint import load_checkpoint, save_checkpoint
from model import LoopFormerLM
from trajectory import evaluation_schedule, long_trajectory


def nonzero_gradient(parameters) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
def test_on_the_apple_gpu_the_compiled_model_gives_the_eager_model_s_loss_and_gradients():
    tokens = random_tokens().to("mps")
    eager = after_training_steps(LoopFormerLM(TINY_CONFIG, device="mps"), tokens)
    compiled = LoopFormerLM(TINY_CONFIG, device="mps")
    compiled.load_state_dict(eager.state_dict())
    compiled.compile_for_device("mps", scope="whole")
    schedule = long_trajectory(LOOP_MAX)

    eager_loss = next_token_loss(eager, tokens, schedule)
    eager_loss.backward()
    compiled_loss = next_token_loss(compiled, tokens, schedule)
    compiled_loss.backward()

    torch.testing.assert_close(compiled_loss, eager_loss, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(compiled.blocks[0].ffn.w1.weight.grad, eager.blocks[0].ffn.w1.weight.grad, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("loops", [1, 2])
def test_a_changed_token_leaves_the_logits_before_it_unchanged(variant, loops):
    model = after_training_steps(tiny_model(variant=variant), random_tokens())
    tokens = random_tokens(seed=1)
    changed_position = 9
    changed_tokens = tokens.clone()
    changed_tokens[:, changed_position] = (tokens[:, changed_position] + 1) % VOCAB_SIZE
    schedule = evaluation_schedule(LOOP_MAX, loops)

    logits, _ = model(tokens, schedule)
    changed_logits, _ = model(changed_tokens, schedule)

    torch.testing.assert_close(changed_logits[:, :changed_position], logits[:, :changed_position])
    assert not torch.allclose(changed_logits[:, changed_position], logits[:, changed_position])


def test_at_initialization_the_gradient_reaches_only_the_modulators_embeddings_and_final_norm():
    model = tiny_model()

    next_token_loss(model, random_tokens(), long_trajectory(LOOP_MAX)).backward()

    for block in model.blocks:
        assert not nonzero_gradient(block.attn.parameters())
        assert not nonzero_gradient(block.ffn.parameters())
        assert nonzero_gradient(block.modulator.parameters())
    assert nonzero_gradient(model.lm_head.parameters())
    assert nonzero_gradient(model.ln_final.parameters())


def test_a_fresh_gate_model_passes_its_input_through_so_every_route_gives_the_same_logits():
    model = tiny_model(variant="gate")
    tokens = random_tokens()

    one_loop, _ = model(tokens, evaluation_schedule(LOOP_MAX, 1))
    full_route, _ = model(tokens, long_trajectory(LOOP_MAX))

    torch.testing.assert_close(one_loop, full_route)


def test_a_fresh_additive_model_ignores_the_timing_of_its_loops():
    model = LoopFormerLM(replace(TINY_CONFIG, loop_max=4, conditioning="additive"))
    tokens = random_tokens()

    coarse_first, _ = model(tokens, [0.5, 0.25, 0.25])
    fine_first, _ = model(tokens, [0.25, 0.25, 0.5])

    torch.testing.assert_close(coarse_first, fine_first)


def test_muon_gets_the_block_matrices_and_nothing_from_outside_the_blocks():
    model = tiny_model()

    muon = {id(p) for p in model.muon_parameters()}

    assert all(p.ndim == 2 for p in model.muon_parameters())
    for block in model.blocks:
        assert id(block.attn.qkv_proj.weight) in muon
        assert id(block.ffn.w1.weight) in muon
        assert id(block.modulator.weight) in muon
        assert id(block.modulator.bias) not in muon
    assert id(model.lm_head.weight) not in muon
    assert all(id(p) not in muon for p in model.time_embedding.parameters())


def test_lora_factors_stay_out_of_muon():
    model = tiny_model(variant="additive-lora")

    muon = {id(p) for p in model.muon_parameters()}
    lora = [p for name, p in model.named_parameters() if "lora" in name]

    assert lora
    assert all(id(p) not in muon for p in lora)


@pytest.mark.parametrize("variant", VARIANTS)
def test_after_a_few_steps_every_parameter_receives_gradient(variant):
    model = after_training_steps(tiny_model(variant=variant), random_tokens())

    next_token_loss(model, random_tokens(), long_trajectory(LOOP_MAX)).backward()

    for name, parameter in model.named_parameters():
        assert nonzero_gradient([parameter]), name


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("loops", [1, 2])
def test_a_saved_and_reloaded_model_gives_identical_logits(tmp_path, variant, loops):
    trained = after_training_steps(tiny_model(seed=0, variant=variant), random_tokens())
    fresh = tiny_model(seed=1, variant=variant)
    tokens = random_tokens(seed=2)
    schedule = evaluation_schedule(LOOP_MAX, loops)
    path = tmp_path / "ckpt.pt"
    iteration = 7

    save_checkpoint(trained, torch.optim.AdamW(trained.parameters()), iteration, path)
    loaded_iteration, _ = load_checkpoint(path, fresh, torch.optim.AdamW(fresh.parameters()))

    assert loaded_iteration == iteration
    torch.testing.assert_close(fresh(tokens, schedule)[0], trained(tokens, schedule)[0])
