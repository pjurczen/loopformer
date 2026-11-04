import pytest
import torch
from support import LOOP_MAX, VOCAB_SIZE, after_training_steps, next_token_loss, random_tokens, tiny_model

from checkpoint import load_checkpoint, save_checkpoint
from trajectory import evaluation_schedule, long_trajectory


def nonzero_gradient(parameters) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters)


@pytest.mark.parametrize("loops", [1, 2])
def test_a_changed_token_leaves_the_logits_before_it_unchanged(loops):
    model = after_training_steps(tiny_model(), random_tokens())
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


def test_after_a_few_steps_every_parameter_receives_gradient():
    model = after_training_steps(tiny_model(), random_tokens())

    next_token_loss(model, random_tokens(), long_trajectory(LOOP_MAX)).backward()

    for name, parameter in model.named_parameters():
        assert nonzero_gradient([parameter]), name


@pytest.mark.parametrize("loops", [1, 2])
def test_a_saved_and_reloaded_model_gives_identical_logits(tmp_path, loops):
    trained = after_training_steps(tiny_model(seed=0), random_tokens())
    fresh = tiny_model(seed=1)
    tokens = random_tokens(seed=2)
    schedule = evaluation_schedule(LOOP_MAX, loops)
    path = tmp_path / "ckpt.pt"
    iteration = 7

    save_checkpoint(trained, torch.optim.AdamW(trained.parameters()), iteration, path)
    loaded_iteration, _ = load_checkpoint(path, fresh, torch.optim.AdamW(fresh.parameters()))

    assert loaded_iteration == iteration
    torch.testing.assert_close(fresh(tokens, schedule)[0], trained(tokens, schedule)[0])
