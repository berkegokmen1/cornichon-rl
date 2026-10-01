import torch

from cornichon_rl.model import ActorCritic


def random_obs(*lead):
    return {"grid": (torch.rand(*lead, 10, 21, 31) > 0.8).float(), "state": torch.randn(*lead, 17)}


def test_sequence_replay_matches_step_by_step_acting():
    torch.manual_seed(0)
    model = ActorCritic(recurrent=True)
    T, B = 12, 3
    obs = random_obs(T, B)
    starts = torch.zeros(T, B)
    starts[0] = 1
    starts[5, 1] = 1  # env 1 starts a new episode mid-sequence
    state0 = (torch.randn(B, 256), torch.randn(B, 256))

    state, step_values = state0, []
    for t in range(T):
        _, value, state = model({k: v[t] for k, v in obs.items()}, state, starts[t])
        step_values.append(value)
    _, sequence_values = model.forward_sequence(obs, state0, starts)
    assert torch.allclose(torch.stack(step_values).flatten(), sequence_values, atol=1e-5)


def test_memory_resets_at_episode_start():
    torch.manual_seed(0)
    model = ActorCritic(recurrent=True)
    obs = random_obs(2)
    fresh = model(obs, model.initial_state(2, "cpu"), torch.ones(2))[1]
    stale = (torch.randn(2, 256), torch.randn(2, 256))
    assert torch.allclose(model(obs, stale, torch.ones(2))[1], fresh)
    assert not torch.allclose(model(obs, stale, torch.zeros(2))[1], fresh)


def test_feed_forward_ignores_memory_arguments():
    model = ActorCritic()
    obs = random_obs(4)
    assert model.initial_state(4, "cpu") is None
    action, log_prob, value, state = model.act(obs, None, torch.ones(4))
    assert action.shape == (4, 5) and state is None
