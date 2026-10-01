"""Actor-critic: a small CNN over the local grid plus an MLP over the state vector, one categorical head per
action dimension and a value head."""

import numpy as np
import torch
from torch import nn

from .service import ACTION_NVEC, GRID_CHANNELS


def _init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, view_height=21, view_width=31, state_size=17, hidden=256):
        super().__init__()
        self.grid = nn.Sequential(
            _init(nn.Conv2d(len(GRID_CHANNELS), 32, 3, padding=1)),
            nn.ReLU(),
            _init(nn.Conv2d(32, 64, 3, stride=2, padding=1)),
            nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            grid_features = self.grid(torch.zeros(1, len(GRID_CHANNELS), view_height, view_width)).shape[1]
        self.state = nn.Sequential(_init(nn.Linear(state_size, 64)), nn.ReLU())
        self.trunk = nn.Sequential(
            _init(nn.Linear(grid_features + 64, hidden)),
            nn.ReLU(),
            _init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
        )
        # Small policy init keeps the first policy close to uniform.
        self.heads = nn.ModuleList(_init(nn.Linear(hidden, n), gain=0.01) for n in ACTION_NVEC)
        self.value = _init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, obs):
        x = self.trunk(torch.cat([self.grid(obs["grid"]), self.state(obs["state"])], dim=1))
        return [head(x) for head in self.heads], self.value(x).squeeze(1)

    def distributions(self, obs):
        logits, value = self(obs)
        return [torch.distributions.Categorical(logits=l) for l in logits], value

    @torch.no_grad()
    def act(self, obs, greedy=False):
        dists, value = self.distributions(obs)
        if greedy:
            action = torch.stack([d.probs.argmax(1) for d in dists], 1)
        else:
            action = torch.stack([d.sample() for d in dists], 1)
        log_prob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        return action, log_prob, value


def to_tensors(obs, device):
    return {key: torch.as_tensor(value, device=device) for key, value in obs.items()}
