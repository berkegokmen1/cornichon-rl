"""Actor-critic: a small CNN over the local grid plus an MLP over the state vector, optionally an LSTM (memory of
where the agent has been, which the 31x21 local view cannot show), then one categorical head per action dimension
and a value head.

Every call takes and returns a recurrent `state` ((h, c) or None for the feed-forward model) and a `starts` mask
(1 where an env begins a new episode on this step), so callers do not need to know which kind they hold.
"""

import numpy as np
import torch
from torch import nn

from .service import ACTION_NVEC, GRID_CHANNELS


def _init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, view_height=21, view_width=31, state_size=20, hidden=256, recurrent=False, grid_channels=None):
        super().__init__()
        self.state_size = state_size
        self.grid_channels = grid_channels or len(GRID_CHANNELS)
        self.recurrent = recurrent
        self.hidden = hidden
        self.grid = nn.Sequential(
            _init(nn.Conv2d(self.grid_channels, 32, 3, padding=1)),
            nn.ReLU(),
            _init(nn.Conv2d(32, 64, 3, stride=2, padding=1)),
            nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, stride=2, padding=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            grid_features = self.grid(torch.zeros(1, self.grid_channels, view_height, view_width)).shape[1]
        self.state = nn.Sequential(_init(nn.Linear(state_size, 64)), nn.ReLU())
        self.trunk = nn.Sequential(
            _init(nn.Linear(grid_features + 64, hidden)),
            nn.ReLU(),
            _init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
        )
        if recurrent:
            self.lstm = nn.LSTMCell(hidden, hidden)
            for name, param in self.lstm.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.zeros_(param)
        # Small policy init keeps the first policy close to uniform.
        self.heads = nn.ModuleList(_init(nn.Linear(hidden, n), gain=0.01) for n in ACTION_NVEC)
        self.value = _init(nn.Linear(hidden, 1), gain=1.0)

    def initial_state(self, batch, device):
        if not self.recurrent:
            return None
        zeros = torch.zeros(batch, self.hidden, device=device)
        return zeros, zeros.clone()

    def _encode(self, obs):
        # channels and state features are only ever appended, so older models read a prefix
        grid = obs["grid"][:, : self.grid_channels]
        return self.trunk(torch.cat([self.grid(grid), self.state(obs["state"][:, : self.state_size])], dim=1))

    def _memory(self, x, state, starts):
        if not self.recurrent:
            return x, None
        keep = (1.0 - starts.float()).unsqueeze(1)
        h, c = self.lstm(x, (state[0] * keep, state[1] * keep))
        return h, (h, c)

    def forward(self, obs, state=None, starts=None):
        """One step for a batch of envs -> (logits per head, value, new state)."""
        x = self._encode(obs)
        if self.recurrent:
            if starts is None:
                starts = torch.zeros(x.shape[0], device=x.device)
            x, state = self._memory(x, state, starts)
        return [head(x) for head in self.heads], self.value(x).squeeze(1), state

    def forward_sequence(self, obs, state, starts):
        """obs leaves shaped (T, B, ...), starts (T, B), state at t=0 -> (logits per head, value), flattened to T*B."""
        steps, batch = starts.shape
        x = self._encode({k: v.flatten(0, 1) for k, v in obs.items()})
        if self.recurrent:
            x = x.view(steps, batch, -1)
            outputs = []
            for t in range(steps):
                out, state = self._memory(x[t], state, starts[t])
                outputs.append(out)
            x = torch.stack(outputs).flatten(0, 1)
        return [head(x) for head in self.heads], self.value(x).squeeze(1)

    @torch.no_grad()
    def act(self, obs, state=None, starts=None, greedy=False):
        """-> action (B, 5), log prob, value, new state."""
        logits, value, state = self(obs, state, starts)
        dists = [torch.distributions.Categorical(logits=l) for l in logits]
        if greedy:
            action = torch.stack([d.probs.argmax(1) for d in dists], 1)
        else:
            action = torch.stack([d.sample() for d in dists], 1)
        log_prob = sum(d.log_prob(action[:, i]) for i, d in enumerate(dists))
        return action, log_prob, value, state


def load_widened(model, state_dict):
    """Load weights from a model that saw fewer grid channels or state features (they are only ever appended).
    The new inputs get zero weights, so the loaded policy acts exactly as before until training uses them."""
    state_dict = dict(state_dict)
    for key in ("grid.0.weight", "state.0.weight"):
        old, new = state_dict[key], model.state_dict()[key]
        if old.shape != new.shape:
            widened = torch.zeros_like(new)
            widened[:, : old.shape[1]] = old
            state_dict[key] = widened
    model.load_state_dict(state_dict)


def to_tensors(obs, device):
    return {key: torch.as_tensor(value, device=device) for key, value in obs.items()}
