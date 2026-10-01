"""Training configuration: dataclass defaults, overridden by a YAML file, overridden by `key.sub=value` CLI args."""

from dataclasses import asdict, dataclass, field, fields, is_dataclass

import yaml

from .curriculum import CurriculumConfig
from .reward import RewardConfig


@dataclass
class EnvConfig:
    num_envs: int = 64
    envs_per_service: int = 16  # levels per Java process; processes simulate in parallel
    # Time limit per level = time_base + time_per_tile * shortest path at the start, capped at max_steps (decisions).
    # Difficulty 1 (~55 tiles) gets ~630, difficulty 10 (~175 tiles) 1350. time_per_tile 0 = always max_steps.
    max_steps: int = 1500
    time_base: int = 300
    time_per_tile: int = 6
    repeat: int = 4  # game frames per decision
    view_width: int = 31
    view_height: int = 21


@dataclass
class PPOConfig:
    total_steps: int = 50_000_000
    horizon: int = 128  # decisions per env per update
    epochs: int = 4
    minibatches: int = 8
    lr: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


@dataclass
class ModelConfig:
    recurrent: bool = False  # LSTM memory; lets the agent remember where it has been
    hidden: int = 256


@dataclass
class EvalConfig:
    every_updates: int = 50
    episodes: int = 100  # held-out seeds per evaluation, at the current curriculum difficulty
    sweep_every_updates: int = 200  # also evaluate every difficulty up to current + 1 (forgetting / readiness)
    sweep_episodes: int = 20  # held-out seeds per difficulty in a sweep
    greedy: bool = True


@dataclass
class TrainConfig:
    name: str = "ppo"
    seed: int = 1
    device: str = "cuda"
    torch_threads: int = 4  # torch's default (all cores) is ~300x slower on a loaded shared node
    out_dir: str = "/data/local/berke/cornichon-rl/runs"  # node-local; run_dir = out_dir/name
    wandb_mode: str = "online"
    wandb_project: str = "cornichon-rl"
    wandb_entity: str = "berkegokmen1"
    checkpoint_every: int = 10  # updates
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    reward: dict = field(default_factory=dict)  # RewardConfig fields to override

    def reward_config(self):
        return RewardConfig.from_dict(self.reward)


def _merge(obj, values, where="config"):
    for key, value in values.items():
        names = {f.name for f in fields(obj)}
        if key not in names:
            raise ValueError(f"unknown key {where}.{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _merge(current, value, f"{where}.{key}")
        elif isinstance(current, dict):
            current.update(value)
        elif isinstance(current, float) and not isinstance(value, bool):
            setattr(obj, key, float(value))  # YAML reads 1e-4 (no dot) as a string
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(obj, key, int(value))
        else:
            setattr(obj, key, value)


def load_config(path=None, overrides=()):
    config = TrainConfig()
    if path:
        with open(path) as f:
            _merge(config, yaml.safe_load(f) or {})
    for item in overrides:
        dotted, raw = item.split("=", 1)
        *parents, last = dotted.split(".")
        nested = {last: yaml.safe_load(raw)}
        for parent in reversed(parents):
            nested = {parent: nested}
        _merge(config, nested)
    config.reward_config()  # fail early on bad reward keys
    return config


def to_dict(config):
    return asdict(config)
