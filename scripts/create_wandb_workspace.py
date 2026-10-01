#!/usr/bin/env python3
"""Publish the curated W&B workspace for cornichon-rl. Re-run whenever logged metrics change (cornichon_rl/ppo.py)."""

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY, PROJECT = "berkegokmen1", "cornichon-rl"

READ_ME = """\
**What is training:** PPO plays the real Cornichon game (headless Java, Box2D, the game's own maze generator).
One episode = one level; the episode ends at the door (**success**), on death, or at the time limit: 300 + 6 decisions per tile of path to the door (timeout).
The sphere rests on the player's head and blocks jumps unless it is moved aside first, so climbing needs both.
The curriculum starts at difficulty 1 and moves up one difficulty when the last 200 episodes at the current one
succeed at least 80% of the time (`curriculum/difficulty`). 20% of episodes replay easier difficulties.

**OBJECTIVES (gradients flow through these):** `loss/total` = `loss/policy` + 0.5 `loss/value` − 0.01 `loss/entropy`.
The reward the policy maximises is the sum of the `reward_parts/*` terms.

**DIAGNOSTICS (logged only, nothing is optimised on them):**
- `rollout/*`: training episodes that finished during the update (`rollout_window/*` = last 200). **`rollout/success` is the number to watch.**
- `reward_parts/*`: per-episode sum of each reward term; shows which term drives the return.
- `eval/*`: the policy (sampled actions, from v3 on; greedy before) on 100 **held-out mazes** (seeds ≥ 1,000,000, never trained on) at the current difficulty.
  Drops right after a promotion are expected: the difficulty just went up.
- `eval_sweep/dNN/*`: every 200 updates, 20 held-out mazes at **every** difficulty up to current + 1: shows
  forgetting of easier levels and readiness for the next one. Difficulty = bigger maze, longer route, more enemies
  (d1: 29x17, 6 mobs; d10: 65x45, 131 mobs and no potions, the original game's formula).
- **Arena runs** (`arena_*`, phase 1): door closed, success = every mob killed, and "difficulty" is the mob density
  (2 to 10) on small mazes. Phase-2 runs start from arena weights. Experiment log: `EXPERIMENTS.md` in the repo.
- `ppo/*`: optimizer health. approx KL ≲ 0.02 and clip fraction ≲ 0.2 are normal; explained variance → 1 means the critic tracks returns.
- `system/*`: throughput. The x axis everywhere is environment steps (agent decisions).
"""


def line(title, *metrics):
    return wr.LinePlot(
        title=title,
        x="Step",
        y=list(metrics),
        smoothing_type="exponentialTimeWeighted",
        smoothing_factor=0.6,
        smoothing_show_original=True,
    )


def section(name, panels, opened=True, rows=2):
    return ws.Section(name=name, panels=panels, is_open=opened, layout_settings=ws.SectionLayoutSettings(columns=3, rows=rows))


workspace = ws.Workspace(
    name="Cornichon RL - objectives and diagnostics",
    entity=ENTITY,
    project=PROJECT,
    auto_generate_panels=False,
    settings=ws.WorkspaceSettings(x_axis="Step", max_runs=10, tooltip_number_of_runs="all_runs"),
    runset_settings=ws.RunsetSettings(
        pinned_columns=[
            "run:displayName",
            "summary:curriculum/difficulty",
            "summary:rollout/success",
            "summary:eval/success",
            "summary:eval/difficulty",
            "summary:system/env_steps",
        ],
    ),
    sections=[
        section("00 READ ME", [wr.MarkdownPanel(READ_ME)], rows=3),
        section(
            "01 Losses and Objectives",
            [
                line("total loss (optimised)", "loss/total"),
                line("policy loss (clipped surrogate)", "loss/policy"),
                line("value loss", "loss/value"),
                line("policy entropy (bonus)", "loss/entropy"),
            ],
        ),
        section(
            "02 Success and Curriculum",
            [
                line("training success rate", "rollout/success", "rollout_window/success"),
                line("curriculum difficulty", "curriculum/difficulty"),
                line("success over promotion window", "curriculum/success_window"),
                line("how episodes end", "rollout/success", "rollout/death", "rollout/timeout"),
                line("episode return", "rollout/return", "rollout_window/return"),
                line("episode length (decisions)", "rollout/length"),
                line("tiles gained toward the door per episode", "rollout/progress", "rollout_window/progress"),
            ],
        ),
        section(
            "03 Reward Components",
            [
                line(
                    "reward terms per episode",
                    "reward_parts/level_complete",
                    "reward_parts/complete_kill_fraction",
                    "reward_parts/death",
                    "reward_parts/path_progress",
                    "reward_parts/mob_killed",
                    "reward_parts/damage_dealt",
                    "reward_parts/damage",
                    "reward_parts/collected",
                    "reward_parts/step",
                ),
                line("path progress shaping", "reward_parts/path_progress"),
                line("combat: kills per episode", "rollout/mobs_killed", "eval/mobs_killed"),
                line("combat: fraction of the level's mobs killed", "rollout/kill_fraction", "eval/kill_fraction"),
                line("arena share of training episodes (phase 1 = 1)", "rollout/arena"),
                line("combat: sphere damage dealt vs damage taken", "rollout/damage_dealt", "rollout/damage"),
                line("potions collected", "rollout/collected"),
                line("game score (not trained on)", "rollout/score"),
            ],
        ),
        section(
            "04 Held-out Evaluation",
            [
                line("held-out success (greedy)", "eval/success"),
                line("held-out difficulty", "eval/difficulty"),
                line("held-out outcomes", "eval/success", "eval/death", "eval/timeout"),
                line("held-out return", "eval/return"),
                line("held-out episode length", "eval/length"),
                line("held-out tiles gained toward the door", "eval/progress"),
            ],
        ),
        section(
            "04b Every Difficulty (sweep)",
            [
                line("held-out success per difficulty", *[f"eval_sweep/d{d:02d}/success" for d in range(1, 11)]),
                line("held-out progress per difficulty", *[f"eval_sweep/d{d:02d}/progress" for d in range(1, 11)]),
                line("held-out deaths per difficulty", *[f"eval_sweep/d{d:02d}/death" for d in range(1, 11)]),
                line("held-out timeouts per difficulty", *[f"eval_sweep/d{d:02d}/timeout" for d in range(1, 11)]),
                line("held-out kill fraction per difficulty", *[f"eval_sweep/d{d:02d}/kill_fraction" for d in range(1, 11)]),
            ],
        ),
        section(
            "05 PPO Health",
            [
                line("approx KL", "ppo/approx_kl"),
                line("clip fraction", "ppo/clip_frac"),
                line("explained variance (critic)", "ppo/explained_variance"),
                line("learning rate", "ppo/lr"),
                line("advantage std", "ppo/advantage_std"),
            ],
            opened=False,
        ),
        section(
            "06 System",
            [
                line("env steps per second (update)", "system/sps"),
                line("env steps per second (rollout only)", "system/rollout_sps"),
                line("updates", "system/update"),
            ],
            opened=False,
        ),
    ],
)

if __name__ == "__main__":
    workspace.save()
    print(workspace.url)
