#!/usr/bin/env python3
"""Benchmark chart for the README from the released model cards (weights/*.json): success and mobs killed per level,
by difficulty, on 200 unseen mazes each.

  python scripts/plot_benchmark.py            # -> docs/media/benchmark.png
"""

import json
import sys
from pathlib import Path

from plotly.subplots import make_subplots

sys.path.insert(0, "/home/berke/.claude/style")
from plotly_style import save  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = [
    ("cornichon_agent", "cornichon_agent (arena, then game)", "#2a7a3b"),
    ("cornichon_ppo_v4", "cornichon_ppo_v4 (game only)", "#9a9a9a"),
]


def main():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=("levels finished (%)", "mobs killed per level"))
    for name, label, color in MODELS:
        rows = json.loads((ROOT / "weights" / f"{name}.json").read_text())["benchmark"]["per_difficulty"]
        x = [r["difficulty"] for r in rows]
        success = [100 * r["success"] for r in rows]
        fig.add_bar(x=x, y=success, name=label, marker_color=color, text=[f"{s:.0f}" for s in success],
                    textposition="outside", row=1, col=1)
        fig.add_bar(x=x, y=[r["mobs_killed"] for r in rows], name=label, marker_color=color, showlegend=False,
                    text=[f"{r['mobs_killed']:.1f}" for r in rows], textposition="outside", row=1, col=2)
    fig.update_layout(
        title=dict(text="Does arena training make a better Cornichon player?<br><sup>200 unseen mazes per difficulty, "
                        "actions sampled from the policy. Difficulty = the game's level number (maze size and mob "
                        "density both grow).</sup>", x=0.02),
        template="plotly_white", font=dict(family="DejaVu Sans", size=16), barmode="group",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"), margin=dict(t=120, b=110, l=60, r=30),
    )
    fig.update_xaxes(title_text="difficulty", dtick=1)
    fig.update_yaxes(range=[0, 105], row=1, col=1)
    fig.update_yaxes(range=[0, 16], row=1, col=2)
    save(fig, ROOT / "docs" / "media" / "benchmark", width=1400, height=560, scale=2)
    print("wrote docs/media/benchmark.png")


if __name__ == "__main__":
    main()
