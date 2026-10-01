# Cluster training

* Simulation is CPU-bound (Java + Box2D); the GPU only runs a small CNN. A run needs < 2 GB of GPU memory, so it
  can share a card. 64 envs = 4 Java processes.
* Nodes need a JDK (`openjdk-17-jdk-headless`) and the venv on local disk (see README_RL.md). The repo and the
  built simulator are on NFS home, shared by every node.
* `scripts/train.sh GPU NAME` starts a tmux session; runs live in `/data/local/berke/cornichon-rl/runs/NAME/`
  (`latest.pt`, `best.pt`, `config.json`), logs in `/data/local/berke/cornichon-rl/logs/`. Both node-local.
* Set `torch_threads` (default 4). torch's default of one thread per core ran 300× slower on a loaded node.
