import numpy as np
import pytest

from cornichon_rl.service import SimService, SimulatorError, decode_grid

WALK_RIGHT = [2, 0, 0, 0, 0]


@pytest.fixture
def service():
    s = SimService(2)
    yield s
    s.close()


def play(service, seed, actions):
    row = service.reset([0], [seed], [1])[0]
    rows = [row]
    for action in actions:
        service.send_step([action, [0, 0, 0, 0, 0]], 4)
        rows.append(service.read_step()[0])
    return rows


def test_same_seed_same_trajectory(service):
    rng = np.random.default_rng(0)
    actions = [[int(rng.integers(n)) for n in (3, 2, 2, 3, 3)] for _ in range(60)]
    first, second = play(service, 11, actions), play(service, 11, actions)
    assert first == second


def test_level_is_the_real_maze(service):
    row = service.reset([0], [3], [1])[0]
    grid = decode_grid(row, service.view_height, service.view_width)
    walls_in_view = grid[0].sum()
    assert 0 < walls_in_view < grid[0].size  # bricks around, but not solid
    assert 0 < row["door_distance"] < 10_000  # door reachable from the start cell
    snapshot = service.snapshot(0)
    assert snapshot["width"] > 20 and snapshot["height"] > 10


def test_different_seeds_give_different_mazes(service):
    a = service.reset([0], [1], [1])[0]
    b = service.reset([0], [2], [1])[0]
    assert a["grid"] != b["grid"] or a["door_distance"] != b["door_distance"]


def test_player_does_not_fall_through_the_floor(service):
    rows = play(service, 5, [WALK_RIGHT] * 100)
    assert all(r["y"] > 0 for r in rows)


def test_bad_request_reports_error_and_service_survives(service):
    with pytest.raises(SimulatorError):
        service.call({"op": "nonsense"})
    assert service.reset([1], [4], [1])[0]["seed"] == 4


def test_service_exits_when_client_closes_pipe():
    s = SimService(1)
    s.proc.stdin.close()
    assert s.proc.wait(timeout=10) == 0
