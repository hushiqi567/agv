"""实验一: 单AGV RL vs 传统A* — 同一地图同样起点终点"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import random, time, csv
from path_planning.rl.dqn_agent import DQNAgent, ACTIONS, ACTION_DELTAS
from path_planning.mapf_planner import a_star_search


def run_experiment_1(num_trials=50, map_size=30, max_steps=500,
                     model_path=None, output_dir="logs/metrics"):
    """单AGV RL vs A* 对比实验"""
    os.makedirs(output_dir, exist_ok=True)
    agent = DQNAgent(grid_size=15)
    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
    agent.set_training(False)

    results = []
    for trial in range(num_trials):
        empty = [(x,y) for x in range(1,map_size-1) for y in range(1,map_size-1)]
        start = random.choice(empty)
        goal = random.choice([c for c in empty if c != start])
        available = [c for c in empty if c not in (start, goal)]
        obstacles = random.sample(available, min(5, len(available)))
        grid = [[0]*map_size for _ in range(map_size)]
        for ox, oy in obstacles:
            grid[oy][ox] = 1

        t0 = time.time()
        astar_path = a_star_search(start, goal, grid, map_size, map_size)
        astar_time = time.time() - t0
        astar_len = len(astar_path) if astar_path else -1

        agent.encoder.map_width = map_size
        agent.encoder.map_height = map_size
        pos = start
        rl_path_len = 0
        rl_success = False
        t0 = time.time()
        for step in range(max_steps):
            local, gvec = agent.encoder.encode(pos, goal, grid, [], [])
            valid = []
            for a in range(5):
                dx, dy = ACTION_DELTAS[ACTIONS[a]]
                nx, ny = pos[0]+dx, pos[1]+dy
                if 0 <= nx < map_size and 0 <= ny < map_size and grid[ny][nx] != 1:
                    valid.append(a)
            if not valid:
                valid = [4]
            action = agent.select_action(local, gvec, valid)
            dx, dy = ACTION_DELTAS[ACTIONS[action]]
            pos = (pos[0]+dx, pos[1]+dy)
            rl_path_len += 1
            if pos == goal:
                rl_success = True
                break
        rl_time = time.time() - t0

        results.append({
            'trial': trial, 'start_x': start[0], 'start_y': start[1],
            'goal_x': goal[0], 'goal_y': goal[1],
            'astar_path_len': astar_len, 'astar_time': astar_time,
            'rl_path_len': rl_path_len, 'rl_time': rl_time,
            'rl_success': rl_success,
            'ratio': rl_path_len/astar_len if astar_len > 0 else -1,
        })

    path = os.path.join(output_dir, "exp1_rl_vs_astar.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    successes = sum(1 for r in results if r['rl_success'])
    ratios = [r['ratio'] for r in results if r['ratio'] > 0]
    avg_ratio = sum(ratios)/len(ratios) if ratios else -1
    print(f"Experiment 1: RL success={successes}/{num_trials} ({successes/num_trials:.1%}), avg ratio={avg_ratio:.3f}")
    return results


if __name__ == "__main__":
    run_experiment_1(num_trials=10)
