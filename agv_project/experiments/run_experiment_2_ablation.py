"""实验二: 消融实验 — 4种配置对比"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import csv
from env.warehouse_env import WarehouseEnv
from scheduler.task_allocator import TaskAllocator
from path_planning.mapf_planner import MAPFPlanner
from path_planning.rl_collision_avoidance import RLCollisionAvoidance
from path_planning.agv_controller import AGVController

CONFIGS = {
    "full":          {"rl_primary": True,  "curriculum": True,  "deadlock": True},
    "no_curriculum": {"rl_primary": True,  "curriculum": False, "deadlock": True},
    "no_avoidance":  {"rl_primary": False, "curriculum": True,  "deadlock": True},
    "rl_only":       {"rl_primary": True,  "curriculum": False, "deadlock": False},
}

def run_ablation(config_name, config, steps=200):
    env = WarehouseEnv()
    env.reset()
    task_alloc = TaskAllocator(env.loading_zones, env.unloading_zones)
    mapf = MAPFPlanner(env.grid, env.width, env.height)
    rl_avoid = RLCollisionAvoidance()
    controller = AGVController(env, mapf, rl_avoid)
    controller.use_rl_primary = config["rl_primary"]
    controller.set_task_allocator(task_alloc)
    task_alloc.set_controller(controller)
    controller.reset()
    task_alloc.reset()

    for _ in range(steps):
        env.step()
        mapf.update_grid(env.grid)
        task_alloc.step()
        controller.step()

    stats = controller.get_statistics()
    return {
        'config': config_name,
        'tasks_completed': stats['tasks_completed'],
        'total_steps': stats['steps_taken'],
    }

def run_experiment_2(steps=200, output_dir="logs/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for name, cfg in CONFIGS.items():
        print(f"Running ablation: {name}")
        r = run_ablation(name, cfg, steps=steps)
        results.append(r)

    path = os.path.join(output_dir, "exp2_ablation.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    for r in results:
        print(f"  {r['config']}: tasks={r['tasks_completed']}")
    return results

if __name__ == "__main__":
    run_experiment_2(steps=100)
