"""实验四: 对比传统方法 — RL vs CBS vs Random"""
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

POLICIES = ["rl", "cbs", "random"]

def run_experiment_4(policies=None, steps=200, output_dir="logs/metrics"):
    if policies is None:
        policies = POLICIES
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for policy in policies:
        env = WarehouseEnv()
        env.reset()
        task_alloc = TaskAllocator(env.loading_zones, env.unloading_zones)
        mapf = MAPFPlanner(env.grid, env.width, env.height)
        rl_avoid = RLCollisionAvoidance()
        controller = AGVController(env, mapf, rl_avoid)

        if policy == "random":
            controller.use_rl_primary = False
            controller.use_random_policy = True
        elif policy == "cbs":
            controller.use_rl_primary = False
        else:
            controller.use_rl_primary = True

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
        dl_stats = controller.deadlock_detector.get_stats()
        results.append({
            'policy': policy,
            'tasks_completed': stats['tasks_completed'],
            'steps_taken': stats['steps_taken'],
            'deadlocks': dl_stats['deadlock_count'],
        })

    path = os.path.join(output_dir, "exp4_comparison.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    for r in results:
        print(f"  {r['policy']}: tasks={r['tasks_completed']}, deadlocks={r['deadlocks']}")
    return results

if __name__ == "__main__":
    run_experiment_4(steps=100)
