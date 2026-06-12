"""实验三: 多AGV可扩展性 — 2台到8台AGV"""
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

def run_experiment_3(agv_counts=None, steps=300, output_dir="logs/metrics"):
    if agv_counts is None:
        agv_counts = [2, 4, 6, 8]
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for n_agv in agv_counts:
        import scheduler.task_allocator as ta
        ta.NUM_AGVS = n_agv
        ta.AGV_INITIAL_POSITIONS = ta.AGV_INITIAL_POSITIONS[:n_agv]

        env = WarehouseEnv()
        env.reset()
        task_alloc = TaskAllocator(env.loading_zones, env.unloading_zones)
        mapf = MAPFPlanner(env.grid, env.width, env.height)
        rl_avoid = RLCollisionAvoidance()
        controller = AGVController(env, mapf, rl_avoid)
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
        results.append({
            'num_agvs': n_agv,
            'tasks_completed': stats['tasks_completed'],
            'steps_taken': stats['steps_taken'],
            'completion_rate': stats['tasks_completed'] / max(steps, 1),
        })

    path = os.path.join(output_dir, "exp3_scalability.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    for r in results:
        print(f"  {r['num_agvs']} AGVs: {r['tasks_completed']} tasks")
    return results

if __name__ == "__main__":
    run_experiment_3(steps=100)
