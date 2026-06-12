"""在真实仓库仿真中训练RL模型 — 局部避撞"""
import sys
import os
import argparse
import logging
import time
import numpy as np

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from env.warehouse_env import WarehouseEnv
from scheduler.od_flow import ODFlowManager
from scheduler.task_allocator import TaskAllocator
from path_planning.mapf_planner import MAPFPlanner
from path_planning.rl_collision_avoidance import RLCollisionAvoidance
from path_planning.agv_controller import AGVController
from path_planning.rl.dqn_agent import DQNAgent

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S')

def run_training_episode(env, task_alloc, mapf, rl_avoid, controller,
                         max_steps=500):
    """运行一个训练回合，返回统计"""
    env.reset()
    task_alloc.reset()
    controller.reset()
    mapf.update_grid(env.grid)

    for step in range(max_steps):
        env.step()
        mapf.update_grid(env.grid)
        task_alloc.step()
        controller.step()

    stats = controller.get_statistics()
    return {
        'tasks_completed': stats['tasks_completed'],
        'steps_taken': stats['steps_taken'],
        'train_steps': controller.dqn_agent.train_steps,
        'epsilon': controller.dqn_agent.epsilon,
        'avg_loss': (np.mean(controller.dqn_agent.loss_history[-100:])
                     if controller.dqn_agent.loss_history else 0),
    }

def main():
    parser = argparse.ArgumentParser(description="RL训练 - 仓库仿真环境")
    parser.add_argument("--episodes", type=int, default=50,
                        help="训练回合数")
    parser.add_argument("--steps", type=int, default=800,
                        help="每回合最大步数")
    parser.add_argument("--save-model", type=str, default="models/rl_warehouse.pth",
                        help="模型保存路径")
    parser.add_argument("--load-model", type=str, default=None,
                        help="加载已有模型继续训练")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("AGVProject.Training")
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print(f"RL仓库训练: {args.episodes} episodes × {args.steps} steps")
    print("=" * 60)

    # 创建仿真环境
    env = WarehouseEnv()
    env.reset()
    task_alloc = TaskAllocator(env.loading_zones, env.unloading_zones)
    mapf = MAPFPlanner(env.grid, env.width, env.height)
    rl_avoid = RLCollisionAvoidance()
    controller = AGVController(env, mapf, rl_avoid)
    controller.set_task_allocator(task_alloc)
    task_alloc.set_controller(controller)

    # 加载或创建RL模型
    if args.load_model and os.path.exists(args.load_model):
        controller.dqn_agent.load_model(args.load_model)
        print(f"Loaded model: {args.load_model}")

    controller.dqn_agent.set_training(True)
    controller.dqn_agent.reset_exploration(0.5)

    results = []
    best_tasks = 0
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        result = run_training_episode(env, task_alloc, mapf, rl_avoid,
                                      controller, args.steps)
        results.append(result)

        # 保存最佳模型
        if result['tasks_completed'] > best_tasks:
            best_tasks = result['tasks_completed']
            controller.dqn_agent.save_model(args.save_model)
            print(f"  → Best model saved ({best_tasks} tasks)")

        if ep % 5 == 0 or ep == 1:
            elapsed = time.time() - start_time
            recent = results[-5:]
            avg_tasks = np.mean([r['tasks_completed'] for r in recent])
            print(f"Ep {ep:3d}/{args.episodes} | "
                  f"tasks={result['tasks_completed']:3d} | "
                  f"avg_tasks(5)={avg_tasks:.0f} | "
                  f"loss={result['avg_loss']:.4f} | "
                  f"ε={result['epsilon']:.3f} | "
                  f"{elapsed:.0f}s")

    # 保存最终模型
    controller.dqn_agent.save_model(args.save_model)
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print("训练完成")
    print(f"{'='*60}")
    print(f"总时间: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"总回合数: {args.episodes}")
    all_tasks = [r['tasks_completed'] for r in results]
    print(f"平均任务/回合: {np.mean(all_tasks):.1f}")
    print(f"最大任务/回合: {max(all_tasks)}")
    print(f"最终ε: {results[-1]['epsilon']:.4f}")
    print(f"最佳模型: {args.save_model} ({best_tasks} tasks)")

if __name__ == "__main__":
    main()
