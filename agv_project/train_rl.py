"""独立RL训练脚本 — 课程学习 + 收敛验证"""
import sys
import os
import argparse
import logging
import time
import numpy as np

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from path_planning.rl.dqn_agent import DQNAgent, ACTIONS, ACTION_DELTAS
from path_planning.rl.curriculum_trainer import CurriculumTrainer, CurriculumStage


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S')


STAGES = [
    CurriculumStage(
        stage_id=1, map_size=10,
        num_static_obstacles=(3, 6), num_moving_obstacles=(0, 0),
        num_agvs=1, max_steps=150, num_scenarios=150,
        success_threshold=0.75),
    CurriculumStage(
        stage_id=2, map_size=30,
        num_static_obstacles=(5, 8), num_moving_obstacles=(1, 3),
        num_agvs=1, max_steps=400, num_scenarios=200,
        success_threshold=0.70),
    CurriculumStage(
        stage_id=3, map_size=50,
        num_static_obstacles=(8, 12), num_moving_obstacles=(2, 5),
        num_agvs=1, max_steps=600, num_scenarios=200,
        success_threshold=0.60),
]


def run_curriculum_training(agent, stages, model_dir="models"):
    """运行课程学习训练并返回结果"""
    os.makedirs(model_dir, exist_ok=True)
    trainer = CurriculumTrainer(agent, stages=stages)
    all_results = []

    for i, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"Stage {stage.stage_id}: {stage.map_size}x{stage.map_size} "
              f"({stage.num_scenarios} scenarios)")
        print(f"{'='*60}")

        agent.reset_exploration(0.8)
        agent.set_training(True)

        success_rate = trainer.train_stage(stage)
        all_results.append({
            'stage': stage.stage_id,
            'map_size': stage.map_size,
            'success_rate': success_rate,
            'train_steps': agent.train_steps,
            'epsilon': agent.epsilon,
        })

        print(f"Stage {stage.stage_id} complete: "
              f"success_rate={success_rate:.2%}, "
              f"train_steps={agent.train_steps}, "
              f"epsilon={agent.epsilon:.4f}")

        # 保存每阶段模型
        stage_path = os.path.join(model_dir, f"rl_stage{stage.stage_id}.pth")
        agent.save_model(stage_path)
        print(f"Model saved: {stage_path}")

    # 保存最终模型
    final_path = os.path.join(model_dir, "rl_final.pth")
    agent.save_model(final_path)

    return all_results


def evaluate_agent(agent, map_size=50, num_trials=50, max_steps=500):
    """评估训练好的agent"""
    import random
    agent.set_training(False)
    successes = 0
    path_lengths = []

    for trial in range(num_trials):
        empty = [(x, y) for x in range(1, map_size - 1)
                 for y in range(1, map_size - 1)]
        start = random.choice(empty)
        goal = random.choice([c for c in empty if c != start])

        num_obs = random.randint(5, 10)
        available = [c for c in empty if c not in (start, goal)]
        obstacles = random.sample(available, min(num_obs, len(available)))
        grid = [[0] * map_size for _ in range(map_size)]
        for ox, oy in obstacles:
            grid[oy][ox] = 1

        agent.encoder.map_width = map_size
        agent.encoder.map_height = map_size
        pos = start
        path_len = 0

        for step in range(max_steps):
            local, gvec = agent.encoder.encode(
                pos, goal, grid, obstacles, [],
                is_loaded=False, priority=1)

            valid = []
            for a in range(5):
                dx, dy = ACTION_DELTAS[ACTIONS[a]]
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < map_size and 0 <= ny < map_size:
                    if grid[ny][nx] != 1:
                        valid.append(a)
            if not valid:
                valid = [4]

            action = agent.select_action(local, gvec, valid)
            dx, dy = ACTION_DELTAS[ACTIONS[action]]
            pos = (pos[0] + dx, pos[1] + dy)
            path_len += 1

            if pos == goal:
                successes += 1
                path_lengths.append(path_len)
                break

    success_rate = successes / num_trials
    avg_path = np.mean(path_lengths) if path_lengths else -1
    opt_path = map_size  # approximate optimal
    return success_rate, avg_path, opt_path


def main():
    parser = argparse.ArgumentParser(description="AGV RL Training")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3],
                        help="Stages to train (default: 1 2 3)")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Model save directory")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Load pretrained model to continue training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate, skip training")
    parser.add_argument("--ppo", action="store_true",
                        help="Use PPO instead of DQN")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("AGVProject.Training")

    print("=" * 60)
    print("AGV RL 训练系统")
    print("=" * 60)

    # 选择agent
    if args.ppo:
        from path_planning.rl.ppo_agent import PPOAgent
        agent = PPOAgent(grid_size=15, batch_size=128)
        print("Using PPO Agent")
    else:
        agent = DQNAgent(grid_size=15, batch_size=128, lr=0.0005,
                         epsilon_decay=0.998, target_update=200,
                         memory_size=200000)
        print("Using Double DQN Agent")

    if args.load_model:
        agent.load_model(args.load_model)
        print(f"Loaded model: {args.load_model}")

    if not args.eval_only:
        # 选择阶段
        selected_stages = [STAGES[i - 1] for i in args.stages if 1 <= i <= 3]
        if not selected_stages:
            selected_stages = STAGES

        start_time = time.time()
        results = run_curriculum_training(agent, selected_stages, args.model_dir)
        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print("Training Summary")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        for r in results:
            print(f"  Stage {r['stage']} ({r['map_size']}x{r['map_size']}): "
                  f"success_rate={r['success_rate']:.2%}")

    # 最终评估
    print(f"\n{'='*60}")
    print("Final Evaluation (50x50)")
    print(f"{'='*60}")
    success_rate, avg_path, opt = evaluate_agent(
        agent, map_size=50, num_trials=50)
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Avg Path Length: {avg_path:.1f} (optimal ~{opt})")
    print(f"Ratio to Optimal: {avg_path/opt:.2f}x" if avg_path > 0 else "N/A")

    if success_rate >= 0.6:
        print("\n*** 收敛验证通过: RL模型在50x50地图上达到60%+成功率 ***")
    else:
        print(f"\n*** 收敛不足: 当前成功率{success_rate:.1%}, 目标60% ***")
        print("建议: 增加训练场景数或调整奖励函数")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
