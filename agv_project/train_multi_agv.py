"""多AGV RL训练 — 3 AGV参数共享, 在真实仓库环境中训练"""
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
from scheduler.task_allocator import TaskAllocator
from path_planning.mapf_planner import MAPFPlanner
from path_planning.rl_collision_avoidance import RLCollisionAvoidance
from path_planning.agv_controller import AGVController
from path_planning.rl.dqn_agent import DQNAgent, ACTIONS, ACTION_DELTAS
from interface.data_types import AGVStatus


def setup_logging():
    # 训练时关闭所有日志
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.basicConfig(level=logging.CRITICAL)


def train_multi_agv(episodes=100, steps_per_ep=500, load_model=None,
                    save_model="models/rl_multi_agv.pth", lr=0.0003):
    """多AGV参数共享训练"""

    logger = logging.getLogger("AGVProject.MultiAGV")
    logger.setLevel(logging.INFO)

    print("=" * 60)
    print("Multi-AGV RL Training (3 AGVs, Shared Policy)")
    print("=" * 60)

    # 创建共享DQN
    shared_agent = DQNAgent(grid_size=15, lr=lr, batch_size=128,
                            epsilon_decay=0.998, target_update=200,
                            memory_size=200000)
    shared_agent.set_training(True)

    if load_model and os.path.exists(load_model):
        shared_agent.load_model(load_model)
        print(f"Loaded pretrained model: {load_model}")
    else:
        print("Starting from scratch (random weights)")

    # 创建环境(重用,每回合reset)
    env = WarehouseEnv()
    env.reset()
    allocator = TaskAllocator(env.loading_zones, env.unloading_zones)
    mapf_planner = MAPFPlanner(env.grid, env.width, env.height)
    rl_avoidance = RLCollisionAvoidance()
    controller = AGVController(env, mapf_planner, rl_avoidance)
    controller.dqn_agent = shared_agent
    allocator.set_controller(controller)
    controller.set_task_allocator(allocator)

    # 训练
    epsilon = 0.8
    best_tasks = 0
    total_tasks = 0
    total_collisions = 0
    start_time = time.time()

    for ep in range(1, episodes + 1):
        env.reset()
        allocator.reset()
        controller.reset()
        mapf_planner.update_grid(env.grid)

        ep_tasks = 0
        ep_collisions = 0
        ep_rewards = []

        for step in range(steps_per_ep):
            env.step()
            mapf_planner.update_grid(env.grid)
            allocator.step()

            # RL移动
            all_positions = {aid: agv.position for aid, agv in controller.agvs.items()}
            occupied = set(all_positions.values())
            obs_set = set(o.position for o in env.obstacles)

            for agv_id, agv in controller.agvs.items():
                if agv.status not in [AGVStatus.MOVING_TO_PICKUP,
                                      AGVStatus.MOVING_TO_DELIVERY,
                                      AGVStatus.MOVING_TO_CHARGE]:
                    continue
                if agv.goal_pos is None:
                    continue
                if agv.position == agv.goal_pos:
                    # 到达处理
                    if agv.status == AGVStatus.MOVING_TO_PICKUP:
                        agv.is_loaded = True
                        if agv.current_task:
                            agv.goal_pos = agv.current_task.delivery_pos
                            agv.status = AGVStatus.MOVING_TO_DELIVERY
                    elif agv.status == AGVStatus.MOVING_TO_DELIVERY:
                        agv.is_loaded = False
                        if agv.current_task:
                            task_id = agv.current_task.task_id
                            agv.current_task = None
                            agv.goal_pos = None
                            agv.status = AGVStatus.IDLE
                            # 通知分配器
                            from interface.communication import MessageType, Message
                            controller.publish(MessageType.ENV_TASK_COMPLETED, {
                                "task_id": task_id, "agv_id": agv_id,
                                "position": agv.position, "step": step})
                            ep_tasks += 1
                    elif agv.status == AGVStatus.MOVING_TO_CHARGE:
                        agv.status = AGVStatus.IDLE
                        agv.goal_pos = None
                    continue

                # CBS子目标
                nav_target = agv.sub_goal if agv.sub_goal else agv.goal_pos

                # 有效动作
                valid = []
                for a_idx in range(5):
                    dx, dy = ACTION_DELTAS[ACTIONS[a_idx]]
                    nx, ny = agv.position[0] + dx, agv.position[1] + dy
                    if 0 <= nx < env.width and 0 <= ny < env.height:
                        if env.grid[ny][nx] != 1:
                            cell = env.grid[ny][nx]
                            is_zone = cell in (2, 3, 4)
                            if (dx, dy) == (0, 0) or (nx, ny) not in occupied or is_zone:
                                valid.append(a_idx)
                if not valid:
                    valid.append(4)

                # RL决策
                other_agvs = [p for aid, p in all_positions.items() if aid != agv_id]
                local, gvec = shared_agent.encoder.encode(
                    agv.position, nav_target, env.grid,
                    list(obs_set), other_agvs,
                    battery=agv.battery, is_loaded=agv.is_loaded,
                    priority=(agv.current_task.priority if agv.current_task else 1),
                    charging_stations=env.charging_stations)
                action = shared_agent.select_action(local, gvec, valid)

                # 执行动作
                dx, dy = ACTION_DELTAS[ACTIONS[action]]
                new_pos = (agv.position[0] + dx, agv.position[1] + dy)
                old_pos = agv.position

                if action == 4:
                    pass  # 等待
                elif new_pos in occupied and new_pos != old_pos:
                    ep_collisions += 1
                    continue
                else:
                    occupied.discard(old_pos)
                    occupied.add(new_pos)
                    all_positions[agv_id] = new_pos
                    agv.position = new_pos

                # 奖励 + 存储经验
                from interface.data_types import manhattan_distance
                prev_dist = manhattan_distance(old_pos, nav_target)
                curr_dist = manhattan_distance(agv.position, nav_target)
                reward = 0.5 if curr_dist < prev_dist else (-0.5 if curr_dist > prev_dist else 0.0)
                reward -= 0.05
                arrived = (agv.position == agv.goal_pos)
                ep_rewards.append(reward)

                next_other = [p for aid, p in all_positions.items() if aid != agv_id]
                next_local, next_gvec = shared_agent.encoder.encode(
                    agv.position, nav_target, env.grid,
                    list(obs_set), next_other,
                    battery=agv.battery, is_loaded=agv.is_loaded,
                    priority=(agv.current_task.priority if agv.current_task else 1),
                    charging_stations=env.charging_stations)
                shared_agent.store_experience(
                    local, gvec, action, reward,
                    next_local, next_gvec, arrived)

            # 训练
            if len(shared_agent.memory) >= shared_agent.batch_size:
                shared_agent.train_step()

            # CBS子目标规划
            if step % 30 == 0:
                controller._plan_subgoals_with_cbs()

        # 回合统计
        total_tasks += ep_tasks
        total_collisions += ep_collisions
        epsilon = max(0.02, epsilon * 0.995)

        if ep_tasks > best_tasks:
            best_tasks = ep_tasks
            shared_agent.save_model(save_model)

        if ep % 10 == 0:
            elapsed = time.time() - start_time
            avg_tasks = total_tasks / ep
            avg_reward = np.mean(ep_rewards) if ep_rewards else 0
            print(f"Ep {ep:3d}/{episodes} | "
                  f"tasks={ep_tasks:2d} best={best_tasks} "
                  f"collisions={ep_collisions} "
                  f"avg_r={avg_reward:.3f} "
                  f"eps={epsilon:.3f} "
                  f"avg_t/ep={avg_tasks:.1f} "
                  f"{elapsed:.0f}s")

    shared_agent.save_model(save_model)
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Total tasks: {total_tasks}, Best: {best_tasks}, Collisions: {total_collisions}")
    print(f"Model saved: {save_model}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-AGV RL Training")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--save-model", type=str, default="models/rl_multi_agv.pth")
    parser.add_argument("--lr", type=float, default=0.0003)
    args = parser.parse_args()

    setup_logging()
    train_multi_agv(
        episodes=args.episodes,
        steps_per_ep=args.steps,
        load_model=args.load_model,
        save_model=args.save_model,
        lr=args.lr)
