#!/usr/bin/env python
"""Training: 300 episodes, 3 AGV shared policy, with metrics charts"""
import sys, os, argparse, time, numpy as np, logging

# Suppress ALL logging BEFORE any imports
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.basicConfig(level=logging.CRITICAL, force=True)
logging.disable(logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).disabled = True

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env.warehouse_env import WarehouseEnv
from scheduler.task_allocator import TaskAllocator
from path_planning.mapf_planner import MAPFPlanner
from path_planning.rl_collision_avoidance import RLCollisionAvoidance
from path_planning.agv_controller import AGVController, AGVStatus
from path_planning.rl.dqn_agent import DQNAgent, ACTIONS, ACTION_DELTAS
from interface.data_types import manhattan_distance
from interface.communication import MessageType

# Suppress AFTER imports too
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger(name).disabled = True


def plot_charts(metrics, save_dir="logs/metrics"):
    os.makedirs(save_dir, exist_ok=True)
    eps = range(1, len(metrics['reward']) + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps, metrics['reward'], color='forestgreen', linewidth=0.8)
    ax.set_title('Average Reward per Episode'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{save_dir}/training_reward.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps, metrics['loss'], color='firebrick', linewidth=0.8)
    ax.set_title('Average Loss per Episode'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{save_dir}/training_loss.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps, metrics['energy'], color='darkorange', linewidth=0.8)
    ax.set_title('Energy Consumption per Episode'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{save_dir}/training_energy.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps, metrics['tasks'], color='steelblue', linewidth=0.8)
    ax.set_title('Tasks Completed per Episode'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{save_dir}/training_steps.png', dpi=150); plt.close(fig)

    # Combined dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (title, key, color) in zip(
        axes.flat,
        [('Tasks', 'tasks', 'steelblue'), ('Reward', 'reward', 'forestgreen'),
         ('Loss', 'loss', 'firebrick'), ('Energy', 'energy', 'darkorange')]):
        ax.plot(eps, metrics[key], color=color, linewidth=0.5)
        ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.suptitle('AGV RL Training Dashboard', fontsize=16)
    fig.tight_layout(); fig.savefig(f'{save_dir}/training_dashboard.png', dpi=150); plt.close(fig)
    print(f'Charts saved to {save_dir}/', flush=True)


def train(episodes=300, steps_per_ep=300, model_path="models/rl_final_v2.pth"):
    print(f"Training: {episodes} eps x {steps_per_ep} steps, 3 AGVs", flush=True)

    agent = DQNAgent(grid_size=15, lr=0.0003, batch_size=128,
                     epsilon_decay=0.998, target_update=200, memory_size=200000)
    agent.set_training(True)
    agent.reset_exploration(1.0)

    env = WarehouseEnv(); env.reset()
    alloc = TaskAllocator(env.loading_zones, env.unloading_zones)
    mp = MAPFPlanner(env.grid, env.width, env.height)
    ctl = AGVController(env, mp, RLCollisionAvoidance())
    ctl.dqn_agent = agent
    alloc.set_controller(ctl); ctl.set_task_allocator(alloc)

    metrics = {'reward': [], 'loss': [], 'energy': [], 'tasks': []}
    best_tasks = 0; epsilon = 1.0; start_t = time.time()

    for ep in range(1, episodes + 1):
        env.reset(); alloc.reset(); ctl.reset(); mp.update_grid(env.grid)
        ep_tasks = 0; ep_rewards = []; ep_losses = []; ep_energy = 0.0

        for step in range(steps_per_ep):
            env.step(); mp.update_grid(env.grid); alloc.step()
            all_pos = {aid: a.position for aid, a in ctl.agvs.items()}
            occupied = set(all_pos.values())
            obs_set = set(o.position for o in env.obstacles)

            for agv_id, agv in ctl.agvs.items():
                if agv.status not in [AGVStatus.MOVING_TO_PICKUP,
                                      AGVStatus.MOVING_TO_DELIVERY,
                                      AGVStatus.MOVING_TO_CHARGE]:
                    continue
                if agv.goal_pos is None: continue
                if agv.position == agv.goal_pos:
                    if agv.status == AGVStatus.MOVING_TO_PICKUP:
                        agv.is_loaded = True
                        if agv.current_task:
                            agv.goal_pos = agv.current_task.delivery_pos
                            agv.status = AGVStatus.MOVING_TO_DELIVERY
                    elif agv.status == AGVStatus.MOVING_TO_DELIVERY:
                        agv.is_loaded = False
                        if agv.current_task:
                            ctl.publish(MessageType.ENV_TASK_COMPLETED,
                                       {"task_id": agv.current_task.task_id,
                                        "agv_id": agv_id, "position": agv.position,
                                        "step": step})
                            agv.current_task = None; agv.goal_pos = None
                            agv.status = AGVStatus.IDLE; ep_tasks += 1
                    elif agv.status == AGVStatus.MOVING_TO_CHARGE:
                        agv.status = AGVStatus.IDLE; agv.goal_pos = None
                    continue

                nav_target = agv.sub_goal if agv.sub_goal else agv.goal_pos
                valid = []
                for a_idx in range(5):
                    dx, dy = ACTION_DELTAS[ACTIONS[a_idx]]
                    nx, ny = agv.position[0]+dx, agv.position[1]+dy
                    if 0 <= nx < env.width and 0 <= ny < env.height:
                        if env.grid[ny][nx] != 1:
                            cell = env.grid[ny][nx]
                            is_zone = cell in (2, 3, 4)
                            if (dx, dy) == (0, 0) or (nx, ny) not in occupied or is_zone:
                                valid.append(a_idx)
                if not valid: valid.append(4)

                other_agvs = [p for aid, p in all_pos.items() if aid != agv_id]
                local, gvec = agent.encoder.encode(
                    agv.position, nav_target, env.grid, list(obs_set), other_agvs,
                    battery=agv.battery, is_loaded=agv.is_loaded,
                    priority=(agv.current_task.priority if agv.current_task else 1),
                    charging_stations=env.charging_stations)
                action = agent.select_action(local, gvec, valid)
                dx, dy = ACTION_DELTAS[ACTIONS[action]]
                new_pos = (agv.position[0]+dx, agv.position[1]+dy)
                old_pos = agv.position

                if action != 4 and new_pos not in occupied:
                    occupied.discard(old_pos); occupied.add(new_pos)
                    all_pos[agv_id] = new_pos; agv.position = new_pos

                prev_d = manhattan_distance(old_pos, nav_target)
                curr_d = manhattan_distance(agv.position, nav_target)
                r = 0.5 if curr_d < prev_d else (-0.5 if curr_d > prev_d else 0.0)
                r -= 0.05; ep_rewards.append(r)

                next_other = [p for aid, p in all_pos.items() if aid != agv_id]
                next_local, next_gvec = agent.encoder.encode(
                    agv.position, nav_target, env.grid, list(obs_set), next_other,
                    battery=agv.battery, is_loaded=agv.is_loaded,
                    priority=(agv.current_task.priority if agv.current_task else 1),
                    charging_stations=env.charging_stations)
                agent.store_experience(local, gvec, action, r, next_local, next_gvec,
                                      agv.position == agv.goal_pos)

            if len(agent.memory) >= agent.batch_size:
                loss = agent.train_step()
                if loss is not None: ep_losses.append(loss)

            if step % 30 == 0: ctl._plan_subgoals_with_cbs()

        for agv in ctl.agvs.values(): ep_energy += (200.0 - agv.battery)

        metrics['reward'].append(np.mean(ep_rewards) if ep_rewards else 0)
        metrics['loss'].append(np.mean(ep_losses) if ep_losses else 0)
        metrics['energy'].append(ep_energy)
        metrics['tasks'].append(ep_tasks)

        epsilon = max(0.02, epsilon * 0.995); agent.epsilon = epsilon
        if ep_tasks > best_tasks: best_tasks = ep_tasks; agent.save_model(model_path)

        if ep % 10 == 0:
            t = time.time() - start_t
            print(f"Ep {ep:3d}/{episodes} | tasks={ep_tasks:2d} best={best_tasks} "
                  f"r={metrics['reward'][-1]:+.3f} loss={metrics['loss'][-1]:.4f} "
                  f"energy={ep_energy:.0f} eps={epsilon:.3f} t={t:.0f}s", flush=True)

    agent.save_model(model_path)
    t = time.time() - start_t
    print(f"\nDone: {t:.0f}s ({t/60:.1f}min) | Best: {best_tasks} | Model: {model_path}", flush=True)
    plot_charts(metrics)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--save", type=str, default="models/rl_final_v2.pth")
    args = p.parse_args()
    train(args.episodes, args.steps, args.save)
