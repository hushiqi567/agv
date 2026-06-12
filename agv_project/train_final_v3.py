"""300-episode training from scratch with 4 metrics charts"""
import sys, os, time, numpy as np, logging
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.basicConfig(level=logging.CRITICAL, force=True)
logging.disable(logging.CRITICAL)
for n in list(logging.root.manager.loggerDict):
    logging.getLogger(n).setLevel(logging.CRITICAL)

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from env.warehouse_env import WarehouseEnv
from scheduler.task_allocator import TaskAllocator
from path_planning.mapf_planner import MAPFPlanner
from path_planning.rl_collision_avoidance import RLCollisionAvoidance
from path_planning.agv_controller import AGVController, AGVStatus
from path_planning.rl.dqn_agent import DQNAgent, ACTIONS, ACTION_DELTAS
from interface.data_types import manhattan_distance
from interface.communication import MessageType
from interface.config import get_config

BATTERY_PER_STEP = get_config().agv.battery_consumption_per_step

os.makedirs("logs/metrics", exist_ok=True)

def plot_curves(metrics):
    eps = range(1, len(metrics['loss']) + 1)

    # 1. Loss
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps, metrics['loss'], color='firebrick', linewidth=0.6)
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg Huber Loss')
    ax.set_title('Training Loss'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig('logs/metrics/training_loss.png', dpi=150); plt.close(fig)

    # 2. Reward
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(eps, metrics['reward'], color='forestgreen', linewidth=0.6)
    ax.set_xlabel('Episode'); ax.set_ylabel('Avg Reward per Step')
    ax.set_title('Average Reward'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig('logs/metrics/training_reward.png', dpi=150); plt.close(fig)

    # 3. Energy per Task
    fig, ax = plt.subplots(figsize=(12, 5))
    ept = [metrics['energy'][i] / max(1, metrics['tasks'][i]) for i in range(len(eps))]
    ax.plot(eps, ept, color='darkorange', linewidth=0.6)
    ax.set_xlabel('Episode'); ax.set_ylabel('Energy per Task')
    ax.set_title('Energy per Task'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig('logs/metrics/training_energy.png', dpi=150); plt.close(fig)

    # 4. Steps per Task
    fig, ax = plt.subplots(figsize=(12, 5))
    spt = [300 / max(1, metrics['tasks'][i]) for i in range(len(eps))]
    ax.plot(eps, spt, color='mediumpurple', linewidth=0.6)
    ax.set_xlabel('Episode'); ax.set_ylabel('Steps per Task')
    ax.set_title('Steps per Task (lower = more efficient)'); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig('logs/metrics/training_steps_per_task.png', dpi=150); plt.close(fig)

    # Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    cfg = [('Loss', metrics['loss'], 'firebrick'),
           ('Reward', metrics['reward'], 'forestgreen'),
           ('Energy per Task', ept, 'darkorange'),
           ('Steps per Task', spt, 'mediumpurple')]
    for i, (title, data, color) in enumerate(cfg):
        ax = axes[i//2, i%2]
        ax.plot(eps, data, color=color, linewidth=0.4)
        ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.suptitle('AGV RL Training Dashboard (300 Episodes, 3 AGVs)', fontsize=16)
    fig.tight_layout(); fig.savefig('logs/metrics/training_dashboard.png', dpi=150); plt.close(fig)
    print("Charts saved to logs/metrics/", flush=True)


def train(episodes=300, steps_per_ep=300, model_path="models/rl_final_v3.pth"):
    print(f"Training: {episodes} eps x {steps_per_ep} steps, 3 AGVs, from scratch", flush=True)

    agent = DQNAgent(grid_size=15, lr=0.0003, batch_size=128,
                     epsilon_decay=0.998, target_update=200, memory_size=200000)
    agent.set_training(True); agent.reset_exploration(1.0)

    env = WarehouseEnv(); env.reset()
    alloc = TaskAllocator(env.loading_zones, env.unloading_zones)
    mp = MAPFPlanner(env.grid, env.width, env.height)
    ctl = AGVController(env, mp, RLCollisionAvoidance())
    ctl.dqn_agent = agent
    alloc.set_controller(ctl); ctl.set_task_allocator(alloc)

    metrics = {'loss': [], 'reward': [], 'energy': [], 'tasks': [], 'epsilon': []}
    best_tasks = 0; epsilon = 1.0; start_t = time.time()

    for ep in range(1, episodes + 1):
        env.reset(); alloc.reset(); ctl.reset(); mp.update_grid(env.grid)
        ep_tasks = 0; ep_rewards = []; ep_losses = []; ep_battery_used = 0.0

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
                            ctl.publish(MessageType.ENV_TASK_COMPLETED, {
                                "task_id": agv.current_task.task_id, "agv_id": agv_id,
                                "position": agv.position, "step": step})
                            agv.current_task = None; agv.goal_pos = None
                            agv.status = AGVStatus.IDLE; ep_tasks += 1
                    elif agv.status == AGVStatus.MOVING_TO_CHARGE:
                        agv.status = AGVStatus.IDLE; agv.goal_pos = None
                    continue

                nav_target = agv.sub_goal if agv.sub_goal else agv.goal_pos

                # Valid actions
                valid = []
                for a_idx in range(5):
                    dx, dy = ACTION_DELTAS[ACTIONS[a_idx]]
                    nx, ny = agv.position[0]+dx, agv.position[1]+dy
                    if 0 <= nx < env.width and 0 <= ny < env.height:
                        if env.grid[ny][nx] != 1:
                            cell = env.grid[ny][nx]
                            if cell in (2,3,4) or (dx,dy)==(0,0) or (nx,ny) not in occupied:
                                valid.append(a_idx)
                if not valid: valid.append(4)

                other_agvs = [p for aid, p in all_pos.items() if aid != agv_id]
                local, gvec = agent.encoder.encode(
                    agv.position, nav_target, env.grid, list(obs_set), other_agvs,
                    battery=agv.battery, is_loaded=agv.is_loaded,
                    priority=(agv.current_task.priority if agv.current_task else 1),
                    charging_stations=env.charging_stations)
                action = agent.select_action(local, gvec, valid)

                # Determine battery multiplier BEFORE executing
                old_battery = agv.battery
                dx, dy = ACTION_DELTAS[ACTIONS[action]]
                new_pos = (agv.position[0]+dx, agv.position[1]+dy)
                old_pos = agv.position

                if action == 4:
                    multiplier = 0.3
                elif new_pos in occupied and new_pos != old_pos:
                    multiplier = 0  # blocked, no battery used
                else:
                    occupied.discard(old_pos); occupied.add(new_pos)
                    all_pos[agv_id] = new_pos; agv.position = new_pos
                    # Determine multiplier based on actual movement
                    is_detour = (manhattan_distance(new_pos, nav_target) > manhattan_distance(old_pos, nav_target))
                    if is_detour:
                        multiplier = 2.0
                    elif agv.is_loaded:
                        multiplier = 1.5
                    else:
                        multiplier = 1.0

                ep_battery_used += BATTERY_PER_STEP * multiplier

                # Reward
                prev_d = manhattan_distance(old_pos, nav_target)
                curr_d = manhattan_distance(agv.position, nav_target)
                r = 0.5 if curr_d < prev_d else (-0.5 if curr_d > prev_d else 0.0)
                r -= 0.05; ep_rewards.append(r)

                # Store experience
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

        metrics['loss'].append(np.mean(ep_losses) if ep_losses else 0)
        metrics['reward'].append(np.mean(ep_rewards) if ep_rewards else 0)
        metrics['energy'].append(ep_battery_used)
        metrics['tasks'].append(ep_tasks)
        epsilon = max(0.02, epsilon * 0.995); agent.epsilon = epsilon
        metrics['epsilon'].append(epsilon)

        if ep_tasks > best_tasks: best_tasks = ep_tasks; agent.save_model(model_path)

        if ep % 10 == 0:
            t = time.time() - start_t
            ept = ep_battery_used / max(1, ep_tasks)
            spt = 300 / max(1, ep_tasks)
            print(f"Ep {ep:3d}/{episodes} | tasks={ep_tasks:2d} best={best_tasks} "
                  f"r={metrics['reward'][-1]:+.3f} loss={metrics['loss'][-1]:.4f} "
                  f"E/task={ept:.0f} S/task={spt:.0f} eps={epsilon:.3f} t={t:.0f}s", flush=True)

    agent.save_model(model_path)
    t = time.time() - start_t
    print(f"\nDone: {t:.0f}s ({t/60:.1f}min) | Best: {best_tasks} tasks | Model: {model_path}", flush=True)
    plot_curves(metrics)
    return metrics


if __name__ == "__main__":
    train(300, 300, "models/rl_final_v3.pth")
