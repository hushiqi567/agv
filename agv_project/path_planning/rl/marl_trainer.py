"""多智能体RL训练器 — 独立训练 + 参数共享"""
import random
import logging
from typing import Dict

from path_planning.rl.dqn_agent import DQNAgent, ACTIONS, ACTION_DELTAS

class MARLTrainer:
    """
    多智能体训练器。

    策略: 所有AGV共享同一个DQN模型（参数共享）。
    训练时在多AGV场景中随机初始位置，让模型自然学会避让。
    """

    def __init__(self, agent: DQNAgent, num_agvs: int = 4):
        self.agent = agent
        self.num_agvs = num_agvs
        self.logger = logging.getLogger("AGVProject.MARL")

    def train_episode(self, env, task_allocator, max_steps=500) -> Dict:
        env.reset()
        task_allocator.reset()

        agv_positions = self._random_init_positions(env)
        agv_goals = {}
        agv_loaded = {i: False for i in range(self.num_agvs)}

        total_reward = 0
        tasks_completed = 0
        collisions = 0

        for step in range(max_steps):
            env.step()
            task_allocator.step()

            occupied = set(agv_positions.values())
            obstacle_positions = set(o.position for o in env.obstacles)

            for agv_id in range(self.num_agvs):
                pos = agv_positions[agv_id]
                goal = agv_goals.get(agv_id)

                if goal is None:
                    continue

                other_agvs = [p for aid, p in agv_positions.items() if aid != agv_id]
                local, gvec = self.agent.encoder.encode(
                    pos, goal, env.grid,
                    list(obstacle_positions), other_agvs)

                valid = self._get_valid_actions(pos, env, occupied)
                action = self.agent.select_action(local, gvec, valid)
                dx, dy = ACTION_DELTAS[ACTIONS[action]]
                new_pos = (pos[0] + dx, pos[1] + dy)

                if new_pos in obstacle_positions or new_pos in occupied:
                    collisions += 1
                    continue

                agv_positions[agv_id] = new_pos
                occupied.add(new_pos)

                arrived = (new_pos == goal)
                next_other = [p for aid, p in agv_positions.items() if aid != agv_id]
                next_local, next_gvec = self.agent.encoder.encode(
                    new_pos, goal, env.grid,
                    list(obstacle_positions), next_other)

                reward = DQNAgent.compute_reward(
                    pos, new_pos, goal, agv_loaded[agv_id],
                    arrived_pickup=arrived)
                total_reward += reward

                self.agent.store_experience(
                    local, gvec, action, reward, next_local, next_gvec, arrived)

                if arrived:
                    tasks_completed += 1

            self.agent.train_step()

        return {
            'tasks_completed': tasks_completed,
            'collisions': collisions,
            'avg_reward': total_reward / max_steps,
        }

    def _random_init_positions(self, env):
        positions = {}
        for i in range(self.num_agvs):
            while True:
                x = random.randint(1, env.width - 2)
                y = random.randint(1, env.height - 2)
                if env.grid[y][x] == 0 and (x, y) not in positions.values():
                    positions[i] = (x, y)
                    break
        return positions

    def _get_valid_actions(self, pos, env, occupied):
        valid = []
        for a_idx in range(5):
            dx, dy = ACTION_DELTAS[ACTIONS[a_idx]]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < env.width and 0 <= ny < env.height:
                if env.grid[ny][nx] != 1:
                    if (nx, ny) not in occupied or (dx, dy) == (0, 0):
                        valid.append(a_idx)
        return valid or [4]
