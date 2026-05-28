"""三阶段课程学习训练器"""
import random
import logging
from typing import List, Tuple, Dict
from dataclasses import dataclass

from path_planning.rl.ppo_agent import PPOAgent
from path_planning.rl.dqn_agent import DQNAgent, ACTIONS, ACTION_DELTAS


@dataclass
class CurriculumStage:
    """课程阶段配置"""
    stage_id: int
    map_size: int
    num_static_obstacles: Tuple[int, int]
    num_moving_obstacles: Tuple[int, int]
    num_agvs: int
    max_steps: int
    num_scenarios: int = 100
    success_threshold: float = 0.8


STAGES = [
    CurriculumStage(1, 10, (3, 5), (0, 0), 1, 200, 100, 0.8),
    CurriculumStage(2, 30, (5, 8), (2, 3), 2, 500, 100, 0.8),
    CurriculumStage(3, 50, (10, 10), (10, 10), 4, 1000, 100, 0.6),
]


class CurriculumTrainer:
    """课程学习训练器 — 管理三阶段渐进训练"""

    def __init__(self, agent, stages=None):
        self.agent = agent
        self.stages = stages or STAGES
        self.current_stage_idx = 0
        self.logger = logging.getLogger("AGVProject.Curriculum")
        self.stats: Dict[int, list] = {s.stage_id: [] for s in self.stages}

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    def generate_scenario(self, stage: CurriculumStage):
        """生成一个随机场景：小地图随机起点/终点/障碍物"""
        size = stage.map_size
        empty_cells = [(x, y) for x in range(1, size-1) for y in range(1, size-1)]
        start = random.choice(empty_cells)
        goal = random.choice([c for c in empty_cells if c != start])

        num_obs = random.randint(*stage.num_static_obstacles)
        available = [c for c in empty_cells if c not in (start, goal)]
        obstacles = random.sample(available, min(num_obs, len(available)))

        grid = [[0]*size for _ in range(size)]
        for ox, oy in obstacles:
            grid[oy][ox] = 1

        return {
            'grid': grid,
            'start': start,
            'goal': goal,
            'obstacles': obstacles,
            'size': size,
        }

    def train_stage(self, stage: CurriculumStage) -> float:
        """训练一个完整阶段，返回成功率"""
        self.agent.set_training(True)
        successes = 0

        for ep in range(stage.num_scenarios):
            scenario = self.generate_scenario(stage)
            pos = scenario['start']
            goal = scenario['goal']
            grid = scenario['grid']
            size = scenario['size']

            self.agent.encoder.map_width = size
            self.agent.encoder.map_height = size

            for step in range(stage.max_steps):
                local, gvec = self.agent.encoder.encode(
                    pos, goal, grid, [], [],
                    is_loaded=False, priority=1)

                valid = self._get_valid_actions(pos, grid, size, set())
                if not valid:
                    break
                action = self.agent.select_action(local, gvec, valid)
                dx, dy = ACTION_DELTAS[ACTIONS[action]]
                new_pos = (pos[0] + dx, pos[1] + dy)

                arrived = (new_pos == goal)
                in_bounds = 0 <= new_pos[0] < size and 0 <= new_pos[1] < size
                hit_obstacle = in_bounds and grid[new_pos[1]][new_pos[0]] == 1 if in_bounds else True
                next_local, next_gvec = self.agent.encoder.encode(
                    new_pos, goal, grid, [], [], is_loaded=False, priority=1)
                reward = DQNAgent.compute_reward(
                    pos, new_pos, goal, False,
                    arrived_pickup=arrived, arrived_delivery=arrived,
                    obstacle_collision=hit_obstacle)
                self.agent.store_experience(local, gvec, action, reward, next_local, next_gvec, arrived)
                self.agent.train_step()
                pos = new_pos
                if arrived:
                    successes += 1
                    break

            if (ep + 1) % 20 == 0:
                self.logger.info(
                    f"Stage {stage.stage_id} episode {ep+1}/{stage.num_scenarios}, "
                    f"success rate: {successes/(ep+1):.2%}")

        success_rate = successes / stage.num_scenarios
        self.stats[stage.stage_id].append({'success_rate': success_rate})
        return success_rate

    def run(self) -> bool:
        """运行完整课程学习，返回是否完成所有阶段"""
        for i, stage in enumerate(self.stages):
            self.current_stage_idx = i
            self.logger.info(
                f"=== Starting Curriculum Stage {stage.stage_id}: "
                f"{stage.map_size}x{stage.map_size} ===")
            rate = self.train_stage(stage)
            self.logger.info(f"Stage {stage.stage_id} complete. Success rate: {rate:.2%}")
            if rate < stage.success_threshold and i < len(self.stages) - 1:
                self.logger.warning(
                    f"Stage {stage.stage_id} below threshold "
                    f"({stage.success_threshold:.0%}), retrying...")
                rate = self.train_stage(stage)
                if rate < stage.success_threshold:
                    self.logger.warning(
                        f"Stage {stage.stage_id} still below threshold, advancing anyway")
        return True

    @staticmethod
    def _get_valid_actions(pos, grid, size, occupied):
        valid = []
        for a_idx, a_name in enumerate(ACTIONS):
            dx, dy = ACTION_DELTAS[a_name]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < size and 0 <= ny < size:
                if grid[ny][nx] != 1 and (nx, ny) not in occupied:
                    valid.append(a_idx)
        if not valid:
            valid.append(4)
        return valid
