"""
AGV控制器 — 纯A*路径跟踪 + RL局部避撞
"""
import sys
import os
import random
import logging
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from interface.communication import BaseModule, MessageType, Message
from interface.data_types import Task, TaskStatus, AGVStatus
from interface.data_types import manhattan_distance
from interface.config import ConfigManager
from path_planning.mapf_planner import MAPFPlanner, a_star_search
from path_planning.rl_collision_avoidance import RLCollisionAvoidance, ACTIONS, ACTION_DELTAS
from path_planning.rl.dqn_agent import DQNAgent
from path_planning.deadlock_detector import DeadlockDetector
from path_planning.metrics_collector import MetricsCollector


@dataclass
class AGVRuntimeState:
    agv_id: int
    position: Tuple[int, int]
    status: AGVStatus = AGVStatus.IDLE
    battery: float = 100.0
    is_loaded: bool = False
    current_task: Optional[Task] = None
    path: List[Tuple[int, int]] = field(default_factory=list)
    path_index: int = 0
    goal_pos: Optional[Tuple[int, int]] = None
    waiting_steps: int = 0


class AGVController(BaseModule):
    """AGV控制器 — A*路径跟踪 + RL局部避撞"""

    def __init__(self, env, mapf_planner: MAPFPlanner,
                 rl_avoidance: RLCollisionAvoidance):
        super().__init__("controller")
        self.env = env
        self.mapf_planner = mapf_planner
        self.rl_avoidance = rl_avoidance
        self.agvs: Dict[int, AGVRuntimeState] = {}
        self.task_allocator = None
        self.current_step = 0
        self.is_running = False
        self.total_tasks_completed = 0
        self.total_steps_taken = 0

        self.dqn_agent = DQNAgent(grid_size=15)
        self.deadlock_detector = DeadlockDetector()
        self.metrics = MetricsCollector()

        self.use_rl_primary = True
        self.use_random_policy = False

        self.rl_train_interval = 4

        # 电池系统配置
        cfg = ConfigManager()
        self.battery_capacity = cfg.agv.battery_capacity
        self.battery_consumption = cfg.agv.battery_consumption_per_step
        self.charge_rate = cfg.agv.charge_rate
        self.charging_stations = getattr(env, 'charging_stations', [])

        self.logger.info("AGV控制器初始化完成")

    def set_task_allocator(self, task_allocator):
        self.task_allocator = task_allocator

    def _setup_subscriptions(self):
        self.subscribe(MessageType.SCHEDULER_TASK_ASSIGNED, self._on_task_assigned)
        self.subscribe(MessageType.MAIN_SIMULATION_START, self._on_simulation_start)
        self.subscribe(MessageType.MAIN_SIMULATION_STOP, self._on_simulation_stop)
        self.subscribe(MessageType.MAIN_SIMULATION_RESET, self._on_simulation_reset)

    def _on_task_assigned(self, message: Message):
        data = message.data
        agv_id = data.get("agv_id")
        pickup_pos = data.get("pickup_pos")
        delivery_pos = data.get("delivery_pos")
        task_id = data.get("task_id")

        if agv_id not in self.agvs:
            return
        agv = self.agvs[agv_id]
        task = Task(task_id=task_id, pickup_pos=pickup_pos,
                    delivery_pos=delivery_pos, priority=1,
                    status=TaskStatus.ASSIGNED, assigned_agv_id=agv_id)
        agv.current_task = task
        agv.goal_pos = pickup_pos
        agv.status = AGVStatus.MOVING_TO_PICKUP
        agv.path = []
        agv.path_index = 0
        agv.waiting_steps = 0
        self.logger.info(f"AGV {agv_id}: 任务{task_id} 取货 {pickup_pos} → 送货 {delivery_pos}")

    def _on_simulation_start(self, message):
        self.is_running = True
        self.reset()

    def _on_simulation_stop(self, message):
        self.is_running = False
        self.logger.info(f"控制器停止: {self.total_tasks_completed} 个任务完成")

    def _on_simulation_reset(self, message):
        self.reset()

    def reset(self):
        self.current_step = 0
        self.agvs.clear()
        self.total_tasks_completed = 0
        self.total_steps_taken = 0
        self.deadlock_detector.reset()
        from scheduler.task_allocator import AGV_INITIAL_POSITIONS, NUM_AGVS
        for i in range(NUM_AGVS):
            self.agvs[i] = AGVRuntimeState(
                agv_id=i, position=AGV_INITIAL_POSITIONS[i],
                status=AGVStatus.IDLE)

    def step(self):
        self.current_step += 1
        self._manage_battery()
        self._plan_paths()
        self._move_agvs()
        self._check_deadlock_and_recover()
        self._check_task_completion()
        self._publish_state()

    # ============================================================
    # 路径规划: 用简单A*为每个AGV单独规划，不经过进货口/出货口
    # ============================================================
    def _plan_paths(self):
        replan_interval = 12
        for agv_id, agv in self.agvs.items():
            if agv.status not in [AGVStatus.MOVING_TO_PICKUP,
                                  AGVStatus.MOVING_TO_DELIVERY,
                                  AGVStatus.MOVING_TO_CHARGE]:
                continue
            if agv.goal_pos is None:
                continue

            need = (not agv.path or
                    agv.path_index >= len(agv.path) - 2 or
                    agv.waiting_steps > 12 or
                    self.current_step % replan_interval == 0)

            if not need:
                continue

            # 为单AGV规划路径，只考虑静态障碍物
            # 构造带临时占用的网格（其他AGV当前位置视为临时障碍）
            temp_grid = [row[:] for row in self.env.grid]
            for aid, a in self.agvs.items():
                if aid != agv_id and a.position != agv.position:
                    px, py = a.position
                    if temp_grid[py][px] == 0:
                        temp_grid[py][px] = 1  # 视为临时障碍

            path = a_star_search(agv.position, agv.goal_pos, temp_grid,
                                 self.env.width, self.env.height,
                                 max_steps=2000)
            if path and len(path) > 1:
                agv.path = path
                agv.path_index = 0
                agv.waiting_steps = 0

    # ============================================================
    # 电池管理: 消耗、低电告警、自动充电、充电恢复
    # ============================================================
    def _consume_battery(self, agv, multiplier: float = 1.0):
        """消耗电量。multiplier: 1.0=空载, 1.5=满载, 2.0=RL绕行, 0.3=等待"""
        consumption = self.battery_consumption * multiplier
        agv.battery = max(0.0, agv.battery - consumption)

    def _find_nearest_charging_station(self, pos):
        """找到距离pos最近的充电站"""
        if not self.charging_stations:
            return None
        return min(self.charging_stations,
                   key=lambda cs: manhattan_distance(pos, cs))

    def _manage_battery(self):
        """管理所有AGV的电量：充电恢复 / 低电告警并导航到充电站"""
        LOW_THRESHOLD = 30.0
        RESUME_THRESHOLD = 80.0

        for agv_id, agv in self.agvs.items():
            # Case 1: 正在充电站充电
            if agv.status == AGVStatus.CHARGING:
                agv.battery = min(self.battery_capacity,
                                  agv.battery + self.charge_rate)
                if agv.battery >= RESUME_THRESHOLD:
                    agv.status = AGVStatus.IDLE
                    agv.goal_pos = None
                    agv.path = []
                    agv.path_index = 0
                    agv.waiting_steps = 0
                    self.logger.info(
                        f"AGV {agv_id}: 充电完成 ({agv.battery:.1f}%), 恢复空闲")
                continue

            # Case 2: 低电量 → 中断任务，前往充电
            if (agv.battery <= LOW_THRESHOLD
                    and agv.status not in [AGVStatus.CHARGING,
                                           AGVStatus.MOVING_TO_CHARGE]):
                # 放弃当前任务
                if agv.current_task is not None:
                    task_id = agv.current_task.task_id
                    self.logger.warning(
                        f"AGV {agv_id}: 电量过低 ({agv.battery:.1f}%), "
                        f"中断任务 {task_id}, 前往充电")
                    if self.task_allocator:
                        self.task_allocator.abandon_task(task_id, agv_id)
                    agv.current_task = None
                    agv.is_loaded = False

                # 导航到最近充电站
                target = self._find_nearest_charging_station(agv.position)
                if target is not None:
                    agv.status = AGVStatus.MOVING_TO_CHARGE
                    agv.goal_pos = target
                    agv.path = []
                    agv.path_index = 0
                    agv.waiting_steps = 0

    # ============================================================
    # 死锁检测与恢复: 每10步构建等待图，DFS判环并回退打破
    # ============================================================
    def _check_deadlock_and_recover(self):
        """检测死锁并在检测到时恢复。"""
        # 构建DeadlockDetector所需的agv_states字典
        agv_states = {}
        for agv_id, agv in self.agvs.items():
            if agv.status in [AGVStatus.MOVING_TO_PICKUP,
                              AGVStatus.MOVING_TO_DELIVERY,
                              AGVStatus.MOVING_TO_CHARGE]:
                agv_states[agv_id] = {
                    'position': agv.position,
                    'goal_pos': agv.goal_pos,
                    'path': agv.path,
                    'path_index': agv.path_index,
                    'is_loaded': agv.is_loaded,
                }

        occupied = set(a.position for a in self.agvs.values())

        # 检测死锁
        cycle = self.deadlock_detector.detect(
            agv_states, occupied, self.current_step)
        if cycle is None:
            return

        # 检测到死锁 → 执行恢复
        recovery = self.deadlock_detector.recover(agv_states, cycle)
        recovered_agv = None
        for agv_id, new_pos in recovery.items():
            if agv_id in self.agvs:
                agv = self.agvs[agv_id]
                # 验证新位置有效（在边界内且未被占用）
                if (0 <= new_pos[0] < self.env.width
                        and 0 <= new_pos[1] < self.env.height
                        and new_pos not in occupied):
                    agv.position = new_pos
                    agv.waiting_steps = 0
                    agv.path = []       # 清空路径，下一步_plan_paths会重规划
                    agv.path_index = 0
                    recovered_agv = agv_id

        # 记录死锁事件
        self.metrics.record_deadlock(
            cycle, recovered_agv or -1, self.current_step)

    # ============================================================
    # AGV移动: 严格跟踪A*路径，受阻时短暂等待或RL绕行
    # ============================================================
    def _move_agvs(self):
        all_positions = {aid: a.position for aid, a in self.agvs.items()}
        occupied = set(all_positions.values())
        obs_set = set(o.position for o in self.env.obstacles)

        for agv_id, agv in self.agvs.items():
            if agv.status not in [AGVStatus.MOVING_TO_PICKUP,
                                  AGVStatus.MOVING_TO_DELIVERY,
                                  AGVStatus.MOVING_TO_CHARGE]:
                continue
            if agv.goal_pos is None:
                continue
            if agv.position == agv.goal_pos:
                self._handle_arrival(agv)
                continue

            # 确定目标步: A*路径的下一步
            target_pos = agv.goal_pos
            if agv.path and agv.path_index + 1 < len(agv.path):
                target_pos = agv.path[agv.path_index + 1]
            elif agv.path and agv.path_index < len(agv.path):
                target_pos = agv.path[agv.path_index]

            # 如果目标步被占用(其他AGV/障碍物) → 等待或RL绕行
            blocked = (target_pos in occupied and target_pos != agv.position)
            blocked |= (target_pos in obs_set)

            if blocked and target_pos != agv.goal_pos:
                # 目标步被挡 → 等待(容忍限度内) 或 RL绕行
                if agv.waiting_steps < 6:
                    agv.waiting_steps += 1
                    self._consume_battery(agv, 0.3)
                    continue
                else:
                    # 等待太久 → RL选绕行方向
                    self._rl_step_around(agv, target_pos, all_positions,
                                         occupied, obs_set)
            elif blocked and target_pos == agv.goal_pos:
                # 最终目标被挡(例如另一个AGV在卸货口) → 等待
                agv.waiting_steps += 1
                self._consume_battery(agv, 0.3)
                continue
            else:
                # 目标步空闲 → 直接移动
                self._move_to(agv, target_pos, all_positions, occupied)

        # 记录指标 + RL训练
        self.metrics.record_step(
            {aid: {'position': a.position, 'status': str(a.status),
                   'is_loaded': a.is_loaded, 'battery': a.battery}
             for aid, a in self.agvs.items()},
            self.total_tasks_completed, self.current_step)

        if self.dqn_agent.epsilon > self.dqn_agent.epsilon_end:
            loss = self.dqn_agent.train_step()
            if self.current_step % 20 == 0:
                self.metrics.record_training(loss, None, self.dqn_agent.epsilon)

    def _move_to(self, agv, target_pos, all_positions, occupied):
        """将AGV移动到target_pos，更新路径索引"""
        old_pos = agv.position
        agv.position = target_pos
        occupied.discard(old_pos)
        occupied.add(target_pos)
        all_positions[agv.agv_id] = target_pos
        agv.waiting_steps = 0
        self.total_steps_taken += 1
        multiplier = 1.5 if agv.is_loaded else 1.0
        self._consume_battery(agv, multiplier)

        # 更新路径索引
        if agv.path and agv.path_index + 1 < len(agv.path):
            if target_pos == agv.path[agv.path_index + 1]:
                agv.path_index += 1

    def _rl_step_around(self, agv, intended_target, all_positions,
                        occupied, obs_set):
        """RL选择绕行方向，避开阻挡后继续朝目标前进"""
        other_agvs = [p for aid, p in all_positions.items() if aid != agv.agv_id]

        # 用A*路径的下两步作为子目标(如果存在)
        sub_goal = intended_target
        if agv.path and agv.path_index + 2 < len(agv.path):
            sub_goal = agv.path[agv.path_index + 2]

        valid = self._get_valid_actions(agv.position, occupied | obs_set)
        if not valid:
            agv.waiting_steps += 1
            return

        # RL决策
        local, gvec = self.dqn_agent.encoder.encode(
            agv.position, sub_goal, self.env.grid,
            list(obs_set), other_agvs,
            battery=agv.battery, is_loaded=agv.is_loaded, priority=1)
        action = self.dqn_agent.select_action(local, gvec, valid)

        dx, dy = ACTION_DELTAS[ACTIONS[action]]
        new_pos = (agv.position[0] + dx, agv.position[1] + dy)

        if action == 4:
            agv.waiting_steps += 1
            self._consume_battery(agv, 0.3)
        else:
            old_pos = agv.position
            agv.position = new_pos
            occupied.discard(old_pos)
            occupied.add(new_pos)
            all_positions[agv.agv_id] = new_pos
            agv.waiting_steps = 0
            self.total_steps_taken += 1
            self._consume_battery(agv, 2.0)

            # 存储RL经验
            next_other = [p for aid, p in all_positions.items() if aid != agv.agv_id]
            next_local, next_gvec = self.dqn_agent.encoder.encode(
                agv.position, sub_goal, self.env.grid,
                list(obs_set), next_other,
                battery=agv.battery, is_loaded=agv.is_loaded)

            prev_dist = abs(old_pos[0]-sub_goal[0]) + abs(old_pos[1]-sub_goal[1])
            curr_dist = abs(agv.position[0]-sub_goal[0]) + abs(agv.position[1]-sub_goal[1])
            reward = 0.5 if curr_dist < prev_dist else -0.5
            reward -= 0.05

            arrived = (agv.position == agv.goal_pos)
            self.dqn_agent.store_experience(
                local, gvec, action, reward, next_local, next_gvec, arrived)

    def _get_valid_actions(self, pos, occupied):
        valid = []
        for a_idx in range(5):
            dx, dy = ACTION_DELTAS[ACTIONS[a_idx]]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                cell = self.env.grid[ny][nx]
                if cell == 1:
                    continue
                is_zone = cell in (2, 3, 4)  # 装货口/卸货口/充电站允许多AGV进入
                if (dx, dy) == (0, 0) or (nx, ny) not in occupied or is_zone:
                    valid.append(a_idx)
        if not valid:
            valid.append(4)
        return valid

    # ============================================================
    # 到达处理
    # ============================================================
    def _handle_arrival(self, agv: AGVRuntimeState):
        if agv.status == AGVStatus.MOVING_TO_PICKUP:
            agv.is_loaded = True
            if agv.current_task:
                agv.goal_pos = agv.current_task.delivery_pos
                agv.status = AGVStatus.MOVING_TO_DELIVERY
                agv.path = []
                agv.path_index = 0
                agv.waiting_steps = 0
                self.logger.info(f"AGV {agv.agv_id}: 已取货 → 送货点 {agv.goal_pos}")

        elif agv.status == AGVStatus.MOVING_TO_CHARGE:
            agv.status = AGVStatus.CHARGING
            agv.goal_pos = None
            agv.path = []
            agv.path_index = 0
            agv.waiting_steps = 0
            self.logger.info(
                f"AGV {agv.agv_id}: 到达充电站，开始充电 "
                f"(电量: {agv.battery:.1f}%)")

        elif agv.status == AGVStatus.MOVING_TO_DELIVERY:
            agv.is_loaded = False
            if agv.current_task:
                task_id = agv.current_task.task_id
                self.total_tasks_completed += 1
                self.logger.info(f"AGV {agv.agv_id}: ★ 任务 {task_id} 完成!")
                self.metrics.record_task_complete(
                    task_id, agv.agv_id, self.total_steps_taken,
                    self.current_step, self.current_step)
                self.publish(MessageType.ENV_TASK_COMPLETED, {
                    "task_id": task_id,
                    "agv_id": agv.agv_id,
                    "position": agv.position,
                    "step": self.current_step
                })
            agv.current_task = None
            agv.goal_pos = None
            agv.status = AGVStatus.IDLE
            agv.path = []
            agv.path_index = 0
            agv.waiting_steps = 0

    def _check_task_completion(self):
        pass

    def _publish_state(self):
        agv_states = {}
        for agv_id, agv in self.agvs.items():
            agv_states[agv_id] = {
                "position": agv.position,
                "status": agv.status.value,
                "battery": agv.battery,
                "is_loaded": agv.is_loaded,
                "goal_pos": agv.goal_pos,
                "waiting_steps": agv.waiting_steps
            }
        self.publish(MessageType.ENV_STATE_UPDATE, {
            "step": self.current_step,
            "agvs": agv_states,
            "total_tasks_completed": self.total_tasks_completed,
            "total_steps_taken": self.total_steps_taken
        })

    def get_agv_positions(self) -> Dict[int, Tuple[int, int]]:
        return {aid: agv.position for aid, agv in self.agvs.items()}

    def get_statistics(self) -> Dict:
        idle = sum(1 for a in self.agvs.values() if a.status == AGVStatus.IDLE)
        moving = sum(1 for a in self.agvs.values() if a.status in
                     [AGVStatus.MOVING_TO_PICKUP, AGVStatus.MOVING_TO_DELIVERY])
        charging = sum(1 for a in self.agvs.values() if a.status in
                       [AGVStatus.CHARGING, AGVStatus.MOVING_TO_CHARGE])
        return {
            "step": self.current_step,
            "agvs": {"total": len(self.agvs), "idle": idle,
                     "moving": moving, "charging": charging},
            "tasks_completed": self.total_tasks_completed,
            "steps_taken": self.total_steps_taken
        }
