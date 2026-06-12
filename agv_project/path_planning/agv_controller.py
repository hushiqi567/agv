"""
AGV控制器 — RL主导路径规划 + CBS全局协调 + A*局部建议
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
    sub_goal: Optional[Tuple[int, int]] = None  # CBS规划的中间子目标
    waiting_steps: int = 0

class AGVController(BaseModule):
    """AGV控制器 — RL主导路径规划 + CBS全局协调 + A*局部建议"""

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
        self.cbs_plan_interval = 30  # CBS每30步重规划子目标
        self.rl_steps_since_a_star = {}  # 追踪RL自主导航步数

        # 电池系统配置
        cfg = ConfigManager()
        self.battery_capacity = cfg.agv.battery_capacity
        self.battery_consumption = cfg.agv.battery_consumption_per_step
        self.charge_rate = cfg.agv.charge_rate
        self.charging_stations = getattr(env, 'charging_stations', [])
        # AGV专属充电桩: AGV_id -> (x, y)
        self.agv_charger = {}
        for i in range(cfg.agv.num_agvs):
            if self.charging_stations:
                self.agv_charger[i] = self.charging_stations[i % len(self.charging_stations)]

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
        self._plan_subgoals_with_cbs()
        self._plan_paths()
        self._move_agvs_rl_primary()
        self._check_deadlock_and_recover()
        self._check_task_completion()
        self._publish_state()

    # ============================================================
    # CBS全局协调: 每30步为所有AGV规划无冲突子目标
    # ============================================================
    def _plan_subgoals_with_cbs(self):
        """CBS为多AGV规划全局无冲突子目标序列"""
        if self.current_step % self.cbs_plan_interval != 0:
            return

        # 收集所有移动中AGV的起点和终点
        starts = {}
        goals = {}
        for agv_id, agv in self.agvs.items():
            if agv.status in [AGVStatus.MOVING_TO_PICKUP,
                              AGVStatus.MOVING_TO_DELIVERY] and agv.goal_pos:
                starts[agv_id] = agv.position
                goals[agv_id] = agv.goal_pos

        if len(starts) < 2:
            # 单AGV: 直接以终点为子目标
            for agv_id in starts:
                self.agvs[agv_id].sub_goal = goals[agv_id]
            return

        # 调用MAPFPlanner的CBS规划
        try:
            agent_list = [(aid, starts[aid], goals[aid]) for aid in starts]
            cbs_paths = self.mapf_planner.solve(agent_list)

            for agv_id, path in cbs_paths.items():
                agv = self.agvs[agv_id]
                # 取路径的前方第5步作为子目标(不太远也不太近)
                if path and len(path) > 5:
                    agv.sub_goal = path[min(5, len(path) - 1)]
                elif path and len(path) > 1:
                    agv.sub_goal = path[-1]
                else:
                    agv.sub_goal = goals.get(agv_id, agv.goal_pos)
        except Exception:
            # CBS失败 → 各AGV直接用终点
            for agv_id in starts:
                self.agvs[agv_id].sub_goal = goals[agv_id]

    # ============================================================
    # A*局部建议: 快速算到子目标的静态路径(RL参考用)
    # ============================================================
    def _plan_paths(self):
        """A*快速路由: 为RL提供到子目标的静态路径建议"""
        replan_interval = 12
        for agv_id, agv in self.agvs.items():
            if agv.status not in [AGVStatus.MOVING_TO_PICKUP,
                                  AGVStatus.MOVING_TO_DELIVERY,
                                  AGVStatus.MOVING_TO_CHARGE]:
                continue

            # A*目标 = CBS子目标(优先) 或 最终目标
            a_star_target = agv.sub_goal if agv.sub_goal else agv.goal_pos
            if a_star_target is None:
                continue

            need = (not agv.path or
                    agv.path_index >= len(agv.path) - 2 or
                    agv.waiting_steps > 12 or
                    self.current_step % replan_interval == 0)

            if not need:
                continue

            # 为单AGV规划路径,只考虑静态障碍物
            temp_grid = [row[:] for row in self.env.grid]
            for aid, a in self.agvs.items():
                if aid != agv_id and a.position != agv.position:
                    px, py = a.position
                    if temp_grid[py][px] == 0:
                        temp_grid[py][px] = 1

            path = a_star_search(agv.position, a_star_target, temp_grid,
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
        """Consume battery. multiplier: 1.0=empty, 1.5=loaded, 2.0=RL detour, 0.3=wait"""
        consumption = self.battery_consumption * multiplier
        agv.battery = max(0.0, agv.battery - consumption)

    def _get_assigned_charger(self, agv_id):
        """返回该AGV的专属充电站位置"""
        return self.agv_charger.get(agv_id)

    def _manage_battery(self):
        """管理所有AGV的电量:充电恢复 / 低电告警并导航到充电站"""
        LOW_THRESHOLD = 20.0
        RESUME_THRESHOLD = 80.0
        IDLE_CHARGE_THRESHOLD = 45.0  # 空闲AGV电量低于此值主动充电

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

            # Case 2: 空闲AGV电池偏低 → 主动充电,避免死区
            if (agv.status == AGVStatus.IDLE
                    and agv.battery < IDLE_CHARGE_THRESHOLD
                    and agv.battery > LOW_THRESHOLD):
                target = self._get_assigned_charger(agv_id)
                if target is not None:
                    agv.status = AGVStatus.MOVING_TO_CHARGE
                    agv.goal_pos = target
                    agv.path = []
                    agv.path_index = 0
                    agv.waiting_steps = 0
                    continue

            # Case 3: 低电量 → 中断任务,前往充电
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
                target = self._get_assigned_charger(agv_id)
                if target is not None:
                    agv.status = AGVStatus.MOVING_TO_CHARGE
                    agv.goal_pos = target
                    agv.path = []
                    agv.path_index = 0
                    agv.waiting_steps = 0

    # ============================================================
    # 死锁检测与恢复: 每10步构建等待图,DFS判环并回退打破
    # ============================================================
    def _check_deadlock_and_recover(self):
        """检测死锁并在检测到时恢复."""
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
                    agv.path = []       # 清空路径,下一步_plan_paths会重规划
                    agv.path_index = 0
                    recovered_agv = agv_id

        # 记录死锁事件
        self.metrics.record_deadlock(
            cycle, recovered_agv or -1, self.current_step)

    # ============================================================
    # RL主导移动: RL每步决策, CBS子目标指引方向, A*提供安全建议
    # ============================================================
    def _move_agvs_rl_primary(self):
        all_positions = {aid: a.position for aid, a in self.agvs.items()}
        # IDLE AGV在功能区时不阻塞其他AGV进入该功能区
        occupied = set()
        for aid, pos in all_positions.items():
            agv = self.agvs[aid]
            cell = self.env.grid[pos[1]][pos[0]]
            if agv.status == AGVStatus.IDLE and cell in (2, 3, 4):
                continue  # 功能区上的IDLE AGV不占用
            occupied.add(pos)
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

            # 当前导航目标 = CBS子目标(优先) 或 最终目标
            nav_target = agv.sub_goal if agv.sub_goal else agv.goal_pos
            if agv.position == nav_target:
                # 到达子目标 → 清空,下次CBS重设
                agv.sub_goal = None
                agv.path = []
                continue

            # 获取A*建议步(作为RL的参考方向)
            a_star_step = None
            if agv.path and agv.path_index + 1 < len(agv.path):
                a_star_step = agv.path[agv.path_index + 1]
            elif agv.path and agv.path_index < len(agv.path):
                a_star_step = agv.path[agv.path_index]

            # ─── RL决策(主角) ───
            valid = self._get_valid_actions(agv.position, occupied | obs_set)
            rl_action = self._rl_select_action(agv, nav_target, all_positions,
                                               obs_set, valid)
            dx, dy = ACTION_DELTAS[ACTIONS[rl_action]]
            rl_new_pos = (agv.position[0] + dx, agv.position[1] + dy)

            # ─── 安全检查 ───
            prev_dist = manhattan_distance(agv.position, nav_target)
            new_dist = manhattan_distance(rl_new_pos, nav_target)
            steps_since_a_star = self.rl_steps_since_a_star.get(agv_id, 0)

            use_a_star = False
            if rl_action == 4:
                # RL选择等待 → 合理(可能在让路)
                pass
            elif rl_new_pos in occupied:
                # RL选了被占的格子 → A*纠正
                use_a_star = True
            elif new_dist > prev_dist + 3:
                # RL严重远离目标 → A*纠正
                use_a_star = True
                steps_since_a_star = 0
            elif new_dist > prev_dist:
                steps_since_a_star += 1
                if steps_since_a_star > 5:
                    # 连续6步远离 → A*纠正
                    use_a_star = True
                    steps_since_a_star = 0
            else:
                # RL朝目标前进 → 好,重置计数器
                steps_since_a_star = max(0, steps_since_a_star - 1)

            self.rl_steps_since_a_star[agv_id] = steps_since_a_star

            # ─── 执行(RL优先, A*兜底) ───
            if use_a_star and a_star_step and a_star_step not in occupied:
                self._move_to(agv, a_star_step, all_positions, occupied)
            elif rl_action == 4:
                agv.waiting_steps += 1
                self._consume_battery(agv, 0.3)
            else:
                old_pos = agv.position
                agv.position = rl_new_pos
                occupied.discard(old_pos)
                occupied.add(rl_new_pos)
                all_positions[agv.agv_id] = rl_new_pos
                agv.waiting_steps = 0
                self.total_steps_taken += 1
                self._consume_battery(agv, 2.0 if steps_since_a_star > 2 else
                                      (1.5 if agv.is_loaded else 1.0))
                # RL经验存储(用移动前的位置)
                self._store_rl_experience(agv, old_pos, nav_target, rl_action,
                                         all_positions, obs_set)

        # 记录指标
        self.metrics.record_step(
            {aid: {'position': a.position, 'status': str(a.status),
                   'is_loaded': a.is_loaded, 'battery': a.battery}
             for aid, a in self.agvs.items()},
            self.total_tasks_completed, self.current_step)

    def _rl_select_action(self, agv, nav_target, all_positions, obs_set, valid):
        """RL选择动作:用sub_goal做目标,A*路径做参考方向"""
        if not valid:
            return 4  # wait

        other_agvs = [p for aid, p in all_positions.items() if aid != agv.agv_id]
        local, gvec = self.dqn_agent.encoder.encode(
            agv.position, nav_target, self.env.grid,
            list(obs_set), other_agvs,
            battery=agv.battery, is_loaded=agv.is_loaded,
            priority=(agv.current_task.priority if agv.current_task else 1),
            charging_stations=self.charging_stations)
        return self.dqn_agent.select_action(local, gvec, valid)

    def _store_rl_experience(self, agv, old_pos, nav_target, action,
                             all_positions, obs_set):
        """存储RL经验: old_pos是移动前的位置, agv.position是移动后"""
        if self.dqn_agent.epsilon <= self.dqn_agent.epsilon_end:
            return

        other_agvs = [p for aid, p in all_positions.items() if aid != agv.agv_id]
        prev_local, prev_gvec = self.dqn_agent.encoder.encode(
            old_pos, nav_target, self.env.grid,
            list(obs_set), other_agvs,
            battery=agv.battery, is_loaded=agv.is_loaded,
            priority=(agv.current_task.priority if agv.current_task else 1),
            charging_stations=self.charging_stations)

        prev_dist = manhattan_distance(old_pos, nav_target)
        curr_dist = manhattan_distance(agv.position, nav_target)
        reward = 0.5 if curr_dist < prev_dist else (-0.5 if curr_dist > prev_dist else 0.0)
        reward -= 0.05

        arrived = (agv.position == nav_target or agv.position == agv.goal_pos)
        next_other = [p for aid, p in all_positions.items() if aid != agv.agv_id]
        next_local, next_gvec = self.dqn_agent.encoder.encode(
            agv.position, nav_target, self.env.grid,
            list(obs_set), next_other,
            battery=agv.battery, is_loaded=agv.is_loaded,
            priority=(agv.current_task.priority if agv.current_task else 1),
            charging_stations=self.charging_stations)

        self.dqn_agent.store_experience(
            prev_local, prev_gvec, action, reward,
            next_local, next_gvec, arrived)

    def _move_to(self, agv, target_pos, all_positions, occupied):
        """将AGV移动到target_pos,更新路径索引"""
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
                f"AGV {agv.agv_id}: 到达充电站,开始充电 "
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
            # 离开卸货口,腾出空间给其他AGV
            self._step_away_from_zone(agv)

    def _step_away_from_zone(self, agv: AGVRuntimeState):
        """AGV完成任务后从卸货口移开"""
        x, y = agv.position
        cell = self.env.grid[y][x]
        if cell not in (2, 3):
            return
        occupied = {a.position for a in self.agvs.values()}
        all_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # 尝试找到可用的相邻格(空地优先,功能区其次,任何可通行格兜底)
        for dx, dy in all_dirs:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.env.width and 0 <= ny < self.env.height
                    and self.env.grid[ny][nx] == 0
                    and (nx, ny) not in occupied):
                agv.position = (nx, ny)
                return
        for dx, dy in all_dirs:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.env.width and 0 <= ny < self.env.height
                    and self.env.grid[ny][nx] in (2, 3)
                    and (nx, ny) not in occupied):
                agv.position = (nx, ny)
                return
        # 最后手段: 移到任何可通行且非障碍物的相邻格
        for dx, dy in all_dirs:
            nx, ny = x + dx, y + dy
            if (0 <= nx < self.env.width and 0 <= ny < self.env.height
                    and self.env.grid[ny][nx] != 1
                    and (nx, ny) not in occupied):
                agv.position = (nx, ny)
                return

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
