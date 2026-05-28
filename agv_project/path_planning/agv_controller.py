"""
============================================
AGV控制器模块
============================================
本模块整合了MAPF全局规划和RL实时避撞，
控制AGV在仓库中的移动、取货和送货。

核心功能：
1. 接收调度模块的任务分配
2. 使用MAPF规划全局无冲突路径
3. 使用RL处理动态避撞
4. 管理AGV的取货/送货流程
5. 更新AGV位置到环境

使用方式：
    from path_planning.agv_controller import AGVController
    controller = AGVController(env, mapf_planner, rl_avoidance)
    controller.step()
"""

import sys
import os
import logging
import math
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field

# 将项目根目录添加到系统路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from interface.communication import BaseModule, MessageType, Message
from interface.data_types import Task, TaskStatus, AGVState, AGVStatus
from interface.data_types import manhattan_distance
from path_planning.mapf_planner import MAPFPlanner
from path_planning.rl_collision_avoidance import RLCollisionAvoidance, ACTIONS, ACTION_DELTAS
from path_planning.rl.dqn_agent import DQNAgent
from path_planning.deadlock_detector import DeadlockDetector
from path_planning.metrics_collector import MetricsCollector


# ==========================================
# AGV运行时状态
# ==========================================

@dataclass
class AGVRuntimeState:
    """
    AGV运行时状态（扩展AGVState）
    
    Attributes:
        agv_id: AGV ID
        position: 当前位置
        status: AGV状态
        battery: 电量
        is_loaded: 是否已装载货物
        current_task: 当前执行的任务
        path: 当前规划的路径
        path_index: 路径中的当前位置索引
        goal_pos: 当前目标位置（取货点或送货点）
        waiting_steps: 等待步数计数
    """
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
    """
    AGV控制器
    
    整合MAPF全局规划和RL实时避撞，控制所有AGV的移动。
    通过消息总线与调度模块和环境模块通信。
    
    Usage:
        >>> controller = AGVController(env, mapf_planner, rl_avoidance)
        >>> controller.reset()
        >>> for _ in range(100):
        ...     controller.step()
    """
    
    def __init__(self, env, mapf_planner: MAPFPlanner, 
                 rl_avoidance: RLCollisionAvoidance):
        """
        初始化AGV控制器
        
        Args:
            env: 仓库环境实例
            mapf_planner: MAPF规划器
            rl_avoidance: RL避撞控制器
        """
        super().__init__("controller")

        self.env = env
        self.mapf_planner = mapf_planner
        self.rl_avoidance = rl_avoidance

        # AGV运行时状态
        self.agvs: Dict[int, AGVRuntimeState] = {}

        # 调度模块引用（由外部设置）
        self.task_allocator = None

        # 仿真状态
        self.current_step = 0
        self.is_running = False

        # 统计信息
        self.total_tasks_completed = 0
        self.total_steps_taken = 0

        # ===== 新增: RL核心模块 =====
        self.dqn_agent = DQNAgent(grid_size=15)
        self.deadlock_detector = DeadlockDetector()
        self.metrics = MetricsCollector()

        # 标记是否使用 RL 主导路径规划
        self.use_rl_primary = True  # True=RL主导, False=CBS主导(兼容旧模式)
        self.use_random_policy = False

        # ===== RL训练相关 =====
        self.agv_prev_states: Dict[int, any] = {}
        self.agv_prev_positions: Dict[int, Tuple[int, int]] = {}

        self.train_step_counter = 0
        self.rl_train_interval = 4
        
        self.logger.info("AGV控制器初始化完成")
    
    def set_task_allocator(self, task_allocator):
        """设置调度模块引用"""
        self.task_allocator = task_allocator
    
    def _setup_subscriptions(self):
        """设置消息订阅"""
        self.subscribe(MessageType.SCHEDULER_TASK_ASSIGNED, self._on_task_assigned)
        self.subscribe(MessageType.MAIN_SIMULATION_START, self._on_simulation_start)
        self.subscribe(MessageType.MAIN_SIMULATION_STOP, self._on_simulation_stop)
        self.subscribe(MessageType.MAIN_SIMULATION_RESET, self._on_simulation_reset)
    
    def _on_task_assigned(self, message: Message):
        """
        处理任务分配消息
        
        为AGV设置目标，规划路径。
        """
        data = message.data
        task_id = data.get("task_id")
        agv_id = data.get("agv_id")
        pickup_pos = data.get("pickup_pos")
        delivery_pos = data.get("delivery_pos")
        
        if agv_id not in self.agvs:
            self.logger.warning(f"AGV {agv_id} 不存在")
            return
        
        agv = self.agvs[agv_id]
        
        # 创建任务对象
        from interface.data_types import Task, TaskStatus
        task = Task(
            task_id=task_id,
            pickup_pos=pickup_pos,
            delivery_pos=delivery_pos,
            priority=1,
            status=TaskStatus.ASSIGNED,
            assigned_agv_id=agv_id
        )
        agv.current_task = task
        
        # 设置第一阶段目标：取货点
        agv.goal_pos = pickup_pos
        agv.status = AGVStatus.MOVING_TO_PICKUP
        
        self.logger.info(f"AGV {agv_id}: 任务{task_id} 前往取货点 {pickup_pos}")
    
    def _on_simulation_start(self, message: Message):
        """处理仿真开始消息"""
        self.is_running = True
        self.reset()
        self.logger.info("控制器：仿真开始")
    
    def _on_simulation_stop(self, message: Message):
        """处理仿真停止消息"""
        self.is_running = False
        self.logger.info(f"控制器：仿真停止，完成 {self.total_tasks_completed} 个任务")
    
    def _on_simulation_reset(self, message: Message):
        """处理仿真重置消息"""
        self.reset()
    
    def reset(self):
        """重置AGV控制器"""
        self.current_step = 0
        self.agvs.clear()
        self.total_tasks_completed = 0
        self.total_steps_taken = 0
        
        # 初始化AGV
        from scheduler.task_allocator import AGV_INITIAL_POSITIONS, NUM_AGVS
        for i in range(NUM_AGVS):
            pos = AGV_INITIAL_POSITIONS[i]
            self.agvs[i] = AGVRuntimeState(
                agv_id=i,
                position=pos,
                status=AGVStatus.IDLE
            )
        
        self.logger.info("控制器已重置")
    
    def step(self):
        """
        执行一步AGV控制
        
        流程：
        1. 步数递增
        2. 为活跃AGV规划/更新路径
        3. 移动所有AGV
        4. 检查任务完成状态
        5. 发布状态更新
        """
        self.current_step += 1
        
        # 1. 为活跃AGV规划路径
        self._plan_paths()
        
        # 2. 移动所有AGV
        self._move_agvs()
        
        # 3. 检查任务完成
        self._check_task_completion()
        
        # 4. 发布状态更新
        self._publish_state()
    
    def _plan_paths(self):
        """
        为所有活跃AGV规划全局路径。
        增量更新：只在无路径、卡住、或定期重规划时调用CBS。
        """
        replan_interval = 8  # 每8步重规划，让路径实时适应障碍物变化
        agents_to_plan = []
        for agv_id, agv in self.agvs.items():
            if agv.status in [AGVStatus.MOVING_TO_PICKUP, AGVStatus.MOVING_TO_DELIVERY]:
                if agv.goal_pos is None:
                    continue
                need_replan = (
                    not agv.path or  # 无路径
                    agv.path_index >= len(agv.path) - 1 or  # 路径走完
                    agv.waiting_steps > 15 or  # 卡住太久
                    self.current_step % replan_interval == 0  # 定期重规划
                )
                if need_replan:
                    agents_to_plan.append((agv_id, agv.position, agv.goal_pos))
        
        if not agents_to_plan:
            return
        
        # 获取当前障碍物位置
        obstacles = set(o.position for o in self.env.obstacles)
        
        # 获取其他AGV当前位置
        other_positions = set()
        for aid, a in self.agvs.items():
            if aid not in [a[0] for a in agents_to_plan]:
                other_positions.add(a.position)
        
        # 使用MAPF规划路径
        solution = self.mapf_planner.solve(agents_to_plan, occupied_positions=other_positions)

        # 更新AGV路径（带平滑，去除冗余转折点）
        for agv_id, path in solution.items():
            if agv_id in self.agvs and len(path) > 1:
                smoothed = self._smooth_path(path, obstacles)
                self.agvs[agv_id].path = smoothed
                self.agvs[agv_id].path_index = 0

    def _smooth_path(self, path, obstacles):
        """路径平滑：移除直线可达的中间路标点，消除不必要的转折"""
        if len(path) <= 2:
            return path
        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            # 从当前位置尽可能远地沿直线跳
            furthest = i + 1
            for j in range(len(path) - 1, i, -1):
                if self._line_clear(path[i], path[j], obstacles):
                    furthest = j
                    break
            result.append(path[furthest])
            i = furthest
        return result

    def _line_clear(self, p1, p2, obstacles):
        """检查p1到p2的直线路径是否无障碍。
        移动障碍物不阻挡——它们会移动，不应让AGV绕远路。
        只检查静态障碍物（grid值为1的墙壁/货架）。"""
        x1, y1 = p1; x2, y2 = p2
        dx = abs(x2 - x1); dy = abs(y2 - y1)
        sx = 1 if x2 > x1 else -1 if x2 < x1 else 0
        sy = 1 if y2 > y1 else -1 if y2 < y1 else 0
        err = dx - dy
        cx, cy = x1, y1
        while (cx, cy) != (x2, y2):
            if (cx, cy) != (x1, y1):
                if 0 <= cx < self.env.width and 0 <= cy < self.env.height:
                    if self.env.grid[cy][cx] == 1:
                        return False
                else:
                    return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; cx += sx
            if e2 < dx:
                err += dx; cy += sy
        return True
    
    def _move_agvs(self):
        """RL驱动的AGV移动 — 每步用策略网络决策方向"""
        all_positions = {aid: a.position for aid, a in self.agvs.items()}
        occupied = set(all_positions.values())
        obstacle_positions = set(o.position for o in self.env.obstacles)

        # 死锁检测
        agv_state_snapshots = {
            aid: {
                'position': a.position, 'goal_pos': a.goal_pos,
                'path': a.path, 'path_index': a.path_index,
                'waiting_steps': a.waiting_steps, 'is_loaded': a.is_loaded,
            }
            for aid, a in self.agvs.items()
        }
        deadlock_cycle = self.deadlock_detector.detect(
            agv_state_snapshots, occupied | obstacle_positions, self.current_step)

        if deadlock_cycle:
            recovery = self.deadlock_detector.recover(agv_state_snapshots, deadlock_cycle)
            for agv_id, new_pos in recovery.items():
                if agv_id in self.agvs:
                    self.agvs[agv_id].position = new_pos
                    self.agvs[agv_id].waiting_steps = 0
            self.metrics.record_deadlock(deadlock_cycle,
                list(recovery.keys())[0] if recovery else -1, self.current_step)

        for agv_id, agv in self.agvs.items():
            if agv.status not in [AGVStatus.MOVING_TO_PICKUP, AGVStatus.MOVING_TO_DELIVERY]:
                continue
            if agv.goal_pos is None:
                continue

            # 到达目标检查
            if agv.position == agv.goal_pos:
                self._handle_arrival(agv)
                continue

            if self.use_random_policy:
                self._random_move_agv(agv, all_positions, occupied, obstacle_positions)
            elif self.use_rl_primary:
                self._rl_move_agv(agv, all_positions, occupied, obstacle_positions)
            else:
                self._mapf_move_agv(agv, all_positions, occupied, obstacle_positions)

        # 记录指标
        self.metrics.record_step(
            {aid: {'position': a.position, 'status': str(a.status),
                   'is_loaded': a.is_loaded, 'battery': a.battery}
             for aid, a in self.agvs.items()},
            self.total_tasks_completed, self.current_step)

        # RL训练步骤
        if self.dqn_agent.epsilon > self.dqn_agent.epsilon_end:
            loss = self.dqn_agent.train_step()
            self.metrics.record_training(loss, None, self.dqn_agent.epsilon)

    def _rl_move_agv(self, agv, all_positions, occupied, obstacle_positions):
        """RL 策略网络决策移动方向，以 MAPF 路径路标为子目标"""
        other_agvs = [pos for aid, pos in all_positions.items() if aid != agv.agv_id]

        # 确定子目标：优先用5步前瞻（更远的目标让RL有更好的方向感）
        sub_goal = agv.goal_pos
        if agv.path and agv.path_index < len(agv.path) - 1:
            lookahead = min(5, len(agv.path) - agv.path_index - 1)
            sub_goal = agv.path[agv.path_index + lookahead]

        dist_to_subgoal = abs(agv.position[0]-sub_goal[0]) + abs(agv.position[1]-sub_goal[1])
        use_direct = dist_to_subgoal > 12

        valid = self._get_valid_rl_actions(agv.position, occupied | obstacle_positions)

        # 防振荡：禁止直接反向上一帧的移动
        REVERSE = {0: 1, 1: 0, 2: 3, 3: 2}
        last_action = getattr(agv, 'last_action', None)
        if last_action is not None and last_action in REVERSE:
            reverse = REVERSE[last_action]
            no_reverse = [a for a in valid if a != reverse]
            if no_reverse:
                valid = no_reverse

        if not valid:
            agv.waiting_steps += 1
            return

        if use_direct:
            # 远距离：贪心朝子目标移动
            # 计算每个动作的距离改进
            scored = []
            for a in valid:
                dx, dy = ACTION_DELTAS[ACTIONS[a]]
                nx, ny = agv.position[0]+dx, agv.position[1]+dy
                d = abs(nx-sub_goal[0]) + abs(ny-sub_goal[1])
                scored.append((d, a))
            scored.sort()

            best_dist, best_action = scored[0]

            # 关键修复：如果最佳动作是等待或距离没有改进
            # 检查是否朝子目标方向——如果不是，说明被挡住了，应该等待
            # 而不是绕路（绕路导致后续更多的绕路，形成绕圈）
            cur_dist = abs(agv.position[0]-sub_goal[0]) + abs(agv.position[1]-sub_goal[1])

            if best_action == 4:
                action = 4  # 等待
            elif best_dist >= cur_dist and agv.waiting_steps < 5:
                # 最佳动作也不能缩短距离 → 可能被挡住了，短暂等待
                action = 4
            else:
                action = best_action

        # 始终编码状态
        local, gvec = self.dqn_agent.encoder.encode(
            agv.position, sub_goal, self.env.grid,
            list(obstacle_positions), other_agvs,
            battery=agv.battery, is_loaded=agv.is_loaded, priority=1)

        if not use_direct:
            action = self.dqn_agent.select_action(local, gvec, valid)

        agv.last_action = action

        dx, dy = ACTION_DELTAS[ACTIONS[action]]
        new_pos = (agv.position[0] + dx, agv.position[1] + dy)

        waited = (action == 4)
        old_pos = agv.position

        if not waited:
            agv.position = new_pos
            if old_pos in occupied:
                occupied.discard(old_pos)
            occupied.add(new_pos)
            all_positions[agv.agv_id] = new_pos
            agv.waiting_steps = 0
            self.total_steps_taken += 1

            # 沿路径前进时更新 path_index
            if agv.path and agv.path_index + 1 < len(agv.path):
                if new_pos == agv.path[agv.path_index + 1]:
                    agv.path_index += 1
        else:
            agv.waiting_steps += 1

        if new_pos in obstacle_positions:
            self.metrics.record_collision(agv.agv_id, self.current_step)
        collision_agv = sum(1 for aid, pos in all_positions.items()
                            if aid != agv.agv_id and pos == new_pos)

        next_other_agvs = [pos for aid, pos in all_positions.items() if aid != agv.agv_id]
        next_local, next_gvec = self.dqn_agent.encoder.encode(
            agv.position, sub_goal, self.env.grid,
            list(obstacle_positions), next_other_agvs,
            battery=agv.battery, is_loaded=agv.is_loaded)

        arrived = (agv.position == agv.goal_pos)
        congestion = [p for p in other_agvs
                      if abs(p[0]-agv.position[0])+abs(p[1]-agv.position[1]) <= 3]
        reward = DQNAgent.compute_reward(
            old_pos, agv.position, sub_goal, agv.is_loaded,
            arrived_pickup=arrived,
            obstacle_collision=(new_pos in obstacle_positions),
            agv_collision=(collision_agv > 0),
            waited=waited, battery=agv.battery,
            congestion_count=len(congestion))

        done = arrived or (new_pos in obstacle_positions) or (collision_agv > 0)
        self.dqn_agent.store_experience(local, gvec, action, reward, next_local, next_gvec, done)

    def _get_valid_rl_actions(self, pos, occupied):
        """获取 RL 的有效动作列表。装货口/卸货口允许多AGV进入。"""
        valid = []
        for a_idx in range(len(ACTIONS)):
            dx, dy = ACTION_DELTAS[ACTIONS[a_idx]]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                cell = self.env.grid[ny][nx]
                if cell == 1:
                    continue
                is_zone = cell in (2, 3)
                if (nx, ny) not in occupied or (dx, dy) == (0, 0) or is_zone:
                    valid.append(a_idx)
        if not valid:
            valid.append(4)
        return valid

    def _mapf_move_agv(self, agv, all_positions, occupied, obstacle_positions):
        """MAPF路径移动（CBS fallback）"""
        prev_pos = agv.position

        if agv.path and agv.path_index < len(agv.path) - 1:
            next_pos = agv.path[agv.path_index + 1]
            if next_pos not in occupied or next_pos == agv.position:
                if next_pos not in obstacle_positions:
                    old_pos = agv.position
                    agv.position = next_pos
                    agv.path_index += 1
                    occupied.discard(old_pos)
                    occupied.add(next_pos)
                    all_positions[agv.agv_id] = next_pos
                    self.total_steps_taken += 1
                    return

        self._rl_avoid(agv, all_positions, occupied, obstacle_positions)

    def _random_move_agv(self, agv, all_positions, occupied, obstacle_positions):
        """随机移动策略"""
        import random
        valid = self._get_valid_rl_actions(agv.position, occupied | obstacle_positions)
        if not valid:
            agv.waiting_steps += 1
            return
        action = random.choice(valid)
        dx, dy = ACTION_DELTAS[ACTIONS[action]]
        if action != 4:
            new_pos = (agv.position[0] + dx, agv.position[1] + dy)
            old_pos = agv.position
            agv.position = new_pos
            occupied.discard(old_pos)
            occupied.add(new_pos)
            all_positions[agv.agv_id] = new_pos
            agv.waiting_steps = 0
            self.total_steps_taken += 1
        else:
            agv.waiting_steps += 1
    
    def _rl_avoid(self, agv: AGVRuntimeState, all_positions: Dict[int, Tuple[int, int]],
                  occupied: Set[Tuple[int, int]], obstacle_positions: Set[Tuple[int, int]]):
        """
        使用RL进行实时避撞
        
        Args:
            agv: AGV运行时状态
            all_positions: 所有AGV位置
            occupied: 被占用的位置集合
            obstacle_positions: 障碍物位置集合
        """
        # 获取其他AGV位置
        other_agvs = [pos for aid, pos in all_positions.items() if aid != agv.agv_id]
        
        # 获取状态
        state = self.rl_avoidance.get_state(
            agv.position, agv.goal_pos,
            self.env.grid,
            list(obstacle_positions),
            other_agvs
        )
        
        # 获取有效动作
        all_occupied = occupied | obstacle_positions
        valid_actions = self.rl_avoidance.get_valid_actions(
            agv.position, self.env.grid, self.env.width, self.env.height,
            all_occupied
        )
        
        if not valid_actions:
            agv.waiting_steps += 1
            return
        
        # 选择动作
        action = self.rl_avoidance.select_action(state, valid_actions)
        
        # 记录移动前的位置
        prev_pos = agv.position
        
        # 应用动作
        new_pos = self.rl_avoidance.apply_action(agv.position, action)
        
        waited = False
        if new_pos != agv.position:
            old_pos = agv.position
            agv.position = new_pos
            if old_pos in occupied:
                occupied.remove(old_pos)
            occupied.add(new_pos)
            all_positions[agv.agv_id] = new_pos
            agv.waiting_steps = 0
            self.total_steps_taken += 1
        else:
            agv.waiting_steps += 1
            waited = True
        
        # 收集RL经验（训练模式下）
        if self.rl_avoidance.is_training:
            # 获取下一状态
            next_other_agvs = [pos for aid, pos in all_positions.items() if aid != agv.agv_id]
            next_state = self.rl_avoidance.get_state(
                agv.position, agv.goal_pos,
                self.env.grid,
                list(obstacle_positions),
                next_other_agvs
            )
            
            # 计算奖励
            arrived = (agv.position == agv.goal_pos)
            collided = (agv.position in obstacle_positions)
            task_completed = False  # 由 _handle_arrival 处理
            
            reward = self.rl_avoidance.compute_reward(
                agv.position, prev_pos, agv.goal_pos,
                arrived, collided, task_completed, waited
            )
            
            # 判断是否终止（到达目标或碰撞）
            done = arrived or collided
            
            # 存储经验
            self.rl_avoidance.store_experience(state, action, reward, next_state, done)
    
    def _handle_arrival(self, agv: AGVRuntimeState):
        """
        处理AGV到达目标
        
        Args:
            agv: AGV运行时状态
        """
        if agv.status == AGVStatus.MOVING_TO_PICKUP:
            # 到达取货点，装载货物
            agv.is_loaded = True
            agv.status = AGVStatus.LOADING
            
            # 设置下一目标：送货点
            if agv.current_task:
                agv.goal_pos = agv.current_task.delivery_pos
                agv.status = AGVStatus.MOVING_TO_DELIVERY
                agv.path = []
                agv.path_index = 0
                
                self.logger.info(f"AGV {agv.agv_id}: 已取货，前往送货点 {agv.goal_pos}")
        
        elif agv.status == AGVStatus.MOVING_TO_DELIVERY:
            # 到达送货点，卸载货物
            agv.is_loaded = False
            agv.status = AGVStatus.UNLOADING
            
            # 完成任务
            if agv.current_task:
                task_id = agv.current_task.task_id
                self.total_tasks_completed += 1
                
                self.logger.info(f"AGV {agv.agv_id}: 任务 {task_id} 完成！")
                
                # 发布任务完成消息
                self.publish(MessageType.ENV_TASK_COMPLETED, {
                    "task_id": task_id,
                    "agv_id": agv.agv_id,
                    "position": agv.position,
                    "step": self.current_step
                })
                
                # 重置AGV状态
                agv.current_task = None
                agv.goal_pos = None
                agv.status = AGVStatus.IDLE
                agv.path = []
                agv.path_index = 0
    
    def _check_task_completion(self):
        """检查是否有任务需要完成"""
        # 由 _handle_arrival 处理
        pass
    
    def _publish_state(self):
        """发布AGV状态更新"""
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
        """获取所有AGV位置"""
        return {aid: agv.position for aid, agv in self.agvs.items()}
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        idle = sum(1 for a in self.agvs.values() if a.status == AGVStatus.IDLE)
        moving = sum(1 for a in self.agvs.values() if a.status in 
                     [AGVStatus.MOVING_TO_PICKUP, AGVStatus.MOVING_TO_DELIVERY])
        loading = sum(1 for a in self.agvs.values() if a.status in 
                      [AGVStatus.LOADING, AGVStatus.UNLOADING])
        
        return {
            "step": self.current_step,
            "agvs": {
                "total": len(self.agvs),
                "idle": idle,
                "moving": moving,
                "loading": loading
            },
            "tasks_completed": self.total_tasks_completed,
            "steps_taken": self.total_steps_taken
        }


# ==========================================
# 独立运行测试
# ==========================================

def run_test():
    """测试AGV控制器"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("AGV控制器测试")
    print("=" * 60)
    
    # 创建环境
    from env.warehouse_env import WarehouseEnv
    env = WarehouseEnv()
    env.reset()
    
    # 创建MAPF规划器
    mapf = MAPFPlanner(env.grid, env.width, env.height)
    
    # 创建RL避撞
    rl = RLCollisionAvoidance()
    
    # 创建AGV控制器
    controller = AGVController(env, mapf, rl)
    controller.reset()
    
    print(f"\nAGV数量: {len(controller.agvs)}")
    for agv_id, agv in controller.agvs.items():
        print(f"  AGV {agv_id}: 位置 {agv.position}, 状态 {agv.status.value}")
    
    # 模拟任务分配
    print("\n模拟任务分配:")
    loading_zones = env.loading_zones
    unloading_zones = env.unloading_zones
    
    # 给前4台AGV分配任务
    for i in range(4):
        if i < len(controller.agvs):
            agv = controller.agvs[i]
            pickup = loading_zones[i % len(loading_zones)]
            delivery = unloading_zones[i % len(unloading_zones)]
            
            from interface.data_types import Task, TaskStatus
            task = Task(
                task_id=i,
                pickup_pos=pickup,
                delivery_pos=delivery,
                priority=1,
                status=TaskStatus.ASSIGNED,
                assigned_agv_id=i
            )
            
            agv.current_task = task
            agv.goal_pos = pickup
            agv.status = AGVStatus.MOVING_TO_PICKUP
            
            print(f"  AGV {i}: 取货 {pickup} → 送货 {delivery}")
    
    # 运行几步测试
    print(f"\n运行10步测试...")
    for step in range(10):
        controller.step()
        
        if step < 3 or step == 9:
            stats = controller.get_statistics()
            print(f"  步数 {step+1}: "
                  f"空闲={stats['agvs']['idle']}, "
                  f"移动中={stats['agvs']['moving']}, "
                  f"已完成={stats['tasks_completed']}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    run_test()
