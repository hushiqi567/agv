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
        
        # ===== RL训练相关 =====
        # 记录每个AGV的上一步状态和位置（用于经验收集）
        self.agv_prev_states: Dict[int, any] = {}      # agv_id -> prev_state
        self.agv_prev_positions: Dict[int, Tuple[int, int]] = {}  # agv_id -> prev_pos
        
        # 训练计数器
        self.train_step_counter = 0
        self.rl_train_interval = 4  # 每4步训练一次
        
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
        为所有活跃AGV规划路径
        
        使用MAPF为所有需要路径的AGV规划无冲突路径。
        """
        # 收集需要规划路径的AGV
        agents_to_plan = []
        for agv_id, agv in self.agvs.items():
            if agv.status in [AGVStatus.MOVING_TO_PICKUP, AGVStatus.MOVING_TO_DELIVERY]:
                if agv.goal_pos is not None:
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
        
        # 更新AGV路径
        for agv_id, path in solution.items():
            if agv_id in self.agvs:
                agv = self.agvs[agv_id]
                if len(path) > 1:
                    agv.path = path
                    agv.path_index = 0
    
    def _move_agvs(self):
        """
        移动所有AGV
        
        对每个AGV：
        1. 如果有MAPF路径，沿路径前进
        2. 如果前方有动态障碍物，使用RL避撞
        3. 更新位置
        4. 收集RL训练经验（训练模式下）
        """
        # 获取当前所有AGV位置
        all_positions = {aid: a.position for aid, a in self.agvs.items()}
        occupied = set(all_positions.values())
        
        # 获取障碍物位置
        obstacle_positions = set(o.position for o in self.env.obstacles)
        
        for agv_id, agv in self.agvs.items():
            if agv.status not in [AGVStatus.MOVING_TO_PICKUP, AGVStatus.MOVING_TO_DELIVERY]:
                continue
            
            if agv.goal_pos is None:
                continue
            
            # 记录移动前的位置（用于奖励计算）
            prev_pos = agv.position
            
            # 检查是否已到达目标
            if agv.position == agv.goal_pos:
                # 先收集经验（在 _handle_arrival 修改 goal_pos 之前）
                if self.rl_avoidance.is_training:
                    self._collect_arrival_experience(agv_id, agv, prev_pos, arrived=True)
                self._handle_arrival(agv)
                continue
            
            # 尝试沿MAPF路径前进
            moved = False
            if agv.path and agv.path_index < len(agv.path) - 1:
                next_pos = agv.path[agv.path_index + 1]
                
                # 检查下一位置是否被占用
                if next_pos not in occupied or next_pos == agv.position:
                    # 检查是否有障碍物
                    if next_pos not in obstacle_positions:
                        # 移动AGV
                        old_pos = agv.position
                        agv.position = next_pos
                        agv.path_index += 1
                        occupied.remove(old_pos)
                        occupied.add(next_pos)
                        all_positions[agv_id] = next_pos
                        moved = True
                        self.total_steps_taken += 1
                        
                        # 收集沿路径移动的经验
                        if self.rl_avoidance.is_training:
                            self._collect_path_move_experience(agv_id, agv, prev_pos, next_pos, 
                                                               occupied, obstacle_positions)
            
            if not moved:
                # 使用RL避撞
                self._rl_avoid(agv, all_positions, occupied, obstacle_positions)
        
        # 每步结束后执行RL训练
        if self.rl_avoidance.is_training:
            self._perform_rl_training()
    
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
    
    # ==========================================
    # RL训练辅助方法
    # ==========================================
    
    def _collect_path_move_experience(self, agv_id: int, agv: AGVRuntimeState,
                                       prev_pos: Tuple[int, int], new_pos: Tuple[int, int],
                                       occupied: Set[Tuple[int, int]], 
                                       obstacle_positions: Set[Tuple[int, int]]):
        """
        收集沿MAPF路径移动的经验
        
        Args:
            agv_id: AGV ID
            agv: AGV运行时状态
            prev_pos: 移动前位置
            new_pos: 移动后位置
            occupied: 被占用的位置集合
            obstacle_positions: 障碍物位置集合
        """
        # 获取移动前的状态
        other_agvs_before = [pos for aid, pos in self.get_agv_positions().items() if aid != agv_id]
        state = self.rl_avoidance.get_state(
            prev_pos, agv.goal_pos,
            self.env.grid,
            list(obstacle_positions),
            other_agvs_before
        )
        
        # 动作：沿路径移动的方向
        dx = new_pos[0] - prev_pos[0]
        dy = new_pos[1] - prev_pos[1]
        action = 4  # 默认wait
        for a_idx, action_name in enumerate(ACTIONS):
            adx, ady = ACTION_DELTAS[action_name]
            if (adx, ady) == (dx, dy):
                action = a_idx
                break
        
        # 获取移动后的状态
        other_agvs_after = [pos for aid, pos in self.get_agv_positions().items() if aid != agv_id]
        next_state = self.rl_avoidance.get_state(
            new_pos, agv.goal_pos,
            self.env.grid,
            list(obstacle_positions),
            other_agvs_after
        )
        
        # 计算奖励
        arrived = (new_pos == agv.goal_pos)
        collided = (new_pos in obstacle_positions)
        reward = self.rl_avoidance.compute_reward(
            new_pos, prev_pos, agv.goal_pos,
            arrived, collided, False, False
        )
        
        done = arrived or collided
        
        # 存储经验
        self.rl_avoidance.store_experience(state, action, reward, next_state, done)
    
    def _collect_arrival_experience(self, agv_id: int, agv: AGVRuntimeState,
                                     prev_pos: Tuple[int, int], arrived: bool):
        """
        收集到达目标时的经验
        
        Args:
            agv_id: AGV ID
            agv: AGV运行时状态
            prev_pos: 移动前位置
            arrived: 是否到达目标
        """
        # 获取到达前的状态
        other_agvs = [pos for aid, pos in self.get_agv_positions().items() if aid != agv_id]
        state = self.rl_avoidance.get_state(
            prev_pos, agv.goal_pos,
            self.env.grid,
            [o.position for o in self.env.obstacles],
            other_agvs
        )
        
        # 动作：最后一步的方向
        dx = agv.position[0] - prev_pos[0]
        dy = agv.position[1] - prev_pos[1]
        action = 4  # 默认wait
        for a_idx, action_name in enumerate(ACTIONS):
            adx, ady = ACTION_DELTAS[action_name]
            if (adx, ady) == (dx, dy):
                action = a_idx
                break
        
        # 到达后的状态（与当前位置相同，但标记为到达）
        next_state = self.rl_avoidance.get_state(
            agv.position, agv.goal_pos,
            self.env.grid,
            [o.position for o in self.env.obstacles],
            other_agvs
        )
        
        # 计算奖励（到达目标）
        task_completed = (agv.status == AGVStatus.IDLE and agv.current_task is None)
        reward = self.rl_avoidance.compute_reward(
            agv.position, prev_pos, agv.goal_pos,
            arrived, False, task_completed, False
        )
        
        done = True  # 到达目标，该段路径结束
        
        # 存储经验
        self.rl_avoidance.store_experience(state, action, reward, next_state, done)
    
    def _perform_rl_training(self):
        """
        执行RL训练
        
        每 rl_train_interval 步调用一次 train_step。
        """
        self.train_step_counter += 1
        
        if self.train_step_counter % self.rl_train_interval == 0:
            loss = self.rl_avoidance.train_step()
            if loss is not None and self.current_step % 20 == 0:
                self.logger.debug(
                    f"RL训练: 步数={self.current_step}, "
                    f"损失={loss:.4f}, "
                    f"探索率={self.rl_avoidance.epsilon:.3f}, "
                    f"经验池={len(self.rl_avoidance.memory)}"
                )
    
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
