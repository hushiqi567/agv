"""
============================================
任务分配算法模块
============================================
本模块实现了调度方案的核心逻辑，包括：
1. 泊松分布生成任务：每步按 Poisson(λ) 生成新货物
2. 出货口分配：从空闲出货口中等概率随机选择
3. 最近距离分配：将任务分配给距离取货点最近的空闲AGV
4. 通过消息总线与环境和路径规划模块通信

使用方式：
    python task_allocator.py  # 独立运行测试
"""

import sys
import os
import random
import logging
import math
import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field

# 将项目根目录添加到系统路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from interface.communication import BaseModule, MessageType, Message
from interface.data_types import Task, TaskStatus, AGVState, AGVStatus
from interface.data_types import manhattan_distance
from interface.config import get_config
from scheduler.od_flow import ODFlowManager


# ==========================================
# 常量定义
# ==========================================

# 泊松分布参数（平均每步到达的货物数量）
# 从全局配置读取
_config = get_config()
POISSON_LAMBDA = _config.simulation.poisson_lambda

# AGV数量
NUM_AGVS = 8

# AGV初始位置（左上侧4台，右上侧4台）
AGV_INITIAL_POSITIONS = [
    # 左上侧区域 (x=1~5, y=1~10)
    (2, 2), (2, 5), (2, 8), (5, 3),
    # 右上侧区域 (x=44~48, y=1~10)
    (47, 2), (47, 5), (47, 8), (44, 3),
]


class TaskAllocator(BaseModule):
    """
    任务分配器
    
    继承 BaseModule，通过消息总线与其他模块通信。
    负责：
    - 泊松分布生成新任务
    - 出货口占用管理
    - 最近距离任务分配
    
    Usage:
        >>> allocator = TaskAllocator(loading_zones, unloading_zones)
        >>> allocator.reset()
        >>> for _ in range(100):
        ...     allocator.step()
    """
    
    def __init__(self, loading_zones: List[Tuple[int, int]], 
                 unloading_zones: List[Tuple[int, int]]):
        """
        初始化任务分配器
        
        Args:
            loading_zones: 进货口位置列表
            unloading_zones: 出货口位置列表
        """
        super().__init__("scheduler")
        
        # 进货口和出货口
        self.loading_zones = loading_zones
        self.unloading_zones = unloading_zones
        
        # OD流程管理器
        self.od_flow = ODFlowManager(unloading_zones)
        
        # AGV控制器引用（由外部设置）
        self.controller = None
        
        # AGV状态管理
        self.agvs: Dict[int, AGVState] = {}
        self._init_agvs()
        
        # 泊松分布参数
        self.poisson_lambda = POISSON_LAMBDA
        
        # 随机数生成器
        self.rng = random.Random(42)
        self.np_rng = np.random.RandomState(42)
        
        # 仿真状态
        self.current_step = 0
        self.is_running = False
        
        self.logger.info(
            f"任务分配器初始化完成: "
            f"{len(loading_zones)}个进货口, "
            f"{len(unloading_zones)}个出货口, "
            f"{NUM_AGVS}台AGV"
        )
    
    def _init_agvs(self):
        """初始化AGV"""
        for i in range(NUM_AGVS):
            pos = AGV_INITIAL_POSITIONS[i]
            self.agvs[i] = AGVState(
                agv_id=i,
                position=pos,
                status=AGVStatus.IDLE
            )
    
    def _setup_subscriptions(self):
        """设置消息订阅"""
        self.subscribe(MessageType.ENV_STATE_UPDATE, self._on_env_state_update)
        self.subscribe(MessageType.ENV_TASK_COMPLETED, self._on_task_completed)
        self.subscribe(MessageType.MAIN_SIMULATION_START, self._on_simulation_start)
        self.subscribe(MessageType.MAIN_SIMULATION_STOP, self._on_simulation_stop)
        self.subscribe(MessageType.MAIN_SIMULATION_RESET, self._on_simulation_reset)
    
    def _on_env_state_update(self, message: Message):
        """
        处理环境状态更新消息
        
        更新AGV位置和状态信息。
        """
        data = message.data
        # 这里可以从环境状态中提取AGV位置信息
        # 目前环境模块还没有AGV，后续任务4会添加
        pass
    
    def _on_task_completed(self, message: Message):
        """
        处理任务完成消息
        
        释放出货口，更新AGV状态。
        """
        data = message.data
        task_id = data.get("task_id")
        agv_id = data.get("agv_id")
        
        if task_id is not None:
            self.od_flow.complete_task(task_id)
        
        if agv_id is not None and agv_id in self.agvs:
            self.agvs[agv_id].status = AGVStatus.IDLE
            self.agvs[agv_id].current_task = None
    
    def _on_simulation_start(self, message: Message):
        """处理仿真开始消息"""
        self.is_running = True
        self.reset()
        self.logger.info("调度模块：仿真开始")
    
    def _on_simulation_stop(self, message: Message):
        """处理仿真停止消息"""
        self.is_running = False
        self.logger.info(f"调度模块：仿真停止，共运行 {self.current_step} 步")
    
    def _on_simulation_reset(self, message: Message):
        """处理仿真重置消息"""
        self.reset()
    
    def reset(self):
        """重置任务分配器"""
        self.current_step = 0
        self.od_flow.reset()
        self._init_agvs()
        self.logger.info("调度模块已重置")
    
    def step(self):
        """
        执行一步调度
        
        流程：
        1. 步数递增
        2. 泊松分布生成新任务
        3. 尝试分配待分配任务给空闲AGV
        4. 发布调度结果消息
        """
        self.current_step += 1
        
        # 1. 泊松分布生成新任务
        self._generate_tasks()
        
        # 2. 分配待分配任务
        assigned_tasks = self._assign_pending_tasks()
        
        # 3. 发布消息
        if assigned_tasks:
            self._publish_task_assignments(assigned_tasks)
        
        # 4. 发布OD流程更新
        self._publish_od_flow_update()
        
        return assigned_tasks
    
    def _generate_tasks(self):
        """
        泊松分布生成新任务
        
        每步按 Poisson(λ) 生成新货物，每个货物：
        1. 从进货口中等概率随机选一个作为取货点
        2. 从空闲出货口中等概率随机选一个作为送货点
        """
        # 泊松分布生成货物数量
        num_new_tasks = self.np_rng.poisson(self.poisson_lambda)
        
        if num_new_tasks <= 0:
            return
        
        self.logger.debug(f"步数 {self.current_step}: 生成 {num_new_tasks} 个新货物")
        
        for _ in range(num_new_tasks):
            # 随机选一个进货口作为取货点
            pickup_pos = self.rng.choice(self.loading_zones)

            # 获取空闲出货口
            available_unloading = self.od_flow.get_available_unloading_zones()

            if not available_unloading:
                self.logger.debug("没有空闲出货口，跳过生成")
                continue

            # 从空闲出货口中等概率随机选一个
            delivery_pos = self.rng.choice(available_unloading)

            # 创建任务（含随机优先级1-5）
            task = self.od_flow.create_task(pickup_pos, delivery_pos)

            if task:
                task.create_time = self.current_step
                task.priority = self.rng.randint(1, 5)  # 随机优先级
                self.logger.info(
                    f"生成任务 {task.task_id}: "
                    f"取货 {pickup_pos} → 送货 {delivery_pos}"
                )
                
                # 发布新任务生成消息
                self.publish(MessageType.SCHEDULER_TASK_GENERATED, {
                    "task_id": task.task_id,
                    "pickup_pos": pickup_pos,
                    "delivery_pos": delivery_pos,
                    "create_time": self.current_step
                })
    
    def _assign_pending_tasks(self) -> List[Dict]:
        """
        分配待分配任务给空闲AGV（考虑负载均衡）

        避免多个AGV同时涌向同一取货口区域。
        """
        pending_tasks = self.od_flow.get_pending_tasks()
        assigned_tasks = []

        if not pending_tasks:
            return assigned_tasks

        # 高优先级任务先分配
        pending_tasks.sort(key=lambda t: t.priority, reverse=True)

        idle_agvs = {
            agv_id: agv for agv_id, agv in self.agvs.items()
            if agv.status == AGVStatus.IDLE
        }

        if not idle_agvs:
            return assigned_tasks

        # 统计正在前往的取货口
        active_pickups = {}
        for agv_id, agv in self.agvs.items():
            if agv.current_task and agv.status in [AGVStatus.MOVING, AGVStatus.MOVING_TO_PICKUP]:
                pk = agv.current_task.pickup_pos
                active_pickups[pk] = active_pickups.get(pk, 0) + 1

        for task in pending_tasks:
            if not idle_agvs:
                break

            best_agv_id = None
            best_score = float('inf')

            for agv_id, agv in idle_agvs.items():
                dist = manhattan_distance(agv.position, task.pickup_pos)
                # 如果AGV就在取货口旁边（距离<5），直接分配，不受拥挤惩罚影响
                congestion_penalty = 0 if dist < 5 else active_pickups.get(task.pickup_pos, 0) * 5
                score = dist + congestion_penalty
                if score < best_score:
                    best_score = score
                    best_agv_id = agv_id

            if best_agv_id is not None:
                if self.od_flow.assign_task(task.task_id, best_agv_id):
                    agv = idle_agvs[best_agv_id]
                    agv.status = AGVStatus.MOVING
                    agv.current_task = task
                    active_pickups[task.pickup_pos] = active_pickups.get(task.pickup_pos, 0) + 1

                    del idle_agvs[best_agv_id]

                    assigned_tasks.append({
                        "task_id": task.task_id,
                        "agv_id": best_agv_id,
                        "pickup_pos": task.pickup_pos,
                        "delivery_pos": task.delivery_pos,
                        "distance": best_score
                    })
                    
                    dist = manhattan_distance(
                        self.agvs[best_agv_id].position, task.pickup_pos)
                    self.logger.info(
                        f"分配任务 {task.task_id} → AGV {best_agv_id} "
                        f"(距离 {dist})"
                    )
        
        return assigned_tasks
    
    def _publish_task_assignments(self, assigned_tasks: List[Dict]):
        """
        发布任务分配结果消息
        
        Args:
            assigned_tasks: 分配的任务列表
        """
        for assignment in assigned_tasks:
            self.publish(MessageType.SCHEDULER_TASK_ASSIGNED, assignment)
    
    def _publish_od_flow_update(self):
        """发布OD流程更新消息"""
        stats = self.od_flow.get_statistics()
        self.publish(MessageType.SCHEDULER_OD_FLOW_UPDATED, {
            "step": self.current_step,
            "statistics": stats,
            "agv_states": {
                agv_id: {
                    "position": agv.position,
                    "status": agv.status.value,
                    "battery": agv.battery,
                    "is_loaded": agv.is_loaded
                }
                for agv_id, agv in self.agvs.items()
            }
        })
    
    def set_controller(self, controller):
        """
        设置AGV控制器引用
        
        Args:
            controller: AGVController实例
        """
        self.controller = controller
    
    def get_statistics(self) -> Dict:
        """
        获取调度统计信息
        
        Returns:
            统计信息字典
        """
        od_stats = self.od_flow.get_statistics()
        
        # AGV状态统计
        idle_count = sum(1 for agv in self.agvs.values() if agv.status == AGVStatus.IDLE)
        busy_count = NUM_AGVS - idle_count
        
        return {
            "step": self.current_step,
            "agvs": {
                "total": NUM_AGVS,
                "idle": idle_count,
                "busy": busy_count
            },
            "tasks": od_stats,
            "poisson_lambda": self.poisson_lambda
        }


# ==========================================
# 独立运行测试
# ==========================================

def run_test():
    """
    独立运行测试
    
    测试调度模块的核心逻辑：
    - 泊松分布生成任务
    - 出货口占用管理
    - 最近距离分配
    """
    import logging
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("调度模块测试")
    print("=" * 60)
    
    # 模拟进货口和出货口
    loading_zones = [(0, y) for y in [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]]
    loading_zones += [(49, y) for y in [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]]
    
    unloading_zones = [(24, y) for y in [4, 12, 20, 28, 36, 44]]
    unloading_zones += [(25, y) for y in [8, 16, 24, 32, 40, 48]]
    
    print(f"\n进货口数量: {len(loading_zones)}")
    print(f"出货口数量: {len(unloading_zones)}")
    print(f"AGV数量: {NUM_AGVS}")
    print(f"泊松分布 λ: {POISSON_LAMBDA}")
    
    # 创建任务分配器
    allocator = TaskAllocator(loading_zones, unloading_zones)
    allocator.reset()
    
    # 运行50步测试
    print(f"\n运行50步测试...")
    print(f"\n{'步数':<6} {'生成任务':<10} {'待分配':<8} {'活跃':<8} {'已完成':<8} {'空闲AGV':<8} {'占用出货口':<10}")
    print("-" * 60)
    
    for step in range(50):
        assigned = allocator.step()
        stats = allocator.get_statistics()
        
        if step < 10 or step % 10 == 9:
            print(
                f"{step+1:<6} "
                f"{stats['tasks']['total_tasks']:<10} "
                f"{stats['tasks']['pending']:<8} "
                f"{stats['tasks']['active']:<8} "
                f"{stats['tasks']['completed']:<8} "
                f"{stats['agvs']['idle']:<8} "
                f"{stats['tasks']['occupied_unloading_zones']:<10}"
            )
    
    print("-" * 60)
    print(f"\n最终统计:")
    stats = allocator.get_statistics()
    print(f"  总任务数: {stats['tasks']['total_tasks']}")
    print(f"  待分配: {stats['tasks']['pending']}")
    print(f"  活跃中: {stats['tasks']['active']}")
    print(f"  已完成: {stats['tasks']['completed']}")
    print(f"  空闲AGV: {stats['agvs']['idle']}")
    print(f"  占用出货口: {stats['tasks']['occupied_unloading_zones']}")
    print(f"  空闲出货口: {stats['tasks']['available_unloading_zones']}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    run_test()
