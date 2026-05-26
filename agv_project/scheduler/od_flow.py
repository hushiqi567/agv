"""
============================================
OD流程管理模块
============================================
本模块管理任务的OD（Origin-Destination）流程，包括：
1. 任务池管理：维护所有任务的完整生命周期
2. 出货口占用管理：跟踪每个出货口的占用状态
3. 任务状态流转：PENDING → ASSIGNED → MOVING_TO_PICKUP → LOADING
                     → MOVING_TO_DELIVERY → UNLOADING → COMPLETED

使用方式：
    from scheduler.od_flow import ODFlowManager
"""

import sys
import os
import logging
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum

# 将项目根目录添加到系统路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from interface.data_types import Task, TaskStatus


class ODFlowManager:
    """
    OD流程管理器
    
    管理任务的完整生命周期和出货口的占用状态。
    
    Attributes:
        task_pool: 所有任务的字典 {task_id: Task}
        pending_tasks: 待分配任务列表
        active_tasks: 正在执行的任务列表
        completed_tasks: 已完成任务列表
        unloading_zone_occupied: 出货口占用状态 {position: task_id}
        next_task_id: 下一个任务ID
    """
    
    def __init__(self, unloading_zones: List[Tuple[int, int]]):
        """
        初始化OD流程管理器
        
        Args:
            unloading_zones: 所有出货口位置列表
        """
        self.logger = logging.getLogger("AGVProject.ODFlow")
        
        # 出货口列表
        self.unloading_zones = unloading_zones
        
        # 出货口占用状态 {position: task_id}
        self.unloading_zone_occupied: Dict[Tuple[int, int], int] = {}
        
        # 任务池
        self.task_pool: Dict[int, Task] = {}           # task_id -> Task
        self.pending_tasks: List[Task] = []             # 待分配任务
        self.active_tasks: Dict[int, Task] = {}         # 执行中任务 {task_id: Task}
        self.completed_tasks: List[Task] = []           # 已完成任务
        
        # 任务ID计数器
        self.next_task_id = 0
        
        self.logger.info(f"OD流程管理器初始化完成，{len(unloading_zones)}个出货口")
    
    def get_available_unloading_zones(self) -> List[Tuple[int, int]]:
        """
        获取当前空闲的出货口列表
        
        Returns:
            空闲出货口位置列表
        """
        available = []
        for pos in self.unloading_zones:
            if pos not in self.unloading_zone_occupied:
                available.append(pos)
        return available
    
    def occupy_unloading_zone(self, position: Tuple[int, int], task_id: int) -> bool:
        """
        占用一个出货口
        
        Args:
            position: 出货口位置
            task_id: 占用该出货口的任务ID
        
        Returns:
            是否成功占用
        """
        if position in self.unloading_zone_occupied:
            self.logger.warning(f"出货口 {position} 已被占用")
            return False
        
        self.unloading_zone_occupied[position] = task_id
        self.logger.debug(f"出货口 {position} 被任务 {task_id} 占用")
        return True
    
    def release_unloading_zone(self, position: Tuple[int, int]) -> bool:
        """
        释放一个出货口
        
        Args:
            position: 出货口位置
        
        Returns:
            是否成功释放
        """
        if position not in self.unloading_zone_occupied:
            self.logger.warning(f"出货口 {position} 未被占用，无法释放")
            return False
        
        task_id = self.unloading_zone_occupied.pop(position)
        self.logger.debug(f"出货口 {position} 已释放（任务 {task_id} 完成）")
        return True
    
    def create_task(self, pickup_pos: Tuple[int, int], 
                    delivery_pos: Tuple[int, int]) -> Optional[Task]:
        """
        创建一个新任务并加入待分配池
        
        Args:
            pickup_pos: 取货点（进货口）位置
            delivery_pos: 送货点（出货口）位置
        
        Returns:
            创建的任务对象，如果出货口已被占用则返回None
        """
        # 检查出货口是否可用
        if delivery_pos in self.unloading_zone_occupied:
            self.logger.warning(f"出货口 {delivery_pos} 已被占用，无法创建任务")
            return None
        
        # 创建任务
        task = Task(
            task_id=self.next_task_id,
            pickup_pos=pickup_pos,
            delivery_pos=delivery_pos,
            priority=1,
            status=TaskStatus.PENDING,
            create_time=0  # 由外部设置
        )
        self.next_task_id += 1
        
        # 加入任务池
        self.task_pool[task.task_id] = task
        self.pending_tasks.append(task)
        
        # 占用出货口
        self.occupy_unloading_zone(delivery_pos, task.task_id)
        
        self.logger.debug(f"创建任务 {task.task_id}: 取货 {pickup_pos} → 送货 {delivery_pos}")
        return task
    
    def assign_task(self, task_id: int, agv_id: int) -> bool:
        """
        分配任务给AGV
        
        Args:
            task_id: 任务ID
            agv_id: AGV ID
        
        Returns:
            是否成功分配
        """
        task = self.task_pool.get(task_id)
        if task is None:
            self.logger.warning(f"任务 {task_id} 不存在")
            return False
        
        if task.status != TaskStatus.PENDING:
            self.logger.warning(f"任务 {task_id} 状态为 {task.status}，无法分配")
            return False
        
        # 更新任务状态
        task.status = TaskStatus.ASSIGNED
        task.assigned_agv_id = agv_id
        
        # 从待分配池移到活跃池
        if task in self.pending_tasks:
            self.pending_tasks.remove(task)
        self.active_tasks[task_id] = task
        
        self.logger.info(f"任务 {task_id} 分配给 AGV {agv_id}")
        return True
    
    def update_task_status(self, task_id: int, new_status: TaskStatus) -> bool:
        """
        更新任务状态
        
        Args:
            task_id: 任务ID
            new_status: 新状态
        
        Returns:
            是否成功更新
        """
        task = self.task_pool.get(task_id)
        if task is None:
            self.logger.warning(f"任务 {task_id} 不存在")
            return False
        
        old_status = task.status
        task.status = new_status
        
        self.logger.debug(f"任务 {task_id}: {old_status.value} → {new_status.value}")
        
        # 如果任务完成，释放出货口
        if new_status == TaskStatus.COMPLETED:
            self.complete_task(task_id)
        
        return True
    
    def complete_task(self, task_id: int) -> bool:
        """
        完成任务，释放资源
        
        Args:
            task_id: 任务ID
        
        Returns:
            是否成功完成
        """
        task = self.task_pool.get(task_id)
        if task is None:
            return False
        
        # 释放出货口
        self.release_unloading_zone(task.delivery_pos)
        
        # 从活跃池移除
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        # 加入已完成列表
        task.status = TaskStatus.COMPLETED
        self.completed_tasks.append(task)
        
        self.logger.info(f"任务 {task_id} 已完成")
        return True
    
    def get_pending_tasks(self) -> List[Task]:
        """获取所有待分配任务"""
        return self.pending_tasks.copy()
    
    def get_active_tasks(self) -> Dict[int, Task]:
        """获取所有活跃任务"""
        return self.active_tasks.copy()
    
    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        """根据ID获取任务"""
        return self.task_pool.get(task_id)
    
    def get_statistics(self) -> Dict:
        """
        获取OD流程统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "total_tasks": len(self.task_pool),
            "pending": len(self.pending_tasks),
            "active": len(self.active_tasks),
            "completed": len(self.completed_tasks),
            "occupied_unloading_zones": len(self.unloading_zone_occupied),
            "available_unloading_zones": len(self.unloading_zones) - len(self.unloading_zone_occupied)
        }
    
    def reset(self):
        """重置OD流程管理器"""
        self.unloading_zone_occupied.clear()
        self.task_pool.clear()
        self.pending_tasks.clear()
        self.active_tasks.clear()
        self.completed_tasks.clear()
        self.next_task_id = 0
        self.logger.info("OD流程管理器已重置")
