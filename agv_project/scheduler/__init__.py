"""
============================================
调度方案模块
============================================
本模块实现了AGV调度方案的核心逻辑，包括：
1. 任务生成：泊松分布生成货物到达
2. 出货口管理：占用和释放
3. 任务分配：最近距离优先算法
4. OD流程管理：任务完整生命周期

模块组成：
    - task_allocator.py: 任务分配算法（继承BaseModule）
    - od_flow.py: OD流程管理（任务池、出货口占用）

使用方式：
    from scheduler.task_allocator import TaskAllocator
    from scheduler.od_flow import ODFlowManager
"""

from scheduler.task_allocator import TaskAllocator, NUM_AGVS, POISSON_LAMBDA, AGV_INITIAL_POSITIONS
from scheduler.od_flow import ODFlowManager

__all__ = [
    "TaskAllocator",
    "ODFlowManager",
    "NUM_AGVS",
    "POISSON_LAMBDA",
    "AGV_INITIAL_POSITIONS",
]
