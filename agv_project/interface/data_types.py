"""
============================================
数据结构和类型定义模块
============================================
本模块定义了整个项目中所有模块之间共享的数据结构，
包括任务、AGV状态、地图信息等。
使用 dataclass 确保数据结构的一致性和类型安全。
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum


# -- 枚举类型定义 --

class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"        # 等待分配
    ASSIGNED = "assigned"      # 已分配但未开始
    MOVING_TO_PICKUP = "moving_to_pickup"  # 前往取货点
    LOADING = "loading"        # 正在装货
    MOVING_TO_DELIVERY = "moving_to_delivery"  # 前往送货点
    UNLOADING = "unloading"    # 正在卸货
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 失败


class AGVStatus(Enum):
    """AGV状态枚举"""
    IDLE = "idle"              # 空闲等待
    MOVING = "moving"          # 移动中（通用）
    MOVING_TO_PICKUP = "moving_to_pickup"  # 前往取货点
    MOVING_TO_DELIVERY = "moving_to_delivery"  # 前往送货点
    LOADING = "loading"        # 装货中
    UNLOADING = "unloading"    # 卸货中
    MOVING_TO_CHARGE = "moving_to_charge"  # 前往充电站
    CHARGING = "charging"      # 充电中
    BLOCKED = "blocked"        # 被阻塞（等待其他AGV）
    ERROR = "error"            # 故障状态


class CellType(Enum):
    """地图单元格类型枚举"""
    EMPTY = 0                  # 空地（可通行）
    OBSTACLE = 1               # 障碍物/货架（不可通行）
    LOADING_ZONE = 2           # 装货区
    UNLOADING_ZONE = 3         # 卸货区
    CHARGING_STATION = 4       # 充电站
    AGV = 5                    # AGV占用


class Direction(Enum):
    """移动方向枚举"""
    UP = (0, -1)               # 上
    DOWN = (0, 1)              # 下
    LEFT = (-1, 0)             # 左
    RIGHT = (1, 0)             # 右
    STAY = (0, 0)              # 原地等待


# -- 核心数据结构 --

@dataclass
class Task:
    """
    任务数据结构
    
    表示一个运输任务：从取货点将货物运送到送货点。
    
    Attributes:
        task_id: 任务唯一标识符
        pickup_pos: 取货点坐标 (x, y)
        delivery_pos: 送货点坐标 (x, y)
        priority: 任务优先级（1-5，数字越大优先级越高）
        status: 任务当前状态
        assigned_agv_id: 被分配的AGV ID（未分配时为None）
        create_time: 任务创建时间步
        deadline: 任务截止时间步（可选）
        cargo_weight: 货物重量（可选，影响AGV能耗）
    """
    task_id: int
    pickup_pos: Tuple[int, int]
    delivery_pos: Tuple[int, int]
    priority: int = 1
    status: TaskStatus = TaskStatus.PENDING
    assigned_agv_id: Optional[int] = None
    create_time: int = 0
    deadline: Optional[int] = None
    cargo_weight: float = 1.0


@dataclass
class AGVState:
    """
    AGV状态数据结构
    
    记录AGV的完整状态信息，用于调度和路径规划。
    
    Attributes:
        agv_id: AGV唯一标识符
        position: 当前位置坐标 (x, y)
        status: AGV当前状态
        battery: 当前电量百分比 (0-100)
        is_loaded: 是否载有货物
        current_task: 当前执行的任务（空闲时为None）
        task_queue: 待执行任务队列
        planned_path: 当前规划的路径（坐标列表）
        speed: AGV移动速度（格/时间步）
    """
    agv_id: int
    position: Tuple[int, int]
    status: AGVStatus = AGVStatus.IDLE
    battery: float = 100.0
    is_loaded: bool = False
    current_task: Optional[Task] = None
    task_queue: List[Task] = field(default_factory=list)
    planned_path: List[Tuple[int, int]] = field(default_factory=list)
    speed: float = 1.0


@dataclass
class MapConfig:
    """
    地图配置数据结构
    
    定义仓库地图的尺寸和布局。
    
    Attributes:
        width: 地图宽度（列数）
        height: 地图高度（行数）
        grid: 二维数组，每个元素为CellType枚举值
        loading_zones: 装货区坐标列表
        unloading_zones: 卸货区坐标列表
        charging_stations: 充电站坐标列表
        obstacle_positions: 障碍物坐标列表
    """
    width: int
    height: int
    grid: List[List[int]]
    loading_zones: List[Tuple[int, int]] = field(default_factory=list)
    unloading_zones: List[Tuple[int, int]] = field(default_factory=list)
    charging_stations: List[Tuple[int, int]] = field(default_factory=list)
    obstacle_positions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class ScheduleResult:
    """
    调度结果数据结构
    
    调度模块的输出，包含任务分配和OD流程信息。
    
    Attributes:
        task_assignments: 任务分配映射 {task_id: agv_id}
        agv_routes: 每个AGV的OD路线 {agv_id: [(from, to), ...]}
        estimated_times: 预计完成时间 {task_id: time_steps}
        schedule_metrics: 调度性能指标
    """
    task_assignments: Dict[int, int]  # task_id -> agv_id
    agv_routes: Dict[int, List[Tuple[Tuple[int, int], Tuple[int, int]]]]
    estimated_times: Dict[int, int]
    schedule_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PathPlanResult:
    """
    路径规划结果数据结构
    
    路径规划模块的输出，包含所有AGV的无冲突路径。
    
    Attributes:
        paths: 每个AGV的路径 {agv_id: [(x,y), ...]}
        conflicts_resolved: 解决的冲突数量
        total_steps: 总规划时间步数
        computation_time: 规划耗时（秒）
        path_metrics: 路径性能指标
    """
    paths: Dict[int, List[Tuple[int, int]]]
    conflicts_resolved: int = 0
    total_steps: int = 0
    computation_time: float = 0.0
    path_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """
    仿真配置数据结构
    
    控制仿真运行的全局参数。
    
    Attributes:
        max_steps: 最大仿真步数
        num_agvs: AGV数量
        task_generation_interval: 任务生成间隔（步数）
        max_concurrent_tasks: 最大并发任务数
        render_mode: 渲染模式（"human", "rgb_array", None）
        log_level: 日志级别
        seed: 随机种子
    """
    max_steps: int = 1000
    num_agvs: int = 4
    task_generation_interval: int = 10
    max_concurrent_tasks: int = 20
    render_mode: Optional[str] = "human"
    log_level: str = "INFO"
    seed: int = 42


# -- 辅助函数 --

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    计算曼哈顿距离
    
    Args:
        pos1: 第一个位置坐标
        pos2: 第二个位置坐标
    
    Returns:
        两个位置之间的曼哈顿距离
    
    Example:
        >>> manhattan_distance((0, 0), (3, 4))
        7
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def is_adjacent(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
    """
    判断两个位置是否相邻（四连通）
    
    Args:
        pos1: 第一个位置坐标
        pos2: 第二个位置坐标
    
    Returns:
        是否相邻
    """
    return manhattan_distance(pos1, pos2) == 1


def get_direction(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Direction:
    """
    获取从from_pos到to_pos的移动方向
    
    Args:
        from_pos: 起始位置
        to_pos: 目标位置
    
    Returns:
        移动方向枚举值
    
    Raises:
        ValueError: 如果两个位置不相邻
    """
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    if (dx, dy) == (0, -1):
        return Direction.UP
    elif (dx, dy) == (0, 1):
        return Direction.DOWN
    elif (dx, dy) == (-1, 0):
        return Direction.LEFT
    elif (dx, dy) == (1, 0):
        return Direction.RIGHT
    elif (dx, dy) == (0, 0):
        return Direction.STAY
    else:
        raise ValueError(f"位置 {from_pos} 和 {to_pos} 不相邻")
