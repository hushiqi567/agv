"""
============================================
接口定义模块
============================================
本模块定义了整个项目的接口规范和数据结构。
包含三个核心子模块：
- data_types.py: 数据结构与类型定义
- config.py: 全局配置管理
- communication.py: 模块间通信接口

所有模块都应通过本模块导入接口定义。
"""

from .data_types import (
    Task, AGVState, MapConfig,
    ScheduleResult, PathPlanResult, SimulationConfig,
    TaskStatus, AGVStatus, CellType, Direction,
    manhattan_distance, is_adjacent, get_direction
)

from .config import (
    get_config, ConfigManager,
    MapConfig as MapConfigData,
    AGVConfig, SimulationConfig as SimConfigData,
    RLConfig, RewardConfig
)

from .communication import (
    get_message_bus, MessageBus,
    BaseModule, Message, MessageType
)

__all__ = [
    # 数据类型
    'Task', 'AGVState', 'MapConfig',
    'ScheduleResult', 'PathPlanResult', 'SimulationConfig',
    'TaskStatus', 'AGVStatus', 'CellType', 'Direction',
    'manhattan_distance', 'is_adjacent', 'get_direction',
    
    # 配置
    'get_config', 'ConfigManager',
    'MapConfigData', 'AGVConfig', 'SimConfigData',
    'RLConfig', 'RewardConfig',
    
    # 通信
    'get_message_bus', 'MessageBus',
    'BaseModule', 'Message', 'MessageType',
]
