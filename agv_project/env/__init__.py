"""
============================================
仓库环境模块
============================================
本模块由任务2团队负责开发。
包含：
- warehouse_env.py: 50×50仓库环境核心逻辑（进货口、出货口、障碍物管理）
- renderer.py: Pygame可视化渲染器

使用方式：
    # 独立运行测试环境逻辑
    python env/warehouse_env.py
    
    # 独立运行可视化演示
    python env/renderer.py

请继承 interface.communication.BaseModule 实现。
"""

from env.warehouse_env import WarehouseEnv, Obstacle, MAP_WIDTH, MAP_HEIGHT
from env.renderer import WarehouseRenderer

__all__ = [
    "WarehouseEnv",
    "Obstacle",
    "WarehouseRenderer",
    "MAP_WIDTH",
    "MAP_HEIGHT",
]
