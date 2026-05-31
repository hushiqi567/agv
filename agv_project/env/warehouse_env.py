"""
============================================
仓库环境模块 - 核心逻辑
============================================
本模块实现了 50×50 的无人仓储环境，包含：
1. 网格地图：最左列和最右列为进货口（蓝色），中间两列各6个交错出货口（红色）
2. 障碍物管理：10个随机移动的障碍物，每步随机移动
3. 与消息总线集成，支持仿真循环

使用方式：
    python warehouse_env.py  # 独立运行测试
"""

import sys
import os
import random
import logging
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field

# 将项目根目录添加到系统路径（独立运行时使用）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from interface.communication import BaseModule, MessageType, Message
from interface.data_types import CellType


# ==========================================
# 常量定义
# ==========================================

# 地图尺寸
MAP_WIDTH = 50
MAP_HEIGHT = 50

# 单元格类型（使用 CellType 枚举的值）
CELL_EMPTY = CellType.EMPTY.value          # 0 - 空地
CELL_OBSTACLE = CellType.OBSTACLE.value    # 1 - 障碍物
CELL_LOADING = CellType.LOADING_ZONE.value # 2 - 进货口
CELL_UNLOADING = CellType.UNLOADING_ZONE.value # 3 - 出货口
CELL_CHARGING = CellType.CHARGING_STATION.value  # 4 - 充电站

# 颜色定义（RGB）
COLOR_EMPTY = (240, 240, 240)       # 浅灰 - 空地
COLOR_LOADING = (220, 80, 80)       # 红色 - 装货口（AGV在此取货）
COLOR_UNLOADING = (70, 130, 180)    # 蓝色 - 卸货口（AGV在此送货）
COLOR_OBSTACLE = (30, 30, 30)       # 黑色 - 障碍物
COLOR_CHARGING = (255, 215, 0)      # 金色 - 充电站
COLOR_GRID_LINE = (200, 200, 200)   # 灰色 - 网格线
COLOR_BACKGROUND = (255, 255, 255)  # 白色 - 背景

# 障碍物参数
NUM_OBSTACLES = 10

# 进货口和出货口数量
NUM_LOADING_ZONES = 20   # 最左列10个 + 最右列10个
NUM_UNLOADING_ZONES = 12 # 中间两列各6个


@dataclass
class Obstacle:
    """
    障碍物数据结构
    
    Attributes:
        position: 当前位置 (x, y)
        id: 障碍物唯一标识
    """
    position: Tuple[int, int]
    id: int = 0


class WarehouseEnv(BaseModule):
    """
    仓库环境类
    
    管理 50×50 的仓库网格环境，包括：
    - 地图布局（进货口、出货口）
    - 障碍物生成、移动
    - 状态更新和消息发布
    
    Usage:
        >>> env = WarehouseEnv()
        >>> env.reset()
        >>> for _ in range(100):
        ...     env.step()
    """
    
    def __init__(self):
        """初始化仓库环境"""
        super().__init__("env")
        
        # 地图尺寸
        self.width = MAP_WIDTH
        self.height = MAP_HEIGHT
        
        # 网格地图: 0=空地, 1=障碍物, 2=进货口, 3=出货口
        self.grid: List[List[int]] = []
        
        # 进货口位置列表（最左列和最右列各10个）
        self.loading_zones: List[Tuple[int, int]] = []
        
        # 出货口位置列表（中间两列各6个交错排列）
        self.unloading_zones: List[Tuple[int, int]] = []

        # 充电站位置列表（4个内部点位）
        self.charging_stations: List[Tuple[int, int]] = []

        # 障碍物管理
        self.obstacles: List[Obstacle] = []
        self.next_obstacle_id = 0
        
        # 仿真状态
        self.current_step = 0
        self.is_running = False
        
        # 随机数生成器
        self.rng = random.Random(42)
        
        self.logger.info(f"仓库环境初始化完成: {self.width}x{self.height}")
    
    def _setup_subscriptions(self):
        """设置消息订阅"""
        self.subscribe(MessageType.MAIN_SIMULATION_START, self._on_simulation_start)
        self.subscribe(MessageType.MAIN_SIMULATION_STOP, self._on_simulation_stop)
        self.subscribe(MessageType.MAIN_SIMULATION_RESET, self._on_simulation_reset)
    
    def _build_map(self):
        """
        构建 50×50 仓库地图
        
        布局规则：
        - 最左列 (x=0): 10个进货口（蓝色），交错排列
        - 最右列 (x=49): 10个进货口（蓝色），交错排列
        - 中间两列 (x=24, x=25): 各6个交错出货口（红色）
        - 其余: 空地（灰色）
        """
        self.grid = [[CELL_EMPTY] * self.width for _ in range(self.height)]
        self.loading_zones.clear()
        self.unloading_zones.clear()
        
        # 最左列 (x=0) - 10个进货口，交错排列（y=2, 7, 12, 17, 22, 27, 32, 37, 42, 47）
        left_loading_ys = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
        for y in left_loading_ys:
            self.grid[y][0] = CELL_LOADING
            self.loading_zones.append((0, y))
        
        # 最右列 (x=49) - 10个进货口，交错排列（y=2, 7, 12, 17, 22, 27, 32, 37, 42, 47）
        right_loading_ys = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
        for y in right_loading_ys:
            self.grid[y][self.width - 1] = CELL_LOADING
            self.loading_zones.append((self.width - 1, y))
        
        # 中间两列出货口 (x=24, x=25)
        # 各6个，交错排列
        # 左列 (x=24): y=4, 12, 20, 28, 36, 44
        # 右列 (x=25): y=8, 16, 24, 32, 40, 48
        left_col = self.width // 2 - 1   # x=24
        right_col = self.width // 2      # x=25
        
        left_unloading_ys = [4, 12, 20, 28, 36, 44]
        right_unloading_ys = [8, 16, 24, 32, 40, 48]
        
        for y in left_unloading_ys:
            self.grid[y][left_col] = CELL_UNLOADING
            self.unloading_zones.append((left_col, y))
        
        for y in right_unloading_ys:
            self.grid[y][right_col] = CELL_UNLOADING
            self.unloading_zones.append((right_col, y))

        # 充电站 (4个，分布在内部区域，避开进货口/出货口列)
        charging_positions = [(12, 12), (12, 37), (37, 12), (37, 37)]
        for cx, cy in charging_positions:
            self.grid[cy][cx] = CELL_CHARGING
            self.charging_stations.append((cx, cy))

        self.logger.info(
            f"地图构建完成: "
            f"{len(self.loading_zones)}个进货口, "
            f"{len(self.unloading_zones)}个出货口, "
            f"{len(self.charging_stations)}个充电站"
        )
    
    def reset(self):
        """
        重置环境到初始状态
        
        清空障碍物，重置步数，重新构建地图，初始化10个障碍物。
        """
        self.current_step = 0
        self.obstacles.clear()
        self.next_obstacle_id = 0
        self._build_map()
        
        # 初始化10个障碍物（只在重置时生成一次，永久存在）
        self._init_obstacles()
        
        self.logger.info("环境已重置")
        
        # 发布环境状态更新
        self._publish_state()
    
    def step(self):
        """
        执行一步环境更新

        流程：
        1. 步数递增
        2. 障碍物每3步随机移动一格（避开进货口、出货口和其他障碍物）
        3. 发布状态更新
        """
        self.current_step += 1

        # 障碍物每3步移动一次，降低对AGV的干扰
        if self.current_step % 3 == 0:
            self._move_obstacles()

        # 发布状态更新
        self._publish_state()

        return self._get_observation()
    
    def _move_obstacles(self):
        """
        移动所有障碍物
        
        每个障碍物随机移动到相邻的空格（上、下、左、右）。
        障碍物不能进入进货口和出货口的位置。
        障碍物永久存在，不会消失。
        """
        # 获取当前所有障碍物占据的位置
        occupied = set(o.position for o in self.obstacles)
        forbidden = self._get_forbidden_positions()
        
        for obstacle in self.obstacles:
            # 尝试移动障碍物
            old_pos = obstacle.position
            new_pos = self._get_random_adjacent_empty(old_pos, occupied, forbidden)
            if new_pos:
                # 从原位置移除
                old_x, old_y = old_pos
                if self.grid[old_y][old_x] == CELL_OBSTACLE:
                    self.grid[old_y][old_x] = CELL_EMPTY
                
                # 放置到新位置
                new_x, new_y = new_pos
                self.grid[new_y][new_x] = CELL_OBSTACLE
                obstacle.position = new_pos
                
                # 更新占用集合
                occupied.remove(old_pos)
                occupied.add(new_pos)
    
    def _get_random_adjacent_empty(self, pos: Tuple[int, int], 
                                    occupied: Set[Tuple[int, int]],
                                    forbidden: Optional[Set[Tuple[int, int]]] = None) -> Optional[Tuple[int, int]]:
        """
        获取相邻的空格位置（四连通）
        
        Args:
            pos: 当前位置
            occupied: 已被占用的位置集合
            forbidden: 禁止进入的位置集合（进货口、出货口等）
        
        Returns:
            随机选择的相邻空格，如果没有则返回None
        """
        if forbidden is None:
            forbidden = set()
        
        x, y = pos
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上、下、左、右
        self.rng.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # 检查边界
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                continue
            
            # 检查是否为空地、未被占用、且不在禁止区域
            if ((nx, ny) not in occupied 
                and (nx, ny) not in forbidden
                and self.grid[ny][nx] == CELL_EMPTY):
                return (nx, ny)
        
        return None
    
    def _get_forbidden_positions(self) -> Set[Tuple[int, int]]:
        """
        获取障碍物禁止进入的位置集合
        
        障碍物不能出现在：
        - 进货口（蓝色，最左列和最右列）
        - 出货口（红色，中间两列）
        
        Returns:
            禁止位置的集合
        """
        forbidden = set()
        forbidden.update(self.loading_zones)
        forbidden.update(self.unloading_zones)
        forbidden.update(self.charging_stations)
        return forbidden

    def _init_obstacles(self):
        """
        初始化10个障碍物
        
        在重置环境时调用一次，生成10个障碍物随机分布在空地上。
        障碍物不能出现在进货口和出货口的位置。
        障碍物永久存在，不会消失。
        """
        forbidden = self._get_forbidden_positions()
        occupied = set()
        
        while len(self.obstacles) < NUM_OBSTACLES:
            # 获取所有可放置障碍物的空地位置
            empty_cells = []
            
            for y in range(self.height):
                for x in range(self.width):
                    if (self.grid[y][x] == CELL_EMPTY 
                        and (x, y) not in occupied
                        and (x, y) not in forbidden):
                        empty_cells.append((x, y))
            
            if not empty_cells:
                break
            
            # 随机选择一个空地放置障碍物
            pos = self.rng.choice(empty_cells)
            self.grid[pos[1]][pos[0]] = CELL_OBSTACLE
            occupied.add(pos)
            
            obstacle = Obstacle(
                position=pos,
                id=self.next_obstacle_id
            )
            self.next_obstacle_id += 1
            self.obstacles.append(obstacle)
            
            self.logger.debug(f"生成障碍物 {obstacle.id} 在位置 {pos}")
    
    def _get_observation(self) -> Dict:
        """
        获取当前环境观测
        
        Returns:
            包含完整环境状态的字典
        """
        return {
            "step": self.current_step,
            "grid": [row[:] for row in self.grid],
            "width": self.width,
            "height": self.height,
            "loading_zones": self.loading_zones.copy(),
            "unloading_zones": self.unloading_zones.copy(),
            "obstacles": [
                {"id": o.id, "position": o.position}
                for o in self.obstacles
            ],
            "num_obstacles": len(self.obstacles)
        }
    
    def _publish_state(self):
        """发布环境状态更新到消息总线"""
        obs = self._get_observation()
        self.publish(MessageType.ENV_STATE_UPDATE, obs)
    
    def _on_simulation_start(self, message: Message):
        """处理仿真开始消息"""
        self.is_running = True
        self.reset()
        self.logger.info("仿真开始，环境已初始化")
    
    def _on_simulation_stop(self, message: Message):
        """处理仿真停止消息"""
        self.is_running = False
        self.logger.info(f"仿真停止，共运行 {self.current_step} 步")
    
    def _on_simulation_reset(self, message: Message):
        """处理仿真重置消息"""
        self.reset()
    
    def get_grid_for_render(self) -> List[List[int]]:
        """
        获取用于渲染的网格数据
        
        Returns:
            二维网格数据，每个元素为单元格类型
        """
        return [row[:] for row in self.grid]
    
    def get_obstacle_info(self) -> List[Dict]:
        """
        获取障碍物信息用于渲染
        
        Returns:
            障碍物信息列表
        """
        return [
            {"id": o.id, "position": o.position}
            for o in self.obstacles
        ]


# ==========================================
# 独立运行测试
# ==========================================

def run_test():
    """
    独立运行测试（不依赖Pygame）
    
    测试环境的核心逻辑：地图构建、障碍物生成和移动。
    """
    from interface.config import get_config
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("仓库环境模块测试")
    print("=" * 60)
    
    env = WarehouseEnv()
    env.reset()
    
    # 验证地图
    print(f"\n地图尺寸: {env.width}x{env.height}")
    print(f"进货口数量: {len(env.loading_zones)}")
    print(f"出货口数量: {len(env.unloading_zones)}")
    
    # 验证进货口位置
    left_loading = sum(1 for x, y in env.loading_zones if x == 0)
    right_loading = sum(1 for x, y in env.loading_zones if x == env.width - 1)
    print(f"左列进货口: {left_loading}, 右列进货口: {right_loading}")
    
    # 验证出货口位置
    left_unloading = sum(1 for x, y in env.unloading_zones if x == env.width // 2 - 1)
    right_unloading = sum(1 for x, y in env.unloading_zones if x == env.width // 2)
    print(f"左列出货口: {left_unloading}, 右列出货口: {right_unloading}")
    
    # 运行几步测试障碍物
    print(f"\n运行20步测试障碍物...")
    for step in range(20):
        env.step()
        obs_count = len(env.obstacles)
        if step < 5 or step % 5 == 0:
            print(f"  步数 {step + 1}: {obs_count} 个障碍物")
    
    print(f"\n最终障碍物数量: {len(env.obstacles)}")
    print("测试完成！")


if __name__ == "__main__":
    run_test()
