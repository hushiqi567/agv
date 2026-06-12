"""15x15x5 状态编码器 + 全局特征向量"""
import numpy as np
import math
from typing import List, Tuple, Optional

class StateEncoder:
    """
    将环境信息编码为 RL 可用的状态表示。

    局部网格 15×15×5 通道:
      通道0: 静态障碍物 (货架/墙壁)
      通道1: 其他 AGV 位置 (动态实体)
      通道2: 移动障碍物 (随机移动的障碍)
      通道3: 目标方向热力图 (指向目标位置的梯度)
      通道4: 前方拥堵预警 (前后左右 2 步内的 AGV 密度)

    全局特征向量 (6维):
      [0]: 到目标曼哈顿距离 (归一化)
      [1]: 电量百分比
      [2]: 负载状态 (0空/1载)
      [3]: 任务优先级 (归一化)
      [4]: 到最近充电站距离 (归一化)
      [5]: 需要充电标志 (电量<=35%为1)
    """

    def __init__(self, grid_size: int = 15, num_channels: int = 5,
                 map_width: int = 50, map_height: int = 50):
        self.grid_size = grid_size
        self.num_channels = num_channels
        self.half = grid_size // 2
        self.map_width = map_width
        self.map_height = map_height

    def encode(self, agv_pos: Tuple[int, int], goal_pos: Tuple[int, int],
               grid: List[List[int]], obstacles: List[Tuple[int, int]],
               other_agvs: List[Tuple[int, int]],
               battery: float = 100.0, is_loaded: bool = False,
               priority: int = 1,
               charging_stations: List[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            local_grid: (5, 15, 15) float32
            global_vec: (6,) float32 — [dist_to_goal, battery%, is_loaded,
                                        priority, dist_to_cs, needs_charge]
        """
        ax, ay = agv_pos
        gx, gy = goal_pos
        local = np.zeros((self.num_channels, self.grid_size, self.grid_size), dtype=np.float32)

        obs_set = set(obstacles)
        agv_set = set(other_agvs)

        for dy in range(-self.half, self.half + 1):
            for dx in range(-self.half, self.half + 1):
                nx, ny = ax + dx, ay + dy
                lx, ly = dx + self.half, dy + self.half

                if nx < 0 or nx >= self.map_width or ny < 0 or ny >= self.map_height:
                    local[0, ly, lx] = 1.0
                    continue

                cell = grid[ny][nx]
                if cell == 1:
                    local[0, ly, lx] = 1.0
                if (nx, ny) in agv_set:
                    local[1, ly, lx] = 1.0
                if (nx, ny) in obs_set:
                    local[2, ly, lx] = 1.0

        # 通道3: 目标方向热力图
        dx_total = gx - ax
        dy_total = gy - ay
        dist = math.sqrt(dx_total**2 + dy_total**2) + 1e-6
        dir_x = dx_total / dist
        dir_y = dy_total / dist
        for dy in range(-self.half, self.half + 1):
            for dx in range(-self.half, self.half + 1):
                lx, ly = dx + self.half, dy + self.half
                if abs(dx) + abs(dy) == 0:
                    continue
                nd = math.sqrt(dx**2 + dy**2) + 1e-6
                dot = (dx/nd) * dir_x + (dy/nd) * dir_y
                local[3, ly, lx] = max(0.0, dot)

        # 通道4: 前方拥堵预警
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ax + dx, ay + dy
                if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                    if (nx, ny) in agv_set:
                        lx, ly = dx + self.half, dy + self.half
                        if 0 <= lx < self.grid_size and 0 <= ly < self.grid_size:
                            local[4, ly, lx] += 0.2

        # 全局特征向量 (6维)
        max_dist = self.map_width + self.map_height
        cs_list = charging_stations if charging_stations else []
        if cs_list:
            dist_to_cs = min(manhattan_distance(agv_pos, cs) for cs in cs_list)
        else:
            dist_to_cs = max_dist
        global_vec = np.array([
            min(manhattan_distance(agv_pos, goal_pos) / max_dist, 1.0),  # [0] 到目标距离
            battery / 100.0,                                               # [1] 电量百分比
            1.0 if is_loaded else 0.0,                                    # [2] 是否载货
            priority / 5.0,                                                # [3] 任务优先级
            min(dist_to_cs / max_dist, 1.0),                               # [4] 到最近充电站距离
            1.0 if battery <= 35.0 else 0.0,                              # [5] 需要充电标志
        ], dtype=np.float32)

        return local, global_vec

def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def encode_state(agv_pos, goal_pos, grid, obstacles, other_agvs,
                 battery=100.0, is_loaded=False, priority=1,
                 charging_stations=None,
                 grid_size=15, map_width=50, map_height=50):
    """便捷函数，使用默认参数编码状态"""
    encoder = StateEncoder(grid_size, map_width=map_width, map_height=map_height)
    return encoder.encode(agv_pos, goal_pos, grid, obstacles, other_agvs,
                          battery, is_loaded, priority, charging_stations)
