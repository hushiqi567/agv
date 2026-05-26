"""
============================================
MAPF全局规划模块（CBS算法）
============================================
本模块实现了 Conflict-Based Search (CBS) 算法，
为所有AGV规划无冲突的全局路径。

核心功能：
1. A*单AGV寻路（考虑静态障碍物）
2. 冲突检测（顶点冲突、边冲突）
3. CBS冲突解决（约束树搜索）

使用方式：
    from path_planning.mapf_planner import MAPFPlanner
    planner = MAPFPlanner(grid, width, height)
    paths = planner.solve(agents)  # agents: [(start, goal), ...]
"""

import sys
import os
import heapq
import logging
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field

# 将项目根目录添加到系统路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from interface.data_types import manhattan_distance, CellType


# ==========================================
# 数据结构
# ==========================================

@dataclass
class Conflict:
    """
    冲突数据结构
    
    Attributes:
        time: 冲突发生的时间步
        agv1: 第一个AGV的ID
        agv2: 第二个AGV的ID
        pos1: AGV1在time时的位置
        pos2: AGV2在time时的位置
        conflict_type: 'vertex'（顶点冲突）或 'edge'（边冲突）
    """
    time: int
    agv1: int
    agv2: int
    pos1: Tuple[int, int]
    pos2: Tuple[int, int]
    conflict_type: str = 'vertex'


@dataclass
class CBSNode:
    """
    CBS约束树节点
    
    Attributes:
        constraints: 约束列表 [(agv_id, time, position, constraint_type)]
        solution: 当前解 {agv_id: [(x,y), ...]}
        cost: 总代价（所有路径长度之和）
        depth: 节点深度
    """
    constraints: List[Tuple[int, int, Tuple[int, int], str]] = field(default_factory=list)
    solution: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    cost: int = 0
    depth: int = 0
    
    def __lt__(self, other):
        return self.cost < other.cost


# ==========================================
# A* 单AGV寻路
# ==========================================

@dataclass
class AStarNode:
    """A*搜索节点"""
    position: Tuple[int, int]
    g: int = 0
    h: int = 0
    f: int = 0
    parent: Optional['AStarNode'] = None
    time: int = 0
    
    def __lt__(self, other):
        return self.f < other.f


def a_star_search(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    grid: List[List[int]],
    width: int,
    height: int,
    constraints: Optional[Set[Tuple[int, int, int]]] = None,
    occupied_positions: Optional[Set[Tuple[int, int]]] = None,
    max_steps: int = 500
) -> Optional[List[Tuple[int, int]]]:
    """
    A*寻路算法（考虑时间维度的约束）
    
    Args:
        start: 起点 (x, y)
        goal: 终点 (x, y)
        grid: 网格地图
        width: 地图宽度
        height: 地图高度
        constraints: 时空约束 {(x, y, time), ...}，禁止在time时刻位于(x,y)
        occupied_positions: 静态占用位置（其他AGV当前位置等）
        max_steps: 最大搜索步数
    
    Returns:
        路径列表 [(x,y), ...]，如果无解则返回None
    """
    if constraints is None:
        constraints = set()
    if occupied_positions is None:
        occupied_positions = set()
    
    # 检查起点和终点是否有效
    sx, sy = start
    gx, gy = goal
    
    if grid[sy][sx] == 1:  # 障碍物
        return None
    
    # 起点和终点相同，直接返回
    if start == goal:
        return [start]
    
    # 方向：上、下、左、右、等待
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]
    
    # 开放列表和关闭列表
    open_set = []
    closed_set = set()
    
    # 起点节点
    h = manhattan_distance(start, goal)
    start_node = AStarNode(
        position=start,
        g=0,
        h=h,
        f=h,
        time=0
    )
    heapq.heappush(open_set, (start_node.f, id(start_node), start_node))
    
    while open_set and len(closed_set) < max_steps:
        _, _, current = heapq.heappop(open_set)
        
        state_key = (current.position[0], current.position[1], current.time)
        if state_key in closed_set:
            continue
        closed_set.add(state_key)
        
        # 到达目标
        if current.position == goal:
            # 构建路径
            path = []
            node = current
            while node:
                path.append(node.position)
                node = node.parent
            path.reverse()
            return path
        
        # 扩展邻居
        for dx, dy in directions:
            nx, ny = current.position[0] + dx, current.position[1] + dy
            
            # 边界检查
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            
            # 障碍物检查（等待动作不受障碍物限制）
            if (dx, dy) != (0, 0) and grid[ny][nx] == 1:
                continue
            
            # 装货口/卸货口检查：AGV不能经过（除非是起点或终点）
            if (dx, dy) != (0, 0) and grid[ny][nx] in (CellType.LOADING_ZONE.value, CellType.UNLOADING_ZONE.value):
                # 允许起点和终点在装货口/卸货口
                if (nx, ny) != start and (nx, ny) != goal:
                    continue
            
            # 静态占用检查
            if (dx, dy) != (0, 0) and (nx, ny) in occupied_positions:
                continue
            
            next_time = current.time + 1
            
            # 时空约束检查
            if (nx, ny, next_time) in constraints:
                continue
            
            # 边冲突检查（交换位置）
            if (dx, dy) != (0, 0):
                # 检查是否与另一个AGV交换位置
                if (current.position[0], current.position[1], next_time) in constraints:
                    if (nx, ny, current.time) in constraints:
                        continue
            
            next_state_key = (nx, ny, next_time)
            if next_state_key in closed_set:
                continue
            
            h = manhattan_distance((nx, ny), goal)
            neighbor = AStarNode(
                position=(nx, ny),
                g=current.g + 1,
                h=h,
                f=current.g + 1 + h,
                parent=current,
                time=next_time
            )
            heapq.heappush(open_set, (neighbor.f, id(neighbor), neighbor))
    
    return None  # 无解


# ==========================================
# CBS 多AGV路径规划
# ==========================================

class MAPFPlanner:
    """
    MAPF全局规划器（CBS算法）
    
    为所有AGV规划无冲突的全局路径。
    每步调用一次，动态更新路径。
    
    Usage:
        >>> planner = MAPFPlanner(grid, 50, 50)
        >>> agents = [(0, (2,2), (24,4)), (1, (47,2), (25,8))]
        >>> paths = planner.solve(agents)
    """
    
    def __init__(self, grid: List[List[int]], width: int, height: int):
        """
        初始化MAPF规划器
        
        Args:
            grid: 网格地图（0=空地, 1=障碍物）
            width: 地图宽度
            height: 地图高度
        """
        self.logger = logging.getLogger("AGVProject.MAPF")
        self.grid = grid
        self.width = width
        self.height = height
        self.max_iterations = 100  # CBS最大迭代次数
        self.logger.info(f"MAPF规划器初始化完成: {width}x{height}")
    
    def update_grid(self, grid: List[List[int]]):
        """更新网格地图"""
        self.grid = grid
    
    def solve(self, agents: List[Tuple[int, Tuple[int, int], Tuple[int, int]]],
              occupied_positions: Optional[Set[Tuple[int, int]]] = None) -> Dict[int, List[Tuple[int, int]]]:
        """
        为所有AGV规划无冲突路径
        
        Args:
            agents: AGV列表 [(agv_id, start_pos, goal_pos), ...]
            occupied_positions: 静态占用位置集合
        
        Returns:
            {agv_id: [(x,y), ...]} 路径字典
        """
        if occupied_positions is None:
            occupied_positions = set()
        
        # 将agents转换为字典以便通过agv_id查找
        agents_dict = {agv_id: (start, goal) for agv_id, start, goal in agents}
        
        # 步骤1：为每个AGV单独规划路径（不考虑冲突）
        initial_solution = {}
        for agv_id, start, goal in agents:
            path = a_star_search(
                start, goal, self.grid, self.width, self.height,
                occupied_positions=occupied_positions
            )
            if path is None:
                self.logger.warning(f"AGV {agv_id}: 无法找到路径 {start} → {goal}")
                # 返回等待路径
                path = [start] * 10
            initial_solution[agv_id] = path
        
        # 步骤2：CBS解决冲突
        root = CBSNode(solution=initial_solution)
        root.cost = sum(len(p) for p in root.solution.values())
        
        # 检测初始冲突
        conflicts = self._detect_conflicts(root.solution)
        
        if not conflicts:
            self.logger.debug("初始路径无冲突")
            return root.solution
        
        # CBS主循环
        open_list = [root]
        best_solution = root.solution
        
        for iteration in range(self.max_iterations):
            if not open_list:
                break
            
            node = heapq.heappop(open_list)
            
            # 检测冲突
            conflicts = self._detect_conflicts(node.solution)
            
            if not conflicts:
                self.logger.debug(f"CBS找到无冲突解，迭代次数: {iteration}")
                return node.solution
            
            # 处理第一个冲突
            conflict = conflicts[0]
            self.logger.debug(f"冲突: AGV{conflict.agv1} vs AGV{conflict.agv2} "
                            f"在时间 {conflict.time} 位置 {conflict.pos1}")
            
            # 为两个AGV分别添加约束
            for agv_id in [conflict.agv1, conflict.agv2]:
                # 创建子节点
                child = CBSNode(
                    constraints=node.constraints.copy(),
                    solution=node.solution.copy(),
                    depth=node.depth + 1
                )
                
                # 添加约束
                if conflict.conflict_type == 'vertex':
                    # 顶点冲突：禁止在time时刻到达pos
                    child.constraints.append(
                        (agv_id, conflict.time, conflict.pos1, 'vertex')
                    )
                else:
                    # 边冲突：禁止在time时刻从pos1移动到pos2
                    child.constraints.append(
                        (agv_id, conflict.time, conflict.pos1, 'edge')
                    )
                
                # 重新规划该AGV的路径
                agv_start, agv_goal = agents_dict[agv_id]
                
                # 提取该AGV的约束
                agv_constraints = set()
                for c_agv_id, c_time, c_pos, c_type in child.constraints:
                    if c_agv_id == agv_id:
                        agv_constraints.add((c_pos[0], c_pos[1], c_time))
                
                new_path = a_star_search(
                    agv_start, agv_goal, self.grid, self.width, self.height,
                    constraints=agv_constraints,
                    occupied_positions=occupied_positions
                )
                
                if new_path is not None:
                    child.solution[agv_id] = new_path
                    child.cost = sum(len(p) for p in child.solution.values())
                    heapq.heappush(open_list, child)
            
            # 更新最佳解
            if node.cost < sum(len(p) for p in best_solution.values()):
                best_solution = node.solution
        
        self.logger.warning(f"CBS达到最大迭代次数 {self.max_iterations}，返回最佳解")
        return best_solution
    
    def _detect_conflicts(self, solution: Dict[int, List[Tuple[int, int]]]) -> List[Conflict]:
        """
        检测路径中的冲突
        
        Args:
            solution: 路径字典 {agv_id: [(x,y), ...]}
        
        Returns:
            冲突列表
        """
        conflicts = []
        max_len = max(len(path) for path in solution.values())
        
        for t in range(max_len):
            # 检查顶点冲突
            pos_at_time = {}
            for agv_id, path in solution.items():
                if t < len(path):
                    pos = path[t]
                    if pos in pos_at_time:
                        other_agv = pos_at_time[pos]
                        conflicts.append(Conflict(
                            time=t,
                            agv1=other_agv,
                            agv2=agv_id,
                            pos1=pos,
                            pos2=pos,
                            conflict_type='vertex'
                        ))
                    else:
                        pos_at_time[pos] = agv_id
            
            # 检查边冲突（交换位置）
            if t > 0:
                for agv1_id, path1 in solution.items():
                    if t >= len(path1):
                        continue
                    for agv2_id, path2 in solution.items():
                        if agv2_id <= agv1_id or t >= len(path2):
                            continue
                        # AGV1从pos_a到pos_b，AGV2从pos_b到pos_a
                        if (path1[t-1] == path2[t] and path1[t] == path2[t-1]):
                            conflicts.append(Conflict(
                                time=t,
                                agv1=agv1_id,
                                agv2=agv2_id,
                                pos1=path1[t],
                                pos2=path2[t],
                                conflict_type='edge'
                            ))
        
        return conflicts
    
    def plan_path(self, agv_id: int, start: Tuple[int, int], 
                  goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        为单个AGV规划路径（不考虑其他AGV）
        
        Args:
            agv_id: AGV ID
            start: 起点
            goal: 终点
        
        Returns:
            路径列表
        """
        return a_star_search(
            start, goal, self.grid, self.width, self.height
        )


# ==========================================
# 独立运行测试
# ==========================================

def run_test():
    """测试MAPF规划器"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("MAPF规划器测试")
    print("=" * 60)
    
    # 创建简单地图
    width, height = 10, 10
    grid = [[0] * width for _ in range(height)]
    
    # 添加一些障碍物
    grid[3][3] = 1
    grid[3][4] = 1
    grid[3][5] = 1
    grid[6][6] = 1
    grid[6][7] = 1
    
    planner = MAPFPlanner(grid, width, height)
    
    # 测试单AGV寻路
    print("\n测试单AGV寻路:")
    path = planner.plan_path(0, (0, 0), (9, 9))
    if path:
        print(f"  路径长度: {len(path)}")
        print(f"  路径: {path[:5]}...{path[-3:]}")
    
    # 测试多AGV
    print("\n测试多AGV路径规划:")
    agents = [
        (0, (0, 0), (9, 9)),
        (1, (9, 0), (0, 9)),
        (2, (0, 9), (9, 0)),
    ]
    
    solution = planner.solve(agents)
    
    for agv_id, path in solution.items():
        print(f"  AGV {agv_id}: 路径长度 {len(path)}")
    
    # 检测冲突
    conflicts = planner._detect_conflicts(solution)
    print(f"\n冲突数量: {len(conflicts)}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    run_test()
