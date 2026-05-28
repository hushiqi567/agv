"""CBS基线 — 封装MAPFPlanner为实验对照组"""
from path_planning.mapf_planner import MAPFPlanner


class CBSBaseline:
    """CBS对照组 — 纯传统搜索方法"""
    def __init__(self, grid, width, height):
        self.planner = MAPFPlanner(grid, width, height)

    def plan(self, agents):
        """agents: [(agv_id, start_pos, goal_pos), ...]"""
        return self.planner.solve(agents)

    def update_grid(self, grid):
        self.planner.update_grid(grid)
