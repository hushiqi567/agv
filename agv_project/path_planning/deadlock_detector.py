"""死锁检测与恢复模块 — 每10步扫描有向图判环"""
import logging
import random
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict


class DeadlockDetector:
    """
    死锁检测器。

    检测机制:
      每 detection_interval 步构建有向图:
        - 节点 = AGV
        - 有向边 A→B = AGV A 的下一步目标位置被 AGV B 占据 (A 在等待 B)
      用 DFS 检测环，若存在环则判定为死锁。

    恢复策略:
      选参与死锁中负载最轻的 AGV，随机回退一步打破循环。
    """

    def __init__(self, detection_interval: int = 10, max_wait_steps: int = 20,
                 recovery_steps: int = 3):
        self.detection_interval = detection_interval
        self.max_wait_steps = max_wait_steps
        self.recovery_steps = recovery_steps
        self.logger = logging.getLogger("AGVProject.DeadlockDetector")
        self.step_counter = 0
        self.deadlock_count = 0
        self.recovery_count = 0

    def detect(self, agv_states: Dict[int, dict], occupied: Set[Tuple[int, int]],
               step: int) -> Optional[List[int]]:
        self.step_counter += 1
        if self.step_counter % self.detection_interval != 0:
            return None

        graph = defaultdict(set)
        active_agvs = []

        for agv_id, state in agv_states.items():
            pos = state.get('position')
            goal = state.get('goal_pos')
            path = state.get('path', [])
            path_idx = state.get('path_index', 0)

            if goal is None:
                continue

            active_agvs.append(agv_id)

            next_pos = None
            if path and path_idx + 1 < len(path):
                next_pos = path[path_idx + 1]
            elif goal:
                next_pos = goal

            if next_pos and next_pos != pos:
                for other_id, other_state in agv_states.items():
                    if other_id != agv_id and other_state.get('position') == next_pos:
                        graph[agv_id].add(other_id)
                        break

        deadlock_cycle = self._find_cycle(graph, active_agvs)
        if deadlock_cycle:
            self.deadlock_count += 1
            self.logger.warning(
                f"Step {step}: Deadlock detected! Cycle: {deadlock_cycle}")
            return deadlock_cycle
        return None

    def recover(self, agv_states: Dict[int, dict],
                cycle: List[int]) -> Dict[int, Tuple[int, int]]:
        if not cycle:
            return {}

        victim_id = cycle[0]
        for agv_id in cycle:
            state = agv_states.get(agv_id, {})
            if not state.get('is_loaded', True):
                victim_id = agv_id
                break

        self.recovery_count += 1
        state = agv_states[victim_id]
        old_pos = state.get('position', (0, 0))
        goal = state.get('goal_pos', old_pos)

        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        random.shuffle(directions)

        best_pos = old_pos
        best_dist = 0
        for dx, dy in directions:
            nx, ny = old_pos[0] + dx, old_pos[1] + dy
            dist = abs(nx - goal[0]) + abs(ny - goal[1])
            if dist > best_dist:
                best_dist = dist
                best_pos = (nx, ny)

        self.logger.info(
            f"Deadlock recovery: AGV {victim_id} backoff {old_pos} → {best_pos}")
        return {victim_id: best_pos}

    def _find_cycle(self, graph: Dict[int, Set[int]],
                    nodes: List[int]) -> Optional[List[int]]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}
        parent = {}

        def dfs(u):
            color[u] = GRAY
            for v in graph.get(u, set()):
                if v not in color:
                    continue
                if color[v] == GRAY:
                    cycle = [v, u]
                    cur = u
                    while parent.get(cur) and parent[cur] != v:
                        cur = parent[cur]
                        cycle.append(cur)
                    cycle.append(v)
                    return cycle
                if color[v] == WHITE:
                    parent[v] = u
                    result = dfs(v)
                    if result:
                        return result
            color[u] = BLACK
            return None

        for node in nodes:
            if color[node] == WHITE:
                result = dfs(node)
                if result:
                    return result
        return None

    def get_stats(self) -> dict:
        return {
            'deadlock_count': self.deadlock_count,
            'recovery_count': self.recovery_count,
            'detection_interval': self.detection_interval,
        }

    def reset(self):
        self.step_counter = 0
        self.deadlock_count = 0
        self.recovery_count = 0
