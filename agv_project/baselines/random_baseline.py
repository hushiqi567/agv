"""随机策略基线"""
import random

ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]

class RandomBaseline:
    """随机策略对照组"""
    def __init__(self):
        pass

    def select_action(self, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(5))
        return random.choice(valid_actions)

    def move(self, pos, grid, width, height, occupied):
        valid = []
        for i, (dx, dy) in enumerate(ACTIONS):
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < width and 0 <= ny < height:
                if grid[ny][nx] != 1 and (nx, ny) not in occupied:
                    valid.append(i)
        if not valid:
            valid.append(4)
        action = random.choice(valid)
        dx, dy = ACTIONS[action]
        return (pos[0] + dx, pos[1] + dy)
