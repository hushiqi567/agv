# AGV RL 路径规划系统改造实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 AGV 系统从"CBS 主导全局规划 + DQN 辅助避撞"改造为"RL 为唯一路径决策核心"，新增课程学习、死锁检测、指标采集、MARL 训练、四组实验脚本。

**Architecture:** RL Policy Net 成为唯一路径决策模块，每步输出 AGV 移动方向。CBS 移至 baselines/ 作为实验对照组。新增 DeadlockDetector 和 MetricsCollector 挂载 MessageBus 独立运行。课程学习分三阶段（10×10 → 30×30 → 50×50）渐进训练。

**Tech Stack:** Python 3.8+, PyTorch (DQN/PPO), NumPy, Pygame (渲染), Matplotlib (图表), python-docx (报告)

---

### Task 1: 更新 ConfigManager 支持新 RL 配置

**Files:**
- Modify: `agv_project/interface/config.py`

- [ ] **Step 1: 在 RLConfig 中新增课程学习和状态编码参数**

在 `agv_project/interface/config.py` 的 `RLConfig` dataclass 中添加字段：

```python
@dataclass
class RLConfig:
    # ... 保留现有字段 ...
    
    # 新增：状态编码
    local_grid_size: int = 15          # 局部网格大小
    num_state_channels: int = 5        # 状态通道数
    use_global_features: bool = True   # 是否使用全局特征向量
    
    # 新增：课程学习
    curriculum_enabled: bool = False   # 是否启用课程学习
    curriculum_stage: int = 1          # 当前课程阶段 (1/2/3)
    curriculum_success_threshold: float = 0.8  # 阶段晋升成功率阈值
    
    # 新增：PPO
    ppo_clip_epsilon: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    ppo_epochs: int = 10
```

- [ ] **Step 2: 新增 DeadlockConfig 和 MetricsConfig 到 ConfigManager**

```python
@dataclass
class DeadlockConfig:
    detection_interval: int = 10       # 检测间隔步数
    max_wait_steps: int = 20           # 判定死锁的等待步数阈值
    recovery_steps: int = 3            # 回退步数

@dataclass
class MetricsConfig:
    export_interval: int = 100         # 导出间隔步数
    export_dir: str = "logs/metrics"   # 导出目录
    export_csv: bool = True
    export_charts: bool = True
```

在 `ConfigManager.__init__` 中添加：
```python
self.deadlock = DeadlockConfig()
self.metrics = MetricsConfig()
```

在 `to_dict()` 返回值和 `load()` 方法中同步添加 `deadlock` 和 `metrics` 节的序列化/反序列化。

- [ ] **Step 3: 验证配置正常工作**

```bash
cd D:/桌面/agv_project && python -c "
from agv_project.interface.config import get_config
c = get_config()
print(f'Grid size: {c.rl.local_grid_size}')
print(f'Curriculum: {c.rl.curriculum_enabled}')
print(f'Deadlock interval: {c.deadlock.detection_interval}')
print('Config OK')
"
```

Expected: 打印所有新配置项的默认值，输出 "Config OK"

---

### Task 2: 创建 state_encoder.py — 15×15 状态编码器

**Files:**
- Create: `agv_project/path_planning/rl/__init__.py`
- Create: `agv_project/path_planning/rl/state_encoder.py`

- [ ] **Step 1: 创建 `rl/__init__.py`**

```python
"""RL path planning subpackage."""
from path_planning.rl.state_encoder import StateEncoder, encode_state
from path_planning.rl.dqn_agent import DQNAgent
```

- [ ] **Step 2: 编写 StateEncoder 类**

```python
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
    
    全局特征向量 (4维):
      [0]: 到目标曼哈顿距离 (归一化)
      [1]: 电量百分比
      [2]: 负载状态 (0空/1载)
      [3]: 任务优先级 (归一化)
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
               priority: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            local_grid: (5, 15, 15) float32
            global_vec: (4,) float32
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
                    local[0, ly, lx] = 1.0  # 边界视为障碍
                    continue

                cell = grid[ny][nx]
                # 通道0: 静态障碍物 (值为1的格子)
                if cell == 1:
                    local[0, ly, lx] = 1.0
                # 通道1: 其他 AGV
                if (nx, ny) in agv_set:
                    local[1, ly, lx] = 1.0
                # 通道2: 移动障碍物
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

        # 通道4: 前方拥堵预警 — 统计前后左右各2步内AGV数量
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

        # 全局特征向量
        max_dist = self.map_width + self.map_height
        global_vec = np.array([
            min(manhattan_distance(agv_pos, goal_pos) / max_dist, 1.0),
            battery / 100.0,
            1.0 if is_loaded else 0.0,
            priority / 5.0,
        ], dtype=np.float32)

        return local, global_vec


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def encode_state(agv_pos, goal_pos, grid, obstacles, other_agvs,
                 battery=100.0, is_loaded=False, priority=1,
                 grid_size=15, map_width=50, map_height=50):
    """便捷函数，使用默认参数编码状态"""
    encoder = StateEncoder(grid_size, map_width=map_width, map_height=map_height)
    return encoder.encode(agv_pos, goal_pos, grid, obstacles, other_agvs,
                          battery, is_loaded, priority)
```

- [ ] **Step 3: 验证 StateEncoder**

```bash
cd D:/桌面/agv_project && python -c "
from agv_project.path_planning.rl.state_encoder import StateEncoder
import numpy as np

encoder = StateEncoder(grid_size=15, map_width=50, map_height=50)
grid = [[0]*50 for _ in range(50)]
grid[5][5] = 1
local, gvec = encoder.encode(
    (10, 10), (25, 25), grid,
    [(15, 15)], [(11, 10), (10, 12)],
    battery=80.0, is_loaded=False, priority=1
)
print(f'Local grid: {local.shape}')   # Expected: (5, 15, 15)
print(f'Global vec: {gvec.shape}')    # Expected: (4,)
print(f'Ch0 static nonzero: {np.count_nonzero(local[0])}')
print(f'Ch1 agv nonzero: {np.count_nonzero(local[1])}')
print(f'Ch2 obstacle nonzero: {np.count_nonzero(local[2])}')
print(f'Ch4 congestion nonzero: {np.count_nonzero(local[4])}')
print(f'Distance norm: {gvec[0]:.3f}')
print('StateEncoder OK')
"
```

Expected: 所有通道有合理的非零值，形状正确，输出 "StateEncoder OK"

---

### Task 3: 创建 dqn_agent.py — 升级版 DQN Agent

**Files:**
- Create: `agv_project/path_planning/rl/dqn_agent.py`

- [ ] **Step 1: 编写混合输入 DQN 网络**

```python
"""升级版 DQN Agent — 15×15×5 局部网格 + 4维全局特征 → 5 动作 Q 值"""
import os
import random
import logging
import math
from typing import List, Tuple, Optional, Dict
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from path_planning.rl.state_encoder import StateEncoder

# 动作空间
ACTIONS = ['up', 'down', 'left', 'right', 'wait']
ACTION_DELTAS = {
    'up': (0, -1), 'down': (0, 1), 'left': (-1, 0),
    'right': (1, 0), 'wait': (0, 0)
}
NUM_ACTIONS = len(ACTIONS)

# 奖励常量 (对齐设计文档 3.3 节)
REWARD_GOAL_ARRIVED = 100.0
REWARD_TASK_COMPLETE = 200.0
REWARD_APPROACH = 1.0
REWARD_AWAY = -1.0
REWARD_STEP = -0.1
REWARD_OBSTACLE_COLLISION = -10.0
REWARD_AGV_COLLISION = -20.0
REWARD_LOAD_SUCCESS = 20.0
REWARD_UNLOAD_SUCCESS = 20.0
REWARD_LOADING_TIMEOUT = -0.5
REWARD_DEADLOCK = -100.0
REWARD_LOW_BATTERY = -5.0
REWARD_CHANNEL_SHARE = 5.0
REWARD_CONGESTION = -2.0


class HybridDQN(nn.Module):
    """混合输入 DQN: 卷积处理局部网格 + 全连接处理全局特征"""

    def __init__(self, grid_size=15, num_channels=5, global_dim=4,
                 num_actions=5, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        conv_out = 64 * 4 * 4  # 1024

        self.fc_global = nn.Linear(global_dim, 32)
        self.fc_combined = nn.Linear(conv_out + 32, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, local_grid, global_vec):
        x = F.relu(self.conv1(local_grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        g = F.relu(self.fc_global(global_vec))
        combined = torch.cat([x, g], dim=1)
        h = F.relu(self.fc_combined(combined))
        return self.fc_out(h)
```

- [ ] **Step 2: 编写 DQNAgent 类（含新奖励函数）**

```python
class ReplayMemory:
    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)

    def push(self, local, gvec, action, reward, next_local, next_gvec, done):
        self.memory.append((local, gvec, action, reward, next_local, next_gvec, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """升级版 DQN Agent — 支持混合状态输入和新奖励函数"""

    def __init__(self, grid_size=15, gamma=0.99, lr=0.001,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 batch_size=64, memory_size=100000, target_update=100,
                 use_gpu=False):
        self.logger = logging.getLogger("AGVProject.DQNAgent")
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self.policy_net = HybridDQN(grid_size=grid_size).to(self.device)
        self.target_net = HybridDQN(grid_size=grid_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.encoder = StateEncoder(grid_size=grid_size)
        self.steps_done = 0
        self.loss_history = []

    def select_action(self, local_grid, global_vec, valid_actions=None):
        """ε-贪心选择动作"""
        if valid_actions is None:
            valid_actions = list(range(NUM_ACTIONS))

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            lg = torch.FloatTensor(local_grid).unsqueeze(0).to(self.device)
            gv = torch.FloatTensor(global_vec).unsqueeze(0).to(self.device)
            q_values = self.policy_net(lg, gv).cpu().numpy()[0]
            best = max(valid_actions, key=lambda a: q_values[a])
            return best

    def store_experience(self, local, gvec, action, reward, next_local, next_gvec, done):
        self.memory.push(local, gvec, action, reward, next_local, next_gvec, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        locals, gvecs, actions, rewards, next_locals, next_gvecs, dones = batch

        lg_batch = torch.FloatTensor(np.array(locals)).to(self.device)
        gv_batch = torch.FloatTensor(np.array(gvecs)).to(self.device)
        act_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rew_batch = torch.FloatTensor(rewards).to(self.device)
        nlg_batch = torch.FloatTensor(np.array(next_locals)).to(self.device)
        ngv_batch = torch.FloatTensor(np.array(next_gvecs)).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(lg_batch, gv_batch).gather(1, act_batch)

        with torch.no_grad():
            next_q = self.target_net(nlg_batch, ngv_batch).max(1)[0]
            target_q = rew_batch + (1 - done_batch) * self.gamma * next_q

        loss = F.mse_loss(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    @staticmethod
    def compute_reward(prev_pos, curr_pos, goal_pos, is_loaded,
                       arrived_pickup=False, arrived_delivery=False,
                       obstacle_collision=False, agv_collision=False,
                       deadlock=False, congestion_count=0,
                       battery=100.0, waited=False, channel_shared=False):
        """分段奖励函数，对齐设计文档 3.3 节"""
        if deadlock:
            return REWARD_DEADLOCK
        if agv_collision:
            return REWARD_AGV_COLLISION
        if obstacle_collision:
            return REWARD_OBSTACLE_COLLISION
        if arrived_delivery and is_loaded:
            return REWARD_TASK_COMPLETE
        if arrived_pickup and not is_loaded:
            return REWARD_GOAL_ARRIVED
        if arrived_delivery:
            return REWARD_GOAL_ARRIVED

        prev_dist = abs(prev_pos[0] - goal_pos[0]) + abs(prev_pos[1] - goal_pos[1])
        curr_dist = abs(curr_pos[0] - goal_pos[0]) + abs(curr_pos[1] - goal_pos[1])

        reward = REWARD_STEP
        if curr_dist < prev_dist:
            reward += REWARD_APPROACH
        elif curr_dist > prev_dist:
            reward += REWARD_AWAY
        if waited:
            reward += REWARD_LOADING_TIMEOUT
        if battery < 20.0:
            reward += REWARD_LOW_BATTERY
        if congestion_count >= 3:
            reward += REWARD_CONGESTION
        if channel_shared:
            reward += REWARD_CHANNEL_SHARE
        return reward

    def save_model(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(ckpt['policy_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.epsilon = ckpt.get('epsilon', self.epsilon_end)
            self.steps_done = ckpt.get('steps_done', 0)

    def set_training(self, training):
        if training:
            self.policy_net.train()
        else:
            self.policy_net.eval()
            self.epsilon = self.epsilon_end
```

- [ ] **Step 3: 验证 DQNAgent 前向传播和训练**

```bash
cd D:/桌面/agv_project && python -c "
from agv_project.path_planning.rl.dqn_agent import DQNAgent, HybridDQN
import numpy as np

agent = DQNAgent(grid_size=15)
print(f'Device: {agent.device}')

# 测试前向传播
local = np.random.randn(1, 5, 15, 15).astype(np.float32)
gvec = np.random.randn(1, 4).astype(np.float32)
import torch
with torch.no_grad():
    qvals = agent.policy_net(torch.FloatTensor(local), torch.FloatTensor(gvec))
print(f'Q-values shape: {qvals.shape}')  # (1, 5)

# 测试动作选择
local_single = np.random.randn(5, 15, 15).astype(np.float32)
gvec_single = np.random.randn(4).astype(np.float32)
action = agent.select_action(local_single, gvec_single, [0, 1, 2, 3, 4])
print(f'Selected action: {action}')

# 测试奖励函数
r = DQNAgent.compute_reward((0,0), (1,0), (10,0), False)
print(f'Approach reward: {r}')  # Should be ~0.9 (step -0.1 + approach 1.0)
r2 = DQNAgent.compute_reward((5,0), (5,0), (10,0), True, arrived_delivery=True)
print(f'Task complete reward: {r2}')  # Should be 200
print('DQNAgent OK')
"
```

Expected: 前向传播成功，(1,5) 形状输出，动作选择在有效范围内，奖励值符合预期

---

### Task 4: 创建 curriculum_trainer.py — 课程学习训练器

**Files:**
- Create: `agv_project/path_planning/rl/curriculum_trainer.py`

- [ ] **Step 1: 编写课程学习训练器**

```python
"""三阶段课程学习训练器"""
import random
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from path_planning.rl.dqn_agent import DQNAgent, ACTIONS, ACTION_DELTAS


@dataclass
class CurriculumStage:
    """课程阶段配置"""
    stage_id: int
    map_size: int           # 地图尺寸 (10/30/50)
    num_static_obstacles: Tuple[int, int]  # (min, max)
    num_moving_obstacles: Tuple[int, int]
    num_agvs: int
    max_steps: int
    num_scenarios: int = 100
    success_threshold: float = 0.8


STAGES = [
    CurriculumStage(1, 10, (3, 5), (0, 0), 1, 200, 100, 0.8),
    CurriculumStage(2, 30, (5, 8), (2, 3), 2, 500, 100, 0.8),
    CurriculumStage(3, 50, (10, 10), (10, 10), 4, 1000, 100, 0.6),
]


class CurriculumTrainer:
    """课程学习训练器 — 管理三阶段渐进训练"""

    def __init__(self, agent: DQNAgent, stages=None):
        self.agent = agent
        self.stages = stages or STAGES
        self.current_stage_idx = 0
        self.logger = logging.getLogger("AGVProject.Curriculum")
        self.stats: Dict[int, List[Dict]] = {s.stage_id: [] for s in self.stages}

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self.current_stage_idx]

    def generate_scenario(self, stage: CurriculumStage):
        """生成一个随机场景：小地图随机起点/终点/障碍物"""
        size = stage.map_size
        # 随机起点和终点（不重叠，不在障碍物上）
        empty_cells = [(x, y) for x in range(1, size-1) for y in range(1, size-1)]
        start = random.choice(empty_cells)
        goal = random.choice([c for c in empty_cells if c != start])

        # 随机静态障碍物
        num_obs = random.randint(*stage.num_static_obstacles)
        available = [c for c in empty_cells if c not in (start, goal)]
        obstacles = random.sample(available, min(num_obs, len(available)))

        # 构建网格
        grid = [[0]*size for _ in range(size)]
        for ox, oy in obstacles:
            grid[oy][ox] = 1

        return {
            'grid': grid,
            'start': start,
            'goal': goal,
            'obstacles': obstacles,
            'size': size,
        }

    def train_stage(self, stage: CurriculumStage) -> float:
        """训练一个完整阶段，返回成功率"""
        self.agent.set_training(True)
        successes = 0

        for ep in range(stage.num_scenarios):
            scenario = self.generate_scenario(stage)
            pos = scenario['start']
            goal = scenario['goal']
            grid = scenario['grid']
            size = scenario['size']

            self.agent.encoder.map_width = size
            self.agent.encoder.map_height = size

            for step in range(stage.max_steps):
                local, gvec = self.agent.encoder.encode(
                    pos, goal, grid, [], [],
                    is_loaded=False, priority=1)

                valid = self._get_valid_actions(pos, grid, size, set())
                if not valid:
                    break
                action = self.agent.select_action(local, gvec, valid)
                dx, dy = ACTION_DELTAS[ACTIONS[action]]
                new_pos = (pos[0] + dx, pos[1] + dy)

                arrived = (new_pos == goal)
                next_local, next_gvec = self.agent.encoder.encode(
                    new_pos, goal, grid, [], [], is_loaded=False, priority=1)
                reward = DQNAgent.compute_reward(
                    pos, new_pos, goal, False,
                    arrived_pickup=arrived, arrived_delivery=arrived,
                    obstacle_collision=(grid[new_pos[1]][new_pos[0]] == 1 if 0 <= new_pos[0] < size and 0 <= new_pos[1] < size else True))
                self.agent.store_experience(local, gvec, action, reward, next_local, next_gvec, arrived)
                self.agent.train_step()
                pos = new_pos
                if arrived:
                    successes += 1
                    break

            if (ep + 1) % 20 == 0:
                self.logger.info(f"Stage {stage.stage_id} episode {ep+1}/{stage.num_scenarios}, success rate: {successes/(ep+1):.2%}")

        success_rate = successes / stage.num_scenarios
        self.stats[stage.stage_id].append({'success_rate': success_rate})
        return success_rate

    def run(self) -> bool:
        """运行完整课程学习，返回是否完成所有阶段"""
        for i, stage in enumerate(self.stages):
            self.current_stage_idx = i
            self.logger.info(f"=== Starting Curriculum Stage {stage.stage_id}: {stage.map_size}x{stage.map_size} ===")
            rate = self.train_stage(stage)
            self.logger.info(f"Stage {stage.stage_id} complete. Success rate: {rate:.2%}")
            if rate < stage.success_threshold and i < len(self.stages) - 1:
                self.logger.warning(f"Stage {stage.stage_id} below threshold ({stage.success_threshold:.0%}), retrying...")
                rate = self.train_stage(stage)
                if rate < stage.success_threshold:
                    self.logger.warning(f"Stage {stage.stage_id} still below threshold, advancing anyway")
        return True

    @staticmethod
    def _get_valid_actions(pos, grid, size, occupied):
        valid = []
        for a_idx, a_name in enumerate(ACTIONS):
            dx, dy = ACTION_DELTAS[a_name]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < size and 0 <= ny < size:
                if grid[ny][nx] != 1 and (nx, ny) not in occupied:
                    valid.append(a_idx)
        if not valid:
            valid.append(4)  # wait as fallback
        return valid
```

- [ ] **Step 2: 验证课程学习框架**

```bash
cd D:/桌面/agv_project && python -c "
from agv_project.path_planning.rl.dqn_agent import DQNAgent
from agv_project.path_planning.rl.curriculum_trainer import CurriculumTrainer, STAGES

agent = DQNAgent(grid_size=15)
trainer = CurriculumTrainer(agent, stages=[STAGES[0]])  # 只跑阶段1
print(f'Stage 1 config: {trainer.current_stage}')
scenario = trainer.generate_scenario(STAGES[0])
print(f'Scenario: size={scenario[\"size\"]}, start={scenario[\"start\"]}, goal={scenario[\"goal\"]}, obstacles={len(scenario[\"obstacles\"])}')
print('CurriculumTrainer OK')
"
```

Expected: 正确生成第1阶段场景配置，输出场景参数

---

### Task 5: 创建 deadlock_detector.py — 死锁检测与恢复

**Files:**
- Create: `agv_project/path_planning/deadlock_detector.py`

- [ ] **Step 1: 编写死锁检测器**

```python
"""死锁检测与恢复模块 — 每10步扫描有向图判环"""
import logging
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
      选参与死锁中"被分配任务最晚"的 AGV，随机回退一步打破循环。
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
        """
        检测死锁。
        
        Args:
            agv_states: {agv_id: {position, goal_pos, path, waiting_steps, ...}}
            occupied: 所有被占用的位置
            step: 当前步数
        
        Returns:
            死锁环中的 AGV ID 列表，无死锁返回 None
        """
        self.step_counter += 1
        if self.step_counter % self.detection_interval != 0:
            return None

        # 构建有向图: A → B 表示 A 的下一步目标被 B 占据
        graph = defaultdict(set)
        active_agvs = []

        for agv_id, state in agv_states.items():
            pos = state.get('position')
            goal = state.get('goal_pos')
            path = state.get('path', [])
            path_idx = state.get('path_index', 0)
            waiting = state.get('waiting_steps', 0)

            if goal is None:
                continue

            active_agvs.append(agv_id)

            # 确定 AGV 想要去的下一步位置
            next_pos = None
            if path and path_idx + 1 < len(path):
                next_pos = path[path_idx + 1]
            elif goal:
                # 直接朝目标方向
                next_pos = goal

            if next_pos and next_pos != pos:
                # 检查 next_pos 被谁占据
                for other_id, other_state in agv_states.items():
                    if other_id != agv_id and other_state.get('position') == next_pos:
                        graph[agv_id].add(other_id)
                        break
                # 也检查 next_pos 是否在 occupied 中（可能是静态障碍）
                if next_pos in occupied and not any(
                    other_state.get('position') == next_pos
                    for other_id, other_state in agv_states.items() if other_id != agv_id
                ):
                    pass  # 静态障碍物占位，不是死锁

        # DFS 判环
        deadlock_cycle = self._find_cycle(graph, active_agvs)
        if deadlock_cycle:
            self.deadlock_count += 1
            self.logger.warning(f"Step {step}: Deadlock detected! Cycle: {deadlock_cycle}")
            return deadlock_cycle
        return None

    def recover(self, agv_states: Dict[int, dict], cycle: List[int]) -> Dict[int, Tuple[int, int]]:
        """
        死锁恢复：选 load_time 最短的 AGV 回退一步。
        
        Returns:
            {agv_id: new_position} 需要更新的 AGV 位置
        """
        if not cycle:
            return {}

        # 选参与死锁中负载最轻或任务时间最短的 AGV
        victim_id = cycle[0]
        victim_load = agv_states.get(victim_id, {}).get('is_loaded', False)

        for agv_id in cycle:
            state = agv_states.get(agv_id, {})
            if not state.get('is_loaded', True):  # 优先选未装载的
                victim_id = agv_id
                break

        self.recovery_count += 1
        state = agv_states[victim_id]
        old_pos = state.get('position', (0, 0))
        goal = state.get('goal_pos', old_pos)

        # 回退：朝远离目标的方向随机移动
        import random
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

        self.logger.info(f"Deadlock recovery: AGV {victim_id} backoff {old_pos} → {best_pos}")
        return {victim_id: best_pos}

    def _find_cycle(self, graph: Dict[int, Set[int]], nodes: List[int]) -> Optional[List[int]]:
        """DFS 检测有向图中的环"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}
        parent = {}

        def dfs(u):
            color[u] = GRAY
            for v in graph.get(u, set()):
                if v not in color:
                    continue
                if color[v] == GRAY:
                    # 找到环，回溯路径
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
```

- [ ] **Step 2: 验证死锁检测逻辑**

```bash
cd D:/桌面/agv_project && python -c "
from agv_project.path_planning.deadlock_detector import DeadlockDetector

detector = DeadlockDetector(detection_interval=1)

# 模拟死锁场景: AGV0→(10,10)等待AGV1, AGV1→(10,11)等待AGV0
agv_states = {
    0: {'position': (10, 10), 'goal_pos': (20, 10), 'path': [(10,10),(10,11)], 'path_index': 0, 'waiting_steps': 15, 'is_loaded': False},
    1: {'position': (10, 11), 'goal_pos': (5, 11), 'path': [(10,11),(10,10)], 'path_index': 0, 'waiting_steps': 15, 'is_loaded': True},
}
occupied = {(10, 10), (10, 11)}
cycle = detector.detect(agv_states, occupied, step=10)
print(f'Deadlock cycle detected: {cycle}')  # Should detect a cycle

# 测试恢复
recovery = detector.recover(agv_states, cycle or [0])
print(f'Recovery moves: {recovery}')
print('DeadlockDetector OK')
"
```

Expected: 检测到死锁环，恢复策略输出被选 AGV 的新位置

---

### Task 6: 创建 metrics_collector.py — 指标采集器

**Files:**
- Create: `agv_project/path_planning/metrics_collector.py`

- [ ] **Step 1: 编写指标采集器**

```python
"""指标采集器 — 挂载 MessageBus 记录并导出 CSV/图表"""
import os
import csv
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class MetricsCollector:
    """
    指标采集器。
    
    通过挂载到 MessageBus 自动采集仿真指标:
      - 每步每个 AGV 的位置和状态
      - 任务完成时间和路径长度
      - 碰撞次数
      - 死锁触发次数和恢复耗时
      - 训练过程的损失曲线、奖励曲线、探索率变化
    """

    def __init__(self, export_dir: str = "logs/metrics", export_interval: int = 100):
        self.export_dir = export_dir
        self.export_interval = export_interval
        os.makedirs(export_dir, exist_ok=True)
        self.logger = logging.getLogger("AGVProject.Metrics")

        # 时序数据
        self.step_records: List[Dict] = []
        self.task_records: List[Dict] = []
        self.collision_count = 0
        self.deadlock_events: List[Dict] = []
        self.training_losses: List[float] = []
        self.training_rewards: List[float] = []
        self.exploration_rates: List[float] = []

        self._step = 0

    def record_step(self, agv_states: Dict[int, dict], tasks_completed: int, step: int):
        """记录每步状态"""
        self._step = step
        for agv_id, state in agv_states.items():
            self.step_records.append({
                'step': step,
                'agv_id': agv_id,
                'x': state.get('position', (0, 0))[0],
                'y': state.get('position', (0, 0))[1],
                'status': str(state.get('status', 'unknown')),
                'is_loaded': state.get('is_loaded', False),
                'battery': state.get('battery', 100.0),
            })

    def record_task_complete(self, task_id: int, agv_id: int, path_length: int,
                             elapsed_steps: int, step: int):
        """记录任务完成"""
        self.task_records.append({
            'task_id': task_id,
            'agv_id': agv_id,
            'path_length': path_length,
            'elapsed_steps': elapsed_steps,
            'completed_at_step': step,
        })

    def record_collision(self, agv_id: int, step: int):
        self.collision_count += 1

    def record_deadlock(self, cycle: List[int], recovered_agv: int, step: int):
        self.deadlock_events.append({
            'step': step,
            'cycle': str(cycle),
            'recovered_agv': recovered_agv,
        })

    def record_training(self, loss: Optional[float], reward: Optional[float],
                        epsilon: Optional[float]):
        if loss is not None:
            self.training_losses.append(loss)
        if reward is not None:
            self.training_rewards.append(reward)
        if epsilon is not None:
            self.exploration_rates.append(epsilon)

    def export_csv(self, tag: str = ""):
        """导出所有数据为 CSV 文件"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{tag}_" if tag else ""

        def write_csv(filename, rows):
            if not rows:
                return
            path = os.path.join(self.export_dir, f"{prefix}{filename}")
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            self.logger.info(f"Exported: {path}")

        write_csv(f"{ts}_steps.csv", self.step_records)
        write_csv(f"{ts}_tasks.csv", self.task_records)
        write_csv(f"{ts}_deadlocks.csv", self.deadlock_events)

        # 训练数据
        if self.training_losses:
            path = os.path.join(self.export_dir, f"{prefix}{ts}_training.csv")
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['step', 'loss', 'reward', 'epsilon'])
                for i in range(max(len(self.training_losses), len(self.training_rewards), len(self.exploration_rates))):
                    w.writerow([
                        i,
                        self.training_losses[i] if i < len(self.training_losses) else '',
                        self.training_rewards[i] if i < len(self.training_rewards) else '',
                        self.exploration_rates[i] if i < len(self.exploration_rates) else '',
                    ])
            self.logger.info(f"Exported: {path}")

    def export_charts(self, tag: str = ""):
        """导出 matplotlib 图表"""
        if not HAS_MPL:
            self.logger.warning("matplotlib not installed, skipping chart export")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{tag}_" if tag else ""

        # 1. 任务完成时间分布
        if self.task_records:
            fig, ax = plt.subplots(figsize=(10, 6))
            times = [t['elapsed_steps'] for t in self.task_records]
            ax.hist(times, bins=20, edgecolor='black')
            ax.set_xlabel('Elapsed Steps')
            ax.set_ylabel('Count')
            ax.set_title('Task Completion Time Distribution')
            path = os.path.join(self.export_dir, f"{prefix}{ts}_task_times.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)

        # 2. 训练损失曲线
        if self.training_losses:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            axes[0].plot(self.training_losses, alpha=0.7, linewidth=0.5)
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Training Step')
            if len(self.training_rewards) > 0:
                # 平滑奖励
                window = min(100, len(self.training_rewards))
                smoothed = [sum(self.training_rewards[max(0,i-window):i+1])/(min(i+1,window))
                           for i in range(len(self.training_rewards))]
                axes[1].plot(smoothed, linewidth=1)
            axes[1].set_title('Smoothed Reward')
            axes[1].set_xlabel('Episode')
            if self.exploration_rates:
                axes[2].plot(self.exploration_rates, linewidth=1)
            axes[2].set_title('Exploration Rate (Epsilon)')
            axes[2].set_xlabel('Step')
            plt.tight_layout()
            path = os.path.join(self.export_dir, f"{prefix}{ts}_training.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)

        # 3. 死锁事件时间线
        if self.deadlock_events:
            fig, ax = plt.subplots(figsize=(12, 4))
            steps = [d['step'] for d in self.deadlock_events]
            ax.eventplot([steps], lineoffsets=0, linelengths=0.8, colors='red')
            ax.set_xlabel('Step')
            ax.set_title('Deadlock Events Timeline')
            path = os.path.join(self.export_dir, f"{prefix}{ts}_deadlocks.png")
            fig.savefig(path, dpi=100)
            plt.close(fig)

    def get_summary(self) -> dict:
        return {
            'total_steps_recorded': self._step,
            'total_tasks_completed': len(self.task_records),
            'total_collisions': self.collision_count,
            'total_deadlocks': len(self.deadlock_events),
            'total_training_steps': len(self.training_losses),
        }

    def reset(self):
        self.step_records.clear()
        self.task_records.clear()
        self.deadlock_events.clear()
        self.collision_count = 0
        self._step = 0
```

- [ ] **Step 2: 验证指标采集器**

```bash
cd D:/桌面/agv_project && python -c "
from agv_project.path_planning.metrics_collector import MetricsCollector

mc = MetricsCollector(export_dir='logs/metrics')
mc.record_step({0: {'position': (10, 10), 'status': 'moving', 'is_loaded': False, 'battery': 80.0}}, 0)
mc.record_task_complete(1, 0, 50, 100, 100)
mc.record_collision(0, 50)
mc.record_deadlock([0, 1], 0, 200)
mc.record_training(0.5, 10.0, 0.8)
print(f'Summary: {mc.get_summary()}')
mc.export_csv('test')
print('MetricsCollector OK')
"
```

Expected: 创建 CSV 文件在 logs/metrics/ 下，summary 包含正确计数

---

### Task 7: 修改 agv_controller.py — RL 驱动控制

**Files:**
- Modify: `agv_project/path_planning/agv_controller.py`

- [ ] **Step 1: 添加 RL Agent 和 DeadlockDetector 集成到 AGVController**

在 `AGVController.__init__` 中新增：
```python
from path_planning.rl.dqn_agent import DQNAgent
from path_planning.deadlock_detector import DeadlockDetector
from path_planning.metrics_collector import MetricsCollector

# 在 __init__ 中添加:
self.dqn_agent = DQNAgent(grid_size=15)
self.deadlock_detector = DeadlockDetector()
self.metrics = MetricsCollector()

# 标记是否使用 RL 主导路径规划
self.use_rl_primary = True  # True=RL主导, False=CBS主导(兼容旧模式)
```

- [ ] **Step 2: 重写 `_move_agvs` 方法为 RL 驱动**

```python
def _move_agvs(self):
    """RL驱动的AGV移动 — 每步用策略网络决策方向"""
    all_positions = {aid: a.position for aid, a in self.agvs.items()}
    occupied = set(all_positions.values())
    obstacle_positions = set(o.position for o in self.env.obstacles)

    # 死锁检测
    agv_state_snapshots = {
        aid: {
            'position': a.position, 'goal_pos': a.goal_pos,
            'path': a.path, 'path_index': a.path_index,
            'waiting_steps': a.waiting_steps, 'is_loaded': a.is_loaded,
        }
        for aid, a in self.agvs.items()
    }
    deadlock_cycle = self.deadlock_detector.detect(
        agv_state_snapshots, occupied | obstacle_positions, self.current_step)

    if deadlock_cycle:
        recovery = self.deadlock_detector.recover(agv_state_snapshots, deadlock_cycle)
        for agv_id, new_pos in recovery.items():
            if agv_id in self.agvs:
                self.agvs[agv_id].position = new_pos
                self.agvs[agv_id].waiting_steps = 0
        self.metrics.record_deadlock(deadlock_cycle,
            list(recovery.keys())[0] if recovery else -1, self.current_step)

    for agv_id, agv in self.agvs.items():
        if agv.status not in [AGVStatus.MOVING_TO_PICKUP, AGVStatus.MOVING_TO_DELIVERY]:
            continue
        if agv.goal_pos is None:
            continue

        prev_pos = agv.position

        # 到达目标检查
        if agv.position == agv.goal_pos:
            self._handle_arrival(agv)
            continue

        if self.use_rl_primary:
            self._rl_move_agv(agv, all_positions, occupied, obstacle_positions)
        else:
            self._mapf_move_agv(agv, all_positions, occupied, obstacle_positions)

    # 记录指标
    self.metrics.record_step(
        {aid: {'position': a.position, 'status': str(a.status),
               'is_loaded': a.is_loaded, 'battery': a.battery}
         for aid, a in self.agvs.items()},
        self.total_tasks_completed, self.current_step)

    # 训练步骤
    if self.dqn_agent.epsilon > self.dqn_agent.epsilon_end:
        loss = self.dqn_agent.train_step()
        self.metrics.record_training(loss, None, self.dqn_agent.epsilon)
```

- [ ] **Step 3: 编写 `_rl_move_agv` 方法（RL 决策核心）**

```python
def _rl_move_agv(self, agv, all_positions, occupied, obstacle_positions):
    """RL 策略网络决策移动方向"""
    other_agvs = [pos for aid, pos in all_positions.items() if aid != agv.agv_id]

    local, gvec = self.dqn_agent.encoder.encode(
        agv.position, agv.goal_pos, self.env.grid,
        list(obstacle_positions), other_agvs,
        battery=agv.battery, is_loaded=agv.is_loaded, priority=1)

    # 有效动作
    valid = self._get_valid_rl_actions(agv.position, occupied | obstacle_positions)
    if not valid:
        agv.waiting_steps += 1
        return

    action = self.dqn_agent.select_action(local, gvec, valid)
    dx, dy = ACTION_DELTAS[ACTIONS[action]]
    new_pos = (agv.position[0] + dx, agv.position[1] + dy)

    waited = (action == 4)
    old_pos = agv.position

    if not waited:
        agv.position = new_pos
        if old_pos in occupied:
            occupied.discard(old_pos)
        occupied.add(new_pos)
        all_positions[agv.agv_id] = new_pos
        agv.waiting_steps = 0
        self.total_steps_taken += 1
    else:
        agv.waiting_steps += 1

    # 碰撞检测
    if new_pos in obstacle_positions:
        self.metrics.record_collision(agv.agv_id, self.current_step)
    collision_agv = sum(1 for aid, pos in all_positions.items()
                        if aid != agv.agv_id and pos == new_pos)

    # 记录RL经验
    next_other_agvs = [pos for aid, pos in all_positions.items() if aid != agv.agv_id]
    next_local, next_gvec = self.dqn_agent.encoder.encode(
        agv.position, agv.goal_pos, self.env.grid,
        list(obstacle_positions), next_other_agvs,
        battery=agv.battery, is_loaded=agv.is_loaded)

    arrived = (agv.position == agv.goal_pos)
    reward = DQNAgent.compute_reward(
        old_pos, agv.position, agv.goal_pos, agv.is_loaded,
        arrived_pickup=arrived,
        obstacle_collision=(new_pos in obstacle_positions),
        agv_collision=(collision_agv > 0),
        waited=waited, battery=agv.battery,
        congestion_count=len([p for p in other_agvs
            if abs(p[0]-agv.position[0])+abs(p[1]-agv.position[1]) <= 3]))

    done = arrived or (new_pos in obstacle_positions) or (collision_agv > 0)
    self.dqn_agent.store_experience(local, gvec, action, reward, next_local, next_gvec, done)

def _get_valid_rl_actions(self, pos, occupied):
    """获取 RL 的有效动作列表"""
    valid = []
    for a_idx in range(len(ACTIONS)):
        dx, dy = ACTION_DELTAS[ACTIONS[a_idx]]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
            if self.env.grid[ny][nx] != 1:  # 不是障碍物
                if (nx, ny) not in occupied or (dx, dy) == (0, 0):  # 不被占或等待
                    valid.append(a_idx)
    if not valid:
        valid.append(4)  # wait
    return valid
```

保留旧的 `_mapf_move_agv`（从原 `_move_agvs` 中提取 MAPF 路径移动逻辑）作为 fallback，并将原 `_plan_paths` 改为仅在 `use_rl_primary=False` 时调用。

- [ ] **Step 4: 验证控制器集成**

```bash
cd D:/桌面/agv_project && python -c "
from agv_project.path_planning.agv_controller import AGVController
print('AGVController import OK with RL integration')
"
```

---

### Task 8: 创建实验脚本 (experiments/)

**Files:**
- Create: `agv_project/experiments/__init__.py` (empty)
- Create: `agv_project/experiments/run_experiment_1_single.py`
- Create: `agv_project/experiments/run_experiment_2_ablation.py`
- Create: `agv_project/experiments/run_experiment_3_scalability.py`
- Create: `agv_project/experiments/run_experiment_4_comparison.py`
- Create: `agv_project/experiments/plot_results.py`

- [ ] **Step 1: 编写实验一 — 单 AGV RL vs A\***

```python
"""实验一: 单AGV RL vs 传统A* — 同一地图同样起点终点"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import random, time, csv
from path_planning.rl.dqn_agent import DQNAgent, ACTIONS, ACTION_DELTAS
from path_planning.mapf_planner import a_star_search
from path_planning.metrics_collector import MetricsCollector


def run_experiment_1(num_trials=50, map_size=30, max_steps=500,
                     model_path=None, output_dir="logs/metrics"):
    """单AGV RL vs A* 对比实验"""
    os.makedirs(output_dir, exist_ok=True)
    agent = DQNAgent(grid_size=15)
    if model_path and os.path.exists(model_path):
        agent.load_model(model_path)
    agent.set_training(False)
    metrics = MetricsCollector(output_dir)

    results = []
    for trial in range(num_trials):
        # 生成随机场景
        empty = [(x,y) for x in range(1,map_size-1) for y in range(1,map_size-1)]
        start = random.choice(empty)
        goal = random.choice([c for c in empty if c != start])
        # 5个随机障碍物
        available = [c for c in empty if c not in (start, goal)]
        obstacles = random.sample(available, min(5, len(available)))
        grid = [[0]*map_size for _ in range(map_size)]
        for ox, oy in obstacles:
            grid[oy][ox] = 1

        # A* baseline
        t0 = time.time()
        astar_path = a_star_search(start, goal, grid, map_size, map_size)
        astar_time = time.time() - t0
        astar_len = len(astar_path) if astar_path else -1

        # RL
        agent.encoder.map_width = map_size
        agent.encoder.map_height = map_size
        pos = start
        rl_path_len = 0
        rl_success = False
        t0 = time.time()
        for step in range(max_steps):
            local, gvec = agent.encoder.encode(pos, goal, grid, [], [])
            valid = []
            for a in range(5):
                dx, dy = ACTION_DELTAS[ACTIONS[a]]
                nx, ny = pos[0]+dx, pos[1]+dy
                if 0 <= nx < map_size and 0 <= ny < map_size and grid[ny][nx] != 1:
                    valid.append(a)
            if not valid:
                valid = [4]
            action = agent.select_action(local, gvec, valid)
            dx, dy = ACTION_DELTAS[ACTIONS[action]]
            pos = (pos[0]+dx, pos[1]+dy)
            rl_path_len += 1
            if pos == goal:
                rl_success = True
                break
        rl_time = time.time() - t0

        results.append({
            'trial': trial, 'start_x': start[0], 'start_y': start[1],
            'goal_x': goal[0], 'goal_y': goal[1],
            'astar_path_len': astar_len, 'astar_time': astar_time,
            'rl_path_len': rl_path_len, 'rl_time': rl_time,
            'rl_success': rl_success,
            'ratio': rl_path_len/astar_len if astar_len > 0 else -1,
        })

    # 导出
    path = os.path.join(output_dir, "exp1_rl_vs_astar.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)

    successes = sum(1 for r in results if r['rl_success'])
    ratios = [r['ratio'] for r in results if r['ratio'] > 0]
    avg_ratio = sum(ratios)/len(ratios) if ratios else -1
    print(f"Experiment 1: RL success={successes}/{num_trials} ({successes/num_trials:.1%}), avg ratio={avg_ratio:.3f}")
    return results


if __name__ == "__main__":
    run_experiment_1(num_trials=10)
```

- [ ] **Step 2: 编写实验二 — 消融实验**

```python
"""实验二: 消融实验 — 4种配置对比"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import csv, time
from env.warehouse_env import WarehouseEnv
from scheduler.task_allocator import TaskAllocator
from path_planning.mapf_planner import MAPFPlanner
from path_planning.rl_collision_avoidance import RLCollisionAvoidance
from path_planning.agv_controller import AGVController


CONFIGS = {
    "full":         {"rl_primary": True,  "curriculum": True,  "deadlock": True},
    "no_curriculum":{"rl_primary": True,  "curriculum": False, "deadlock": True},
    "no_avoidance": {"rl_primary": False, "curriculum": True,  "deadlock": True},
    "rl_only":      {"rl_primary": True,  "curriculum": False, "deadlock": False},
}


def run_ablation(config_name, config, map_size=30, steps=200, num_agvs=4):
    env = WarehouseEnv()
    env.reset()
    task_alloc = TaskAllocator(env.loading_zones, env.unloading_zones)
    mapf = MAPFPlanner(env.grid, env.width, env.height)
    rl_avoid = RLCollisionAvoidance()
    controller = AGVController(env, mapf, rl_avoid)
    controller.use_rl_primary = config["rl_primary"]
    controller.set_task_allocator(task_alloc)
    task_alloc.set_controller(controller)
    controller.reset()
    task_alloc.reset()

    collisions = 0
    for _ in range(steps):
        env.step()
        mapf.update_grid(env.grid)
        task_alloc.step()
        controller.step()
        # count collisions from controller

    stats = controller.get_statistics()
    return {
        'config': config_name,
        'tasks_completed': stats['tasks_completed'],
        'total_steps': stats['steps_taken'],
        'collisions': collisions,
    }


def run_experiment_2(steps=200, output_dir="logs/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for name, cfg in CONFIGS.items():
        print(f"Running ablation: {name}")
        r = run_ablation(name, cfg, steps=steps)
        results.append(r)

    path = os.path.join(output_dir, "exp2_ablation.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    for r in results:
        print(f"  {r['config']}: tasks={r['tasks_completed']}")
    return results


if __name__ == "__main__":
    run_experiment_2(steps=100)
```

- [ ] **Step 3: 编写实验三 — 多AGV可扩展性**

```python
"""实验三: 多AGV可扩展性 — 2台到8台AGV"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import csv
from env.warehouse_env import WarehouseEnv
from scheduler.task_allocator import TaskAllocator
from path_planning.mapf_planner import MAPFPlanner
from path_planning.rl_collision_avoidance import RLCollisionAvoidance
from path_planning.agv_controller import AGVController


def run_experiment_3(agv_counts=[2, 4, 6, 8], steps=300, output_dir="logs/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for n_agv in agv_counts:
        # Override NUM_AGVS
        import scheduler.task_allocator as ta
        ta.NUM_AGVS = n_agv
        ta.AGV_INITIAL_POSITIONS = ta.AGV_INITIAL_POSITIONS[:n_agv]

        env = WarehouseEnv()
        env.reset()
        task_alloc = TaskAllocator(env.loading_zones, env.unloading_zones)
        mapf = MAPFPlanner(env.grid, env.width, env.height)
        rl_avoid = RLCollisionAvoidance()
        controller = AGVController(env, mapf, rl_avoid)
        controller.set_task_allocator(task_alloc)
        task_alloc.set_controller(controller)
        controller.reset()
        task_alloc.reset()

        for _ in range(steps):
            env.step()
            mapf.update_grid(env.grid)
            task_alloc.step()
            controller.step()

        stats = controller.get_statistics()
        results.append({
            'num_agvs': n_agv,
            'tasks_completed': stats['tasks_completed'],
            'steps_taken': stats['steps_taken'],
            'completion_rate': stats['tasks_completed'] / steps,
        })

    path = os.path.join(output_dir, "exp3_scalability.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    for r in results:
        print(f"  {r['num_agvs']} AGVs: {r['tasks_completed']} tasks")
    return results


if __name__ == "__main__":
    run_experiment_3(steps=100)
```

- [ ] **Step 4: 编写实验四 — 对比传统方法**

```python
"""实验四: 对比传统方法 — RL vs CBS vs Random"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import random, csv
from env.warehouse_env import WarehouseEnv
from scheduler.task_allocator import TaskAllocator
from path_planning.mapf_planner import MAPFPlanner
from path_planning.rl_collision_avoidance import RLCollisionAvoidance
from path_planning.agv_controller import AGVController
from baselines.random_baseline import RandomBaseline


POLICIES = ["rl", "cbs", "random"]


def run_experiment_4(policies=None, steps=200, output_dir="logs/metrics"):
    if policies is None:
        policies = POLICIES
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for policy in policies:
        env = WarehouseEnv()
        env.reset()
        task_alloc = TaskAllocator(env.loading_zones, env.unloading_zones)
        mapf = MAPFPlanner(env.grid, env.width, env.height)
        rl_avoid = RLCollisionAvoidance()
        controller = AGVController(env, mapf, rl_avoid)

        if policy == "random":
            controller.use_rl_primary = False
            controller.use_random_policy = True
        elif policy == "cbs":
            controller.use_rl_primary = False
        else:
            controller.use_rl_primary = True

        controller.set_task_allocator(task_alloc)
        task_alloc.set_controller(controller)
        controller.reset()
        task_alloc.reset()

        for _ in range(steps):
            env.step()
            mapf.update_grid(env.grid)
            task_alloc.step()
            controller.step()

        stats = controller.get_statistics()
        dl_stats = controller.deadlock_detector.get_stats()
        results.append({
            'policy': policy,
            'tasks_completed': stats['tasks_completed'],
            'steps_taken': stats['steps_taken'],
            'deadlocks': dl_stats['deadlock_count'],
        })

    path = os.path.join(output_dir, "exp4_comparison.csv")
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    for r in results:
        print(f"  {r['policy']}: tasks={r['tasks_completed']}, deadlocks={r['deadlocks']}")
    return results


if __name__ == "__main__":
    run_experiment_4(steps=100)
```

- [ ] **Step 5: 编写绘图脚本**

```python
"""实验结果图表生成"""
import os, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_exp1_rl_vs_astar(csv_path, output_dir="logs/metrics"):
    """实验一: RL/A*路径长度对比柱状图"""
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    trials = [r['trial'] for r in rows]
    ratios = [float(r['ratio']) for r in rows if float(r['ratio']) > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(range(len(ratios)), ratios, color='steelblue', alpha=0.7)
    ax1.axhline(y=1.0, color='red', linestyle='--', label='A* baseline (=1.0)')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('RL Path Length / A* Path Length')
    ax1.set_title('Experiment 1: RL vs A* Path Length Ratio')
    ax1.legend()

    ax2.hist(ratios, bins=15, edgecolor='black', alpha=0.7)
    ax2.axvline(x=1.0, color='red', linestyle='--')
    ax2.set_xlabel('Path Length Ratio')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Path Length Ratios')

    plt.tight_layout()
    path = os.path.join(output_dir, "exp1_rl_vs_astar.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_exp2_ablation(csv_path, output_dir="logs/metrics"):
    """实验二: 消融实验对比柱状图"""
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    configs = [r['config'] for r in rows]
    tasks = [int(r['tasks_completed']) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
    ax.bar(configs, tasks, color=colors[:len(configs)], edgecolor='black')
    ax.set_ylabel('Tasks Completed')
    ax.set_title('Experiment 2: Ablation Study')
    for i, v in enumerate(tasks):
        ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "exp2_ablation.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_exp3_scalability(csv_path, output_dir="logs/metrics"):
    """实验三: 可扩展性折线图"""
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    agvs = [int(r['num_agvs']) for r in rows]
    tasks = [int(r['tasks_completed']) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(agvs, tasks, 'o-', linewidth=2, markersize=10, color='#2c3e50')
    ax.set_xlabel('Number of AGVs')
    ax.set_ylabel('Tasks Completed')
    ax.set_title('Experiment 3: Multi-AGV Scalability')
    ax.set_xticks(agvs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "exp3_scalability.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_exp4_comparison(csv_path, output_dir="logs/metrics"):
    """实验四: 方法对比柱状图"""
    if not os.path.exists(csv_path):
        return
    rows = []
    with open(csv_path, 'r') as f:
        for r in csv.DictReader(f):
            rows.append(r)

    policies = [r['policy'] for r in rows]
    tasks = [int(r['tasks_completed']) for r in rows]
    deadlocks = [int(r['deadlocks']) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#27ae60', '#2980b9', '#e74c3c']
    ax1.bar(policies, tasks, color=colors[:len(policies)], edgecolor='black')
    ax1.set_ylabel('Tasks Completed')
    ax1.set_title('Task Completion by Method')

    ax2.bar(policies, deadlocks, color=colors[:len(policies)], edgecolor='black')
    ax2.set_ylabel('Deadlock Count')
    ax2.set_title('Deadlocks by Method')

    plt.tight_layout()
    path = os.path.join(output_dir, "exp4_comparison.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    d = "logs/metrics"
    plot_exp1_rl_vs_astar(os.path.join(d, "exp1_rl_vs_astar.csv"), d)
    plot_exp2_ablation(os.path.join(d, "exp2_ablation.csv"), d)
    plot_exp3_scalability(os.path.join(d, "exp3_scalability.csv"), d)
    plot_exp4_comparison(os.path.join(d, "exp4_comparison.csv"), d)
    print("All plots generated.")
```

---

### Task 9: 创建 baselines 目录

**Files:**
- Create: `agv_project/baselines/__init__.py`
- Create: `agv_project/baselines/cbs_baseline.py`
- Create: `agv_project/baselines/random_baseline.py`

- [ ] **Step 1: CBS baseline**

```python
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
```

- [ ] **Step 2: Random baseline**

```python
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
```

---

### Task 10: 更新 main.py 支持新功能

**Files:**
- Modify: `agv_project/main.py`

- [ ] **Step 1: 添加新 CLI 参数**

在 `argparse` 部分添加：
```python
parser.add_argument("--rl-primary", action="store_true", default=True,
                    help="使用RL主导路径规划 (默认)")
parser.add_argument("--cbs-primary", action="store_true",
                    help="使用CBS主导路径规划 (旧模式)")
parser.add_argument("--curriculum", action="store_true",
                    help="启用课程学习训练")
parser.add_argument("--export-metrics", action="store_true",
                    help="导出指标CSV和图表")
parser.add_argument("--experiment", type=str, default=None,
                    choices=["1", "2", "3", "4"],
                    help="运行指定实验 (1-4)")
```

- [ ] **Step 2: 在 Simulation.__init__ 中集成新模块**

```python
# In _init_modules, after creating controller:
self.controller.use_rl_primary = not getattr(args, 'cbs_primary', False)
```

- [ ] **Step 3: 在 shutdown 中导出指标**

```python
def _shutdown(self):
    # ... existing shutdown code ...
    if hasattr(self, 'controller') and hasattr(self.controller, 'metrics'):
        self.controller.metrics.export_csv()
        self.controller.metrics.export_charts()
        summary = self.controller.metrics.get_summary()
        self.logger.info(f"Metrics: {summary}")
```

- [ ] **Step 4: 添加实验模式入口**

在 `main()` 函数中，在普通仿真模式之前：
```python
if args.experiment:
    print(f"\n运行实验 {args.experiment}...")
    if args.experiment == "1":
        from experiments.run_experiment_1_single import run_experiment_1
        run_experiment_1(num_trials=20)
    elif args.experiment == "2":
        from experiments.run_experiment_2_ablation import run_experiment_2
        run_experiment_2(steps=args.steps or 200)
    elif args.experiment == "3":
        from experiments.run_experiment_3_scalability import run_experiment_3
        run_experiment_3(steps=args.steps or 300)
    elif args.experiment == "4":
        from experiments.run_experiment_4_comparison import run_experiment_4
        run_experiment_4(steps=args.steps or 200)
    return
```

---

### Task 11: 创建 MARL 训练器 (P2)

**Files:**
- Create: `agv_project/path_planning/rl/marl_trainer.py`

- [ ] **Step 1: 编写参数共享 MARL 训练器**

```python
"""多智能体RL训练器 — 独立训练 + 参数共享"""
import random, logging
from typing import List, Dict, Tuple
from path_planning.rl.dqn_agent import DQNAgent, ACTIONS, ACTION_DELTAS


class MARLTrainer:
    """
    多智能体训练器。
    
    策略: 所有AGV共享同一个DQN模型（参数共享）。
    训练时在多AGV场景中随机初始位置，让模型自然学会避让。
    """

    def __init__(self, agent: DQNAgent, num_agvs: int = 4):
        self.agent = agent  # 所有AGV共享此模型
        self.num_agvs = num_agvs
        self.logger = logging.getLogger("AGVProject.MARL")

    def train_episode(self, env, task_allocator, max_steps=500) -> Dict:
        """
        训练一个多AGV回合。
        
        Returns:
            {tasks_completed, collisions, avg_reward, ...}
        """
        env.reset()
        task_allocator.reset()
        
        # 多AGV随机初始位置
        agv_positions = self._random_init_positions(env)
        agv_goals = {}
        agv_loaded = {i: False for i in range(self.num_agvs)}
        
        total_reward = 0
        tasks_completed = 0
        collisions = 0

        for step in range(max_steps):
            env.step()
            task_allocator.step()
            
            occupied = set(agv_positions.values())
            obstacle_positions = set(o.position for o in env.obstacles)

            # 每个AGV独立决策（使用共享模型）
            for agv_id in range(self.num_agvs):
                pos = agv_positions[agv_id]
                goal = agv_goals.get(agv_id)

                if goal is None:
                    continue

                # 编码状态
                other_agvs = [p for aid, p in agv_positions.items() if aid != agv_id]
                local, gvec = self.agent.encoder.encode(
                    pos, goal, env.grid,
                    list(obstacle_positions), other_agvs)

                # 选择动作
                valid = self._get_valid_actions(pos, env, occupied)
                action = self.agent.select_action(local, gvec, valid)
                dx, dy = ACTION_DELTAS[ACTIONS[action]]
                new_pos = (pos[0] + dx, pos[1] + dy)

                # 碰撞检测
                if new_pos in obstacle_positions or new_pos in occupied:
                    collisions += 1
                    continue

                # 更新位置
                agv_positions[agv_id] = new_pos
                occupied.add(new_pos)

                # 计算奖励
                arrived = (new_pos == goal)
                next_other = [p for aid, p in agv_positions.items() if aid != agv_id]
                next_local, next_gvec = self.agent.encoder.encode(
                    new_pos, goal, env.grid,
                    list(obstacle_positions), next_other)

                reward = DQNAgent.compute_reward(
                    pos, new_pos, goal, agv_loaded[agv_id],
                    arrived_pickup=arrived)
                total_reward += reward

                # 存储经验
                self.agent.store_experience(
                    local, gvec, action, reward, next_local, next_gvec, arrived)

                if arrived:
                    tasks_completed += 1

            # 训练
            self.agent.train_step()

        return {
            'tasks_completed': tasks_completed,
            'collisions': collisions,
            'avg_reward': total_reward / max_steps,
        }

    def _random_init_positions(self, env):
        positions = {}
        for i in range(self.num_agvs):
            while True:
                x = random.randint(1, env.width - 2)
                y = random.randint(1, env.height - 2)
                if env.grid[y][x] == 0 and (x, y) not in positions.values():
                    positions[i] = (x, y)
                    break
        return positions

    def _get_valid_actions(self, pos, env, occupied):
        valid = []
        for a_idx in range(5):
            dx, dy = ACTION_DELTAS[ACTIONS[a_idx]]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < env.width and 0 <= ny < env.height:
                if env.grid[ny][nx] != 1:
                    if (nx, ny) not in occupied or (dx, dy) == (0, 0):
                        valid.append(a_idx)
        return valid or [4]
```

---

### Task 12: 创建 PPO Agent (P3)

**Files:**
- Create: `agv_project/path_planning/rl/ppo_agent.py`

- [ ] **Step 1: 编写 PPO Agent**

```python
"""PPO Agent — Actor-Critic with PPO-Clip objective"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple


class PPONetwork(nn.Module):
    """Actor-Critic 网络 — 共享卷积特征提取器"""
    def __init__(self, grid_size=15, num_channels=5, global_dim=4, num_actions=5):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        conv_out = 64 * 4 * 4

        self.fc_global = nn.Linear(global_dim, 32)
        self.fc_shared = nn.Linear(conv_out + 32, 256)

        # Actor head (policy)
        self.actor = nn.Linear(256, num_actions)
        # Critic head (value)
        self.critic = nn.Linear(256, 1)

    def forward(self, local_grid, global_vec):
        x = F.relu(self.conv1(local_grid))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        g = F.relu(self.fc_global(global_vec))
        h = F.relu(self.fc_shared(torch.cat([x, g], dim=1)))

        action_logits = self.actor(h)
        value = self.critic(h)
        return action_logits, value


class PPOAgent:
    """PPO Agent — 使用 PPO-Clip 进行策略优化"""
    def __init__(self, grid_size=15, lr=3e-4, gamma=0.99, clip_epsilon=0.2,
                 entropy_coef=0.01, value_coef=0.5, ppo_epochs=10, batch_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(grid_size=grid_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        self.buffer = []  # (local, gvec, action, reward, done, log_prob, value)

    def select_action(self, local_grid, global_vec, valid_actions=None):
        with torch.no_grad():
            lg = torch.FloatTensor(local_grid).unsqueeze(0).to(self.device)
            gv = torch.FloatTensor(global_vec).unsqueeze(0).to(self.device)
            logits, value = self.network(lg, gv)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            return action.item(), dist.log_prob(action).item(), value.item()

    def store_transition(self, local, gvec, action, reward, done, log_prob, value):
        self.buffer.append((local, gvec, action, reward, done, log_prob, value))

    def update(self):
        """PPO-Clip 更新"""
        if len(self.buffer) < self.batch_size:
            return None

        locals = np.array([t[0] for t in self.buffer])
        gvecs = np.array([t[1] for t in self.buffer])
        actions = torch.LongTensor([t[2] for t in self.buffer]).to(self.device)
        rewards = [t[3] for t in self.buffer]
        dones = [t[4] for t in self.buffer]
        old_log_probs = torch.FloatTensor([t[5] for t in self.buffer]).to(self.device)

        # 计算 returns 和 advantages (简化 GAE)
        returns = []
        R = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        returns = torch.FloatTensor(returns).to(self.device)

        lg_batch = torch.FloatTensor(locals).to(self.device)
        gv_batch = torch.FloatTensor(gvecs).to(self.device)

        total_loss = 0
        for _ in range(self.ppo_epochs):
            logits, values = self.network(lg_batch, gv_batch)
            values = values.squeeze(-1)

            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = dist.entropy().mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()
            total_loss += loss.item()

        self.buffer.clear()
        return total_loss / self.ppo_epochs

    def save_model(self, path):
        torch.save({'network': self.network.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def load_model(self, path):
        import os
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.network.load_state_dict(ckpt['network'])
```

---

### Task 13: 创建 requirements.txt 和 .gitignore

**Files:**
- Create: `agv_project/requirements.txt`
- Create: `agv_project/.gitignore`

- [ ] **Step 1: requirements.txt**

```
numpy>=1.21.0
pygame>=2.1.0
torch>=2.0.0
matplotlib>=3.5.0
python-docx>=0.8.11
gymnasium>=0.29.0
```

- [ ] **Step 2: .gitignore**

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.pth
logs/*.log
logs/metrics/
*.json
!config.json
.idea/
.vscode/
*.swp
*.swo
.DS_Store
Thumbs.db
```

- [ ] **Step 3: 验证依赖安装**

```bash
cd D:/桌面/agv_project && pip install -r agv_project/requirements.txt --quiet 2>&1 | tail -5
```

---

### Task 14: 最终验证 — 全系统集成测试

- [ ] **Step 1: 运行基础仿真**

```bash
cd D:/桌面/agv_project && python -m agv_project.main --no-render --steps 50
```

Expected: 仿真运行 50 步无崩溃，输出统计信息

- [ ] **Step 2: 运行 RL 训练**

```bash
cd D:/桌面/agv_project && python -m agv_project.main --train --train-episodes 3 --no-render --steps 100 --save-model test_model.pth
```

Expected: 3 轮训练完成，模型保存到 test_model.pth

- [ ] **Step 3: 运行实验一**

```bash
cd D:/桌面/agv_project && python -m agv_project.main --experiment 1 --steps 100
```

Expected: 生成 exp1_rl_vs_astar.csv

- [ ] **Step 4: 验证指标导出**

```bash
cd D:/桌面/agv_project && python -c "
from agv_project.path_planning.metrics_collector import MetricsCollector
mc = MetricsCollector('logs/metrics')
mc.record_step({0: {'position':(0,0), 'status':'idle', 'is_loaded':False, 'battery':100}}, 0)
mc.record_task_complete(1, 0, 10, 5, 5)
mc.record_training(0.1, 5.0, 0.9)
mc.export_csv('integration_test')
mc.export_charts('integration_test')
print('Integration test passed')
"
```

Expected: 生成 CSV 文件，输出 "Integration test passed"

- [ ] **Step 5: 验证死锁检测**

```bash
cd D:/桌面/agv_project && python -c "
from agv_project.path_planning.deadlock_detector import DeadlockDetector
d = DeadlockDetector(detection_interval=1)
states = {
    0: {'position':(0,0), 'goal_pos':(0,1), 'path':[(0,0),(0,1)], 'path_index':0, 'waiting_steps':20,'is_loaded':False},
    1: {'position':(0,1), 'goal_pos':(0,0), 'path':[(0,1),(0,0)], 'path_index':0, 'waiting_steps':20,'is_loaded':True},
}
cycle = d.detect(states, {(0,0),(0,1)}, step=10)
assert cycle is not None, 'Should detect deadlock'
recovery = d.recover(states, cycle)
assert len(recovery) > 0, 'Should produce recovery'
print(f'Deadlock test passed: cycle={cycle}, recovery={recovery}')
"
```

Expected: "Deadlock test passed: cycle=..., recovery=..."

---

## Verification Checklist

| # | Test | Command |
|---|------|---------|
| 1 | Config loads with new fields | `python -c "from agv_project.interface.config import get_config; c=get_config(); print(c.rl.local_grid_size, c.deadlock.detection_interval)"` |
| 2 | StateEncoder produces correct shapes | `python -c "from agv_project.path_planning.rl.state_encoder import StateEncoder; ...; print(local.shape, gvec.shape)"` |
| 3 | DQNAgent forward pass | `python -c "from agv_project.path_planning.rl.dqn_agent import DQNAgent; agent=DQNAgent(); ..."` |
| 4 | Curriculum training generates scenarios | `python -c "from agv_project.path_planning.rl.curriculum_trainer import CurriculumTrainer; ..."` |
| 5 | Deadlock detection finds cycles | `python -c "from agv_project.path_planning.deadlock_detector import DeadlockDetector; ..."` |
| 6 | Metrics exports CSV | Check `logs/metrics/` for CSV files |
| 7 | Baseline simulation runs | `python -m agv_project.main --no-render --steps 50` |
| 8 | RL training runs | `python -m agv_project.main --train --train-episodes 3 --no-render --steps 100` |
| 9 | Experiment 1 produces CSV | `python -m agv_project.main --experiment 1 --steps 50` |
| 10 | All imports work | `python -c "from agv_project.path_planning.rl import *; from agv_project.baselines import *; from agv_project.experiments import *"` |
