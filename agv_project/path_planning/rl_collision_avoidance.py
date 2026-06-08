"""
============================================
RL实时避撞模块（DQN算法）
============================================
本模块实现了基于深度Q网络（DQN）的实时避撞算法，
用于处理动态障碍物（其他AGV、移动障碍物）的避让。

核心功能：
1. DQN网络定义：卷积+全连接网络
2. 状态表示：AGV周围5×5局部网格 + 自身信息
3. 动作空间：上、下、左、右、等待（5个动作）
4. 奖励函数：基于任务完成、碰撞、进度等
5. 训练和推理接口：支持经验收集和批量训练

使用方式：
    from path_planning.rl_collision_avoidance import RLCollisionAvoidance
    rl = RLCollisionAvoidance()
    action = rl.select_action(state)
"""

import sys
import os
import random
import logging
import math
from typing import List, Tuple, Optional, Dict
from collections import deque

# 将项目根目录添加到系统路径
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# -- 常量定义 --

# 动作映射
ACTIONS = ['up', 'down', 'left', 'right', 'wait']
ACTION_DELTAS = {
    'up': (0, -1),
    'down': (0, 1),
    'left': (-1, 0),
    'right': (1, 0),
    'wait': (0, 0)
}
NUM_ACTIONS = len(ACTIONS)

# DQN参数
STATE_GRID_SIZE = 5      # 局部网格大小 (5x5)
STATE_CHANNELS = 3       # 通道数：障碍物、AGV、目标方向
HIDDEN_SIZE = 128        # 隐藏层大小
LEARNING_RATE = 0.001    # 学习率
GAMMA = 0.99             # 折扣因子
EPSILON_START = 1.0      # 初始探索率
EPSILON_END = 0.05       # 最小探索率
EPSILON_DECAY = 0.995    # 探索率衰减
BATCH_SIZE = 64          # 训练批次大小
MEMORY_SIZE = 100000     # 经验回放池大小（增大以存储更多经验）
TARGET_UPDATE = 100      # 目标网络更新频率
TRAIN_INTERVAL = 4       # 每多少步训练一次

# ===== 奖励函数常量 =====
REWARD_GOAL_ARRIVED = 100.0       # 到达目标（取货点/送货点）
REWARD_TASK_COMPLETE = 150.0      # 完成任务（卸货完成）
REWARD_COLLISION = -50.0          # 碰撞惩罚
REWARD_STEP_PENALTY = -0.1        # 每步小惩罚（鼓励高效路径）
REWARD_CLOSER = 1.0               # 更接近目标
REWARD_FURTHER = -1.0             # 远离目标
REWARD_WAIT_PENALTY = -0.5        # 等待惩罚
REWARD_IDLE_PENALTY = -0.2        # 空闲AGV每步惩罚


# -- DQN网络定义 --

class DQNNetwork(nn.Module):
    """
    DQN神经网络
    
    输入：5×5×3 的局部网格状态
    输出：5个动作的Q值
    """
    
    def __init__(self, input_channels: int = STATE_CHANNELS, 
                 grid_size: int = STATE_GRID_SIZE,
                 num_actions: int = NUM_ACTIONS):
        """
        初始化DQN网络
        
        Args:
            input_channels: 输入通道数
            grid_size: 网格大小
            num_actions: 动作数量
        """
        super(DQNNetwork, self).__init__()
        
        # 卷积层提取空间特征
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # 计算卷积输出大小
        conv_out_size = 32 * grid_size * grid_size
        
        # 全连接层
        self.fc1 = nn.Linear(conv_out_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, num_actions)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入状态 (batch, channels, grid, grid)
        
        Returns:
            动作Q值 (batch, num_actions)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayMemory:
    """经验回放池"""
    
    def __init__(self, capacity: int = MEMORY_SIZE):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """随机采样"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


# -- RL避撞控制器 --

class RLCollisionAvoidance:
    """
    RL实时避撞控制器
    
    使用DQN算法实现实时避撞。
    支持训练模式和推理模式。
    
    Usage:
        >>> rl = RLCollisionAvoidance()
        >>> state = rl.get_state(agv_pos, goal_pos, grid, obstacles)
        >>> action = rl.select_action(state)
        >>> next_pos = rl.apply_action(agv_pos, action)
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        初始化RL避撞控制器
        
        Args:
            use_gpu: 是否使用GPU
        """
        self.logger = logging.getLogger("AGVProject.RL")
        
        # 检查依赖
        if not HAS_TORCH:
            self.logger.warning("PyTorch未安装，使用规则基策略")
            self.use_rules = True
        else:
            self.use_rules = False
        
        if not HAS_NUMPY:
            self.logger.warning("NumPy未安装，使用Python列表")
            self.use_numpy = False
        else:
            self.use_numpy = True
        
        # DQN网络
        if HAS_TORCH:
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            self.policy_net = DQNNetwork().to(self.device)
            self.target_net = DQNNetwork().to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
            self.memory = ReplayMemory()
        
        # 训练状态
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.is_training = False
        
        # 训练统计
        self.total_rewards = 0.0
        self.episode_rewards = 0.0
        self.episode_count = 0
        self.loss_history = []
        
        # 规则基避撞参数
        self.safe_distance = 2  # 安全距离
        
        self.logger.info(f"RL避撞控制器初始化完成 (PyTorch: {HAS_TORCH})")
    
    # ==========================================
    # 奖励函数
    # ==========================================
    
    def compute_reward(self, agv_pos: Tuple[int, int], 
                       prev_pos: Tuple[int, int],
                       goal_pos: Tuple[int, int],
                       arrived: bool,
                       collided: bool,
                       task_completed: bool,
                       waited: bool) -> float:
        """
        计算奖励值
        
        奖励设计原则：
        - 到达目标/完成任务：大额正奖励
        - 碰撞：大额负惩罚
        - 接近目标：小正奖励
        - 远离目标：小负惩罚
        - 等待：小负惩罚
        
        Args:
            agv_pos: AGV当前位置
            prev_pos: AGV上一位置
            goal_pos: 目标位置
            arrived: 是否到达目标
            collided: 是否发生碰撞
            task_completed: 是否完成任务
            waited: 是否执行了等待动作
        
        Returns:
            奖励值
        """
        # 1. 完成任务（最高优先级）
        if task_completed:
            return REWARD_TASK_COMPLETE
        
        # 2. 到达目标（取货点或送货点）
        if arrived:
            return REWARD_GOAL_ARRIVED
        
        # 3. 碰撞惩罚
        if collided:
            return REWARD_COLLISION
        
        # 4. 基于距离变化的奖励
        prev_dist = manhattan_distance(prev_pos, goal_pos)
        curr_dist = manhattan_distance(agv_pos, goal_pos)
        
        reward = 0.0
        
        if curr_dist < prev_dist:
            # 更接近目标
            reward += REWARD_CLOSER
        elif curr_dist > prev_dist:
            # 远离目标
            reward += REWARD_FURTHER
        
        # 5. 等待惩罚
        if waited:
            reward += REWARD_WAIT_PENALTY
        
        # 6. 每步基础惩罚（鼓励高效路径）
        reward += REWARD_STEP_PENALTY
        
        return reward
    
    # ==========================================
    # 经验收集
    # ==========================================
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        存储一条经验到回放池
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        if not HAS_TORCH or self.use_rules:
            return
        
        self.memory.push(state, action, reward, next_state, done)
        
        # 累计奖励
        self.total_rewards += reward
        self.episode_rewards += reward
    
    # ==========================================
    # 状态表示
    # ==========================================
    
    def get_state(self, agv_pos: Tuple[int, int], 
                  goal_pos: Tuple[int, int],
                  grid: List[List[int]],
                  obstacles: List[Tuple[int, int]],
                  other_agvs: List[Tuple[int, int]]):
        """
        获取状态表示
        
        状态由3个通道的5×5局部网格组成：
        - 通道0: 静态障碍物
        - 通道1: 其他AGV和移动障碍物
        - 通道2: 目标方向指示
        
        Args:
            agv_pos: AGV当前位置
            goal_pos: 目标位置
            grid: 全局网格地图
            obstacles: 障碍物位置列表
            other_agvs: 其他AGV位置列表
        
        Returns:
            状态数组 (3, 5, 5)
        """
        if self.use_numpy:
            state = np.zeros((STATE_CHANNELS, STATE_GRID_SIZE, STATE_GRID_SIZE), dtype=np.float32)
        else:
            state = [[[0.0 for _ in range(STATE_GRID_SIZE)] for _ in range(STATE_GRID_SIZE)] for _ in range(STATE_CHANNELS)]
        
        ax, ay = agv_pos
        half = STATE_GRID_SIZE // 2
        
        # 构建局部网格
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                nx, ny = ax + dx, ay + dy
                lx, ly = dx + half, dy + half
                
                # 检查边界
                grid_width = len(grid[0])
                if nx < 0 or nx >= grid_width:
                    continue
                if ny < 0 or ny >= len(grid):
                    continue
                
                # 通道0: 静态障碍物
                if grid[ny][nx] == 1:  # CELL_OBSTACLE
                    if self.use_numpy:
                        state[0, ly, lx] = 1.0
                    else:
                        state[0][ly][lx] = 1.0
                
                # 通道1: 其他AGV和移动障碍物
                if (nx, ny) in other_agvs or (nx, ny) in obstacles:
                    if self.use_numpy:
                        state[1, ly, lx] = 1.0
                    else:
                        state[1][ly][lx] = 1.0
        
        # 通道2: 目标方向指示（归一化方向向量）
        gx, gy = goal_pos
        dx_total = gx - ax
        dy_total = gy - ay
        dist = math.sqrt(dx_total**2 + dy_total**2)
        
        if dist > 0:
            if self.use_numpy:
                state[2, half, half] = dx_total / dist  # x方向
                state[2, half, half + 1] = dy_total / dist  # y方向
            else:
                state[2][half][half] = dx_total / dist
                state[2][half][half + 1] = dy_total / dist
        
        if self.use_numpy:
            return state
        else:
            return np.array(state) if HAS_NUMPY else state
    
    # ==========================================
    # 动作选择
    # ==========================================
    
    def select_action(self, state, valid_actions: Optional[List[int]] = None) -> int:
        """
        选择动作（ε-贪心策略）
        
        Args:
            state: 当前状态
            valid_actions: 有效动作列表（None表示所有动作都有效）
        
        Returns:
            动作索引 (0=up, 1=down, 2=left, 3=right, 4=wait)
        """
        if valid_actions is None:
            valid_actions = list(range(NUM_ACTIONS))
        
        if self.use_rules:
            return self._rule_based_action(state, valid_actions)
        
        # ε-贪心
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # 使用策略网络选择最佳动作
        if HAS_TORCH:
            with torch.no_grad():
                if self.use_numpy:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                else:
                    state_tensor = torch.FloatTensor([state]).to(self.device)
                q_values = self.policy_net(state_tensor)
                
                # 只考虑有效动作
                valid_q = q_values[0].cpu().numpy()
                best_action = valid_actions[0]
                best_q = valid_q[best_action]
                for a in valid_actions[1:]:
                    if valid_q[a] > best_q:
                        best_q = valid_q[a]
                        best_action = a
                return best_action
        
        return valid_actions[0]
    
    def _rule_based_action(self, state, valid_actions: List[int]) -> int:
        """
        规则基避撞策略（当PyTorch不可用时使用）
        
        简单规则：优先向目标方向移动，如果前方有障碍物则选择其他方向
        """
        # 优先选择非等待动作
        move_actions = [a for a in valid_actions if a != 4]  # 4 = wait
        if not move_actions:
            return 4
        
        # 检查每个方向是否有障碍物
        safe_actions = []
        for a in move_actions:
            dx, dy = ACTION_DELTAS[ACTIONS[a]]
            # 检查前方格子
            center = STATE_GRID_SIZE // 2
            check_x = center + dx
            check_y = center + dy
            
            if self.use_numpy:
                has_obstacle = state[0, check_y, check_x] > 0 or state[1, check_y, check_x] > 0
            else:
                has_obstacle = state[0][check_y][check_x] > 0 or state[1][check_y][check_x] > 0
            
            if not has_obstacle:
                safe_actions.append(a)
        
        if not safe_actions:
            return 4  # 所有方向都有障碍物，等待
        
        return random.choice(safe_actions)
    
    def apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        应用动作，计算新位置
        
        Args:
            pos: 当前位置
            action: 动作索引
        
        Returns:
            新位置
        """
        dx, dy = ACTION_DELTAS[ACTIONS[action]]
        return (pos[0] + dx, pos[1] + dy)
    
    def is_valid_action(self, pos: Tuple[int, int], action: int,
                        grid: List[List[int]], width: int, height: int,
                        occupied: set) -> bool:
        """
        检查动作是否有效
        
        Args:
            pos: 当前位置
            action: 动作索引
            grid: 网格地图
            width: 地图宽度
            height: 地图高度
            occupied: 被占用的位置集合
        
        Returns:
            动作是否有效
        """
        new_pos = self.apply_action(pos, action)
        nx, ny = new_pos
        
        # 边界检查
        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            return False
        
        # 障碍物检查（等待动作总是有效）
        if action != 4 and grid[ny][nx] == 1:
            return False
        
        # 占用检查
        if action != 4 and new_pos in occupied:
            return False
        
        return True
    
    def get_valid_actions(self, pos: Tuple[int, int],
                          grid: List[List[int]], width: int, height: int,
                          occupied: set) -> List[int]:
        """
        获取所有有效动作
        
        Args:
            pos: 当前位置
            grid: 网格地图
            width: 地图宽度
            height: 地图高度
            occupied: 被占用的位置集合
        
        Returns:
            有效动作索引列表
        """
        valid = []
        for a in range(NUM_ACTIONS):
            if self.is_valid_action(pos, a, grid, width, height, occupied):
                valid.append(a)
        return valid
    
    # ==========================================
    # 训练接口
    # ==========================================
    
    def train_step(self):
        """
        执行一步训练
        
        从经验回放池采样一个批次，计算损失并更新网络。
        建议每 TRAIN_INTERVAL 步调用一次。
        
        Returns:
            损失值（如果没有足够经验则返回None）
        """
        if self.use_rules or not HAS_TORCH:
            return None
        
        if len(self.memory) < BATCH_SIZE:
            return None
        
        # 采样经验
        transitions = self.memory.sample(BATCH_SIZE)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # 计算当前Q值: Q(sₜ, aₜ)
        current_q = self.policy_net(state_batch).gather(1, action_batch)
        
        # 计算目标Q值: r + γ·maxₐQ(sₜ₊₁, a)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + (1 - done_batch) * GAMMA * next_q
        
        # 计算损失 (MSE)
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # 更新探索率
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        
        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 记录损失
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        
        return loss_val
    
    def train_on_batch(self, experiences: List[Tuple]) -> Optional[float]:
        """
        在给定的经验批次上训练
        
        Args:
            experiences: 经验列表 [(state, action, reward, next_state, done), ...]
        
        Returns:
            损失值
        """
        if self.use_rules or not HAS_TORCH or len(experiences) < 1:
            return None
        
        batch = list(zip(*experiences))
        
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # 计算当前Q值
        current_q = self.policy_net(state_batch).gather(1, action_batch)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + (1 - done_batch) * GAMMA * next_q
        
        # 计算损失
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # 更新探索率
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        
        # 更新目标网络
        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        
        return loss_val
    
    def end_episode(self):
        """
        结束一个训练回合
        
        记录回合奖励，重置累计奖励。
        """
        self.episode_count += 1
        avg_reward = self.episode_rewards / max(1, self.steps_done)
        
        self.logger.info(
            f"回合 {self.episode_count} 结束: "
            f"总奖励={self.episode_rewards:.1f}, "
            f"平均奖励={avg_reward:.3f}, "
            f"探索率={self.epsilon:.3f}, "
            f"经验池={len(self.memory)}"
        )
        
        self.episode_rewards = 0.0
    
    def set_training(self, training: bool):
        """设置训练模式"""
        self.is_training = training
        if training and HAS_TORCH:
            self.policy_net.train()
            self.logger.info("RL切换到训练模式")
        else:
            if HAS_TORCH:
                self.policy_net.eval()
            self.epsilon = EPSILON_END
            self.logger.info("RL切换到推理模式")
    
    def save_model(self, path: str):
        """保存模型"""
        if HAS_TORCH:
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
                'episode_count': self.episode_count,
                'total_rewards': self.total_rewards
            }, path)
            self.logger.info(f"模型已保存到 {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        if HAS_TORCH and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', EPSILON_END)
            self.steps_done = checkpoint.get('steps_done', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.total_rewards = checkpoint.get('total_rewards', 0.0)
            self.logger.info(f"模型已从 {path} 加载")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        avg_loss = np.mean(self.loss_history[-100:]) if self.loss_history else 0.0
        return {
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "memory_size": len(self.memory) if HAS_TORCH else 0,
            "total_rewards": self.total_rewards,
            "episode_count": self.episode_count,
            "avg_loss_100": avg_loss,
            "is_training": self.is_training
        }


# -- 辅助函数 --

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """计算曼哈顿距离"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# -- 独立运行测试 --

def run_test():
    """测试RL避撞控制器"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("RL避撞控制器测试")
    print("=" * 60)
    
    rl = RLCollisionAvoidance()
    
    # 创建测试场景
    width, height = 10, 10
    grid = [[0] * width for _ in range(height)]
    grid[3][3] = 1
    grid[3][4] = 1
    
    agv_pos = (0, 0)
    goal_pos = (9, 9)
    obstacles = [(5, 5)]
    other_agvs = [(2, 3)]
    
    # 测试状态获取
    print("\n测试状态获取:")
    state = rl.get_state(agv_pos, goal_pos, grid, obstacles, other_agvs)
    if HAS_NUMPY:
        print(f"  状态形状: {state.shape}")
        print(f"  通道0(障碍物)非零数: {np.count_nonzero(state[0])}")
        print(f"  通道1(AGV)非零数: {np.count_nonzero(state[1])}")
    
    # 测试动作选择
    print("\n测试动作选择:")
    occupied = set(obstacles + other_agvs)
    valid_actions = rl.get_valid_actions(agv_pos, grid, width, height, occupied)
    print(f"  有效动作: {[ACTIONS[a] for a in valid_actions]}")
    
    action = rl.select_action(state, valid_actions)
    print(f"  选择动作: {ACTIONS[action]}")
    
    new_pos = rl.apply_action(agv_pos, action)
    print(f"  新位置: {new_pos}")
    
    # 测试奖励函数
    print("\n测试奖励函数:")
    r1 = rl.compute_reward((1, 0), (0, 0), (9, 9), False, False, False, False)
    print(f"  接近目标: {r1}")
    r2 = rl.compute_reward((9, 9), (8, 9), (9, 9), True, False, False, False)
    print(f"  到达目标: {r2}")
    r3 = rl.compute_reward((5, 5), (4, 5), (9, 9), False, True, False, False)
    print(f"  碰撞: {r3}")
    r4 = rl.compute_reward((0, 0), (0, 0), (9, 9), False, False, False, True)
    print(f"  等待: {r4}")
    
    # 测试训练模式
    print("\n测试训练模式:")
    rl.set_training(True)
    print(f"  训练模式: {rl.is_training}")
    
    # 模拟经验收集和训练
    print("\n模拟经验收集和训练:")
    for i in range(10):
        # 模拟一条经验
        s = rl.get_state((i, 0), (9, 9), grid, [], [])
        a = 3  # right
        ns = rl.get_state((i+1, 0), (9, 9), grid, [], [])
        r = rl.compute_reward((i+1, 0), (i, 0), (9, 9), False, False, False, False)
        done = (i+1 == 9)
        rl.store_experience(s, a, r, ns, done)
    
    print(f"  经验池大小: {len(rl.memory)}")
    
    if HAS_TORCH:
        loss = rl.train_step()
        print(f"  训练损失: {loss}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    run_test()
