"""升级版 DQN Agent — Double DQN + Huber Loss + 混合状态输入"""
import os
import random
import logging
import math
from typing import List, Tuple, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from path_planning.rl.state_encoder import StateEncoder

ACTIONS = ['up', 'down', 'left', 'right', 'wait']
ACTION_DELTAS = {
    'up': (0, -1), 'down': (0, 1), 'left': (-1, 0),
    'right': (1, 0), 'wait': (0, 0)
}
NUM_ACTIONS = len(ACTIONS)

# 奖励常量 — 使用温和的取值范围保证Huber Loss数值稳定
REWARD_GOAL_ARRIVED = 10.0
REWARD_TASK_COMPLETE = 20.0
REWARD_APPROACH = 0.5
REWARD_AWAY = -0.5
REWARD_STEP = -0.05
REWARD_OBSTACLE_COLLISION = -5.0
REWARD_AGV_COLLISION = -10.0
REWARD_WAIT = -0.3
REWARD_DEADLOCK = -5.0
REWARD_CONGESTION = -1.0
REWARD_BATTERY_LOW = -2.0  # 电量耗尽惩罚


class HybridDQN(nn.Module):
    """混合输入 DQN: 卷积处理局部网格 + 全连接处理全局特征"""

    def __init__(self, grid_size=15, num_channels=5, global_dim=6,
                 num_actions=5, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        conv_out = 64 * 4 * 4  # 1024

        self.fc_global = nn.Linear(global_dim, 32)
        self.fc_combined = nn.Linear(conv_out + 32, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc_out = nn.Linear(hidden_dim, num_actions)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, local_grid, global_vec):
        x = F.relu(self.bn1(self.conv1(local_grid)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        g = F.relu(self.fc_global(global_vec))
        combined = torch.cat([x, g], dim=1)
        h = F.relu(self.fc_combined(combined))
        h = self.dropout(h)
        return self.fc_out(h)


class ReplayMemory:
    """经验回放池 — 支持优先级采样标记"""

    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def push(self, local, gvec, action, reward, next_local, next_gvec, done,
             priority=1.0):
        self.memory.append((local, gvec, action, reward, next_local, next_gvec, done))
        self.priorities.append(priority)

    def sample(self, batch_size):
        indices = random.choices(
            range(len(self.memory)),
            weights=self.priorities if len(self.priorities) > 0 else None,
            k=min(batch_size, len(self.memory)))
        batch = [self.memory[i] for i in indices]
        return zip(*batch)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """升级版 Double DQN Agent"""

    def __init__(self, grid_size=15, gamma=0.99, lr=0.0005,
                 epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.998,
                 batch_size=128, memory_size=200000, target_update=200,
                 use_gpu=False):
        self.logger = logging.getLogger("AGVProject.DQNAgent")
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        self.policy_net = HybridDQN(grid_size=grid_size).to(self.device)
        self.target_net = HybridDQN(grid_size=grid_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr,
                                    weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5000, gamma=0.95)
        self.memory = ReplayMemory(memory_size)

        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.encoder = StateEncoder(grid_size=grid_size)
        self.steps_done = 0
        self.train_steps = 0
        self.loss_history = []
        self.reward_history = []
        self.q_mean_history = []

    def reset_exploration(self, epsilon=None):
        """重置探索率（用于课程学习阶段切换）"""
        self.epsilon = epsilon if epsilon is not None else self.epsilon_start

    def select_action(self, local_grid, global_vec, valid_actions=None):
        """ε-贪心选择动作"""
        if valid_actions is None:
            valid_actions = list(range(NUM_ACTIONS))

        if not valid_actions:
            return 4  # wait

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            lg = torch.FloatTensor(local_grid).unsqueeze(0).to(self.device)
            gv = torch.FloatTensor(global_vec).unsqueeze(0).to(self.device)
            q_values = self.policy_net(lg, gv).cpu().numpy()[0]
            best = max(valid_actions, key=lambda a: q_values[a])
            return best

    def store_experience(self, local, gvec, action, reward, next_local,
                         next_gvec, done, priority=1.0):
        self.memory.push(local, gvec, action, reward, next_local, next_gvec,
                         done, priority)

    def train_step(self):
        """Double DQN 训练步骤 + Huber Loss"""
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

        # 当前Q值
        current_q = self.policy_net(lg_batch, gv_batch).gather(1, act_batch)

        # Double DQN: policy_net选动作, target_net评估
        with torch.no_grad():
            next_actions = self.policy_net(nlg_batch, ngv_batch).argmax(1)
            next_q_all = self.target_net(nlg_batch, ngv_batch)
            next_q = next_q_all.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rew_batch + (1 - done_batch) * self.gamma * next_q

        # Huber Loss (SmoothL1) — 对大TD误差更robust
        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(),
                                       max_norm=5.0)
        self.optimizer.step()
        self.scheduler.step()

        # 衰减探索率
        self.epsilon = max(self.epsilon_end,
                          self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        self.train_steps += 1

        # 软更新目标网络（Polyak averaging）
        if self.train_steps % self.target_update == 0:
            tau = 0.005
            for target_param, policy_param in zip(
                    self.target_net.parameters(),
                    self.policy_net.parameters()):
                target_param.data.copy_(
                    tau * policy_param.data +
                    (1.0 - tau) * target_param.data)

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    @staticmethod
    def compute_reward(prev_pos, curr_pos, goal_pos, is_loaded,
                       arrived_pickup=False, arrived_delivery=False,
                       obstacle_collision=False, agv_collision=False,
                       deadlock=False, congestion_count=0,
                       battery=100.0, waited=False):
        """分段奖励函数 — 温和的取值范围保证训练稳定"""
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

        prev_dist = abs(prev_pos[0] - goal_pos[0]) + abs(
            prev_pos[1] - goal_pos[1])
        curr_dist = abs(curr_pos[0] - goal_pos[0]) + abs(
            curr_pos[1] - goal_pos[1])

        reward = REWARD_STEP

        # 进度奖励 — 使用比例而非绝对值
        if curr_dist < prev_dist:
            progress = (prev_dist - curr_dist) / max(prev_dist, 1)
            reward += REWARD_APPROACH * (1.0 + progress)
        elif curr_dist > prev_dist:
            regress = (curr_dist - prev_dist) / max(curr_dist, 1)
            reward += REWARD_AWAY * (1.0 + regress)

        if waited:
            reward += REWARD_WAIT
        if congestion_count >= 3:
            reward += REWARD_CONGESTION * (congestion_count / 3.0)

        # 电池激励: 低电量时步惩罚倍增
        if battery < 20.0:
            reward *= 4.0
        elif battery < 50.0:
            reward *= 2.0
        if battery <= 0.0:
            reward += REWARD_BATTERY_LOW

        return reward

    def save_model(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'train_steps': self.train_steps,
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.policy_net.load_state_dict(ckpt['policy_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            self.epsilon = ckpt.get('epsilon', self.epsilon_end)
            self.steps_done = ckpt.get('steps_done', 0)
            self.train_steps = ckpt.get('train_steps', 0)

    def set_training(self, training):
        if training:
            self.policy_net.train()
        else:
            self.policy_net.eval()
            self.epsilon = self.epsilon_end
