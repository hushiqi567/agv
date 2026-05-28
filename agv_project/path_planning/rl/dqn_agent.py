"""升级版 DQN Agent — 15×15×5 局部网格 + 4维全局特征 → 5 动作 Q 值"""
import os
import random
import logging
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

# 奖励常量 — 平衡学习信号强度和loss数值稳定性
REWARD_GOAL_ARRIVED = 80.0
REWARD_TASK_COMPLETE = 150.0
REWARD_APPROACH = 1.0
REWARD_AWAY = -1.0
REWARD_STEP = -0.1
REWARD_OBSTACLE_COLLISION = -10.0
REWARD_AGV_COLLISION = -20.0
REWARD_LOAD_SUCCESS = 10.0
REWARD_UNLOAD_SUCCESS = 10.0
REWARD_LOADING_TIMEOUT = -0.5
REWARD_DEADLOCK = -50.0
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

        conv_out = 64 * 4 * 4

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
        """分段奖励函数。电量作为步惩罚乘数：电量越低，每步代价越大。"""
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

        # 步惩罚随电量降低而增大: 100%电量=-0.1, 0%电量=-0.5
        battery_factor = 1.0 + (1.0 - battery / 100.0) * 4.0
        reward = REWARD_STEP * battery_factor

        if curr_dist < prev_dist:
            reward += REWARD_APPROACH
        elif curr_dist > prev_dist:
            reward += REWARD_AWAY
        if waited:
            reward += REWARD_LOADING_TIMEOUT
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
