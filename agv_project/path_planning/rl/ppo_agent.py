"""PPO Agent — Actor-Critic, 兼容 DQNAgent 接口"""
import os
import random
import logging
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

class PPONetwork(nn.Module):
    def __init__(self, grid_size=15, num_channels=5, global_dim=4,
                 num_actions=5):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        conv_out = 64 * 4 * 4
        self.fc_global = nn.Linear(global_dim, 32)
        self.fc_shared = nn.Linear(conv_out + 32, 256)
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, local_grid, global_vec):
        x = F.relu(self.bn1(self.conv1(local_grid)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        g = F.relu(self.fc_global(global_vec))
        h = F.relu(self.fc_shared(torch.cat([x, g], dim=1)))
        return self.actor(h), self.critic(h)

class PPOAgent:
    """PPO Agent — 存储 old_log_prob 在动作选择时，修复 ratio 始终为1的bug"""

    def __init__(self, grid_size=15, gamma=0.99, lr=3e-4,
                 epsilon_start=0.3, epsilon_end=0.02, epsilon_decay=0.999,
                 batch_size=128, memory_size=4096, target_update=200,
                 use_gpu=False, clip_epsilon=0.2, entropy_coef=0.02,
                 value_coef=0.5):
        self.logger = logging.getLogger("AGVProject.PPOAgent")
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.network = PPONetwork(grid_size=grid_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.encoder = StateEncoder(grid_size=grid_size)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.batch_size = batch_size
        self.target_update = target_update
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self.loss_history = []
        self.buffer = []

    def reset_exploration(self, epsilon=None):
        self.epsilon = epsilon if epsilon is not None else 0.3

    def select_action(self, local_grid, global_vec, valid_actions=None):
        """返回 (action, old_log_prob, value) 三元组"""
        if valid_actions is None:
            valid_actions = list(range(NUM_ACTIONS))
        if not valid_actions:
            return 4, 0.0, 0.0

        if random.random() < self.epsilon:
            action = random.choice(valid_actions)
            with torch.no_grad():
                lg = torch.FloatTensor(local_grid).unsqueeze(0).to(self.device)
                gv = torch.FloatTensor(global_vec).unsqueeze(0).to(self.device)
                logits, value = self.network(lg, gv)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
                log_prob = np.log(probs[action] + 1e-8)
                return action, float(log_prob), value.item()

        with torch.no_grad():
            lg = torch.FloatTensor(local_grid).unsqueeze(0).to(self.device)
            gv = torch.FloatTensor(global_vec).unsqueeze(0).to(self.device)
            logits, value = self.network(lg, gv)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            mask = np.ones(NUM_ACTIONS) * -1e9
            for a in valid_actions:
                mask[a] = 0
            masked = probs + mask
            action = int(np.argmax(masked))
            log_prob = np.log(probs[action] + 1e-8)
            return action, float(log_prob), value.item()

    def store_experience(self, local, gvec, action, reward, next_local,
                         next_gvec, done, old_log_prob=None, value=None):
        self.buffer.append(
            (local, gvec, action, reward, next_local, next_gvec, done,
             old_log_prob, value))

    def train_step(self):
        """PPO-Clip 更新 — 使用存储的 old_log_prob 确保ratio正确"""
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer[:self.batch_size]
        self.buffer = self.buffer[self.batch_size // 2:]

        locals = np.array([t[0] for t in batch])
        gvecs = np.array([t[1] for t in batch])
        actions = torch.LongTensor([t[2] for t in batch]).to(self.device)
        rewards = [t[3] for t in batch]
        dones = [t[6] for t in batch]
        old_log_probs = torch.FloatTensor(
            [t[7] if t[7] is not None else 0.0 for t in batch]).to(self.device)

        # 计算GAE returns
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if dones[i]:
                gae = 0
            gae = rewards[i] + self.gamma * gae
            returns.insert(0, gae)
        returns = torch.FloatTensor(returns).to(self.device)

        lg = torch.FloatTensor(locals).to(self.device)
        gv = torch.FloatTensor(gvecs).to(self.device)

        total_loss = 0
        for _ in range(4):
            logits, values = self.network(lg, gv)
            values = values.squeeze(-1)
            advantages = (returns - values).detach()
            advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8)

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns)
            entropy = dist.entropy().mean()
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                           max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        self.epsilon = max(self.epsilon_end,
                          self.epsilon * self.epsilon_decay)
        self.steps_done += 1
        avg_loss = total_loss / 4
        self.loss_history.append(avg_loss)
        return avg_loss

    def set_training(self, training):
        if training:
            self.network.train()
        else:
            self.network.eval()
            self.epsilon = self.epsilon_end

    def save_model(self, path):
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
            self.network.load_state_dict(ckpt['network'])
            if 'optimizer' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            self.epsilon = ckpt.get('epsilon', self.epsilon_end)
            self.steps_done = ckpt.get('steps_done', 0)
