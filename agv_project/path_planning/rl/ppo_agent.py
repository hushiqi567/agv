"""PPO Agent — Actor-Critic with PPO-Clip objective"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

        self.actor = nn.Linear(256, num_actions)
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

        self.buffer = []

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
        if len(self.buffer) < self.batch_size:
            return None

        locals = np.array([t[0] for t in self.buffer])
        gvecs = np.array([t[1] for t in self.buffer])
        actions = torch.LongTensor([t[2] for t in self.buffer]).to(self.device)
        rewards = [t[3] for t in self.buffer]
        dones = [t[4] for t in self.buffer]
        old_log_probs = torch.FloatTensor([t[5] for t in self.buffer]).to(self.device)

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
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.network.load_state_dict(ckpt['network'])
