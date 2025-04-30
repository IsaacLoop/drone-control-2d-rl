# SAC Agent

import random
from collections import deque

from src.ReplayMemory import ReplayMemory

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Apparently typical values for SAC
LOG_STD_MIN, LOG_STD_MAX = -20, 2


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.μ_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.fc(state)
        μ = self.μ_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = torch.distributions.Normal(μ, std)
        u = dist.rsample()
        a = torch.tanh(u)

        log_prob = dist.log_prob(u).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)

        # Note, for a reader who doesn't know SAC:
        # - a is the action with some stochasticity
        # - log_prob is the "entropy term", needed for updating the actor
        # - torch.tanh(μ) is the deterministic action output by the actor (for test time)
        return a, log_prob, torch.tanh(μ)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.fc(x)


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        batch_size: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float | None = None,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.actor = Actor(state_dim=self.state_dim, action_dim=self.action_dim).to(
            self.device
        )
        self.q1 = Critic(state_dim=self.state_dim, action_dim=self.action_dim).to(
            self.device
        )
        self.q2 = Critic(state_dim=self.state_dim, action_dim=self.action_dim).to(
            self.device
        )

        self.q1_target = Critic(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)
        self.q2_target = Critic(
            state_dim=self.state_dim, action_dim=self.action_dim
        ).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.lr)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        if self.target_entropy is None:
            self.target_entropy = -action_dim

        self.memory = ReplayMemory(capacity=self.capacity)
        self.train_step_counter = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self, state: torch.Tensor | np.ndarray, deterministic: bool
    ) -> torch.Tensor:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            a, _, a_det = self.actor(state)
        a = a_det if deterministic else a
        return a.cpu().numpy()[0]

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.memory.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        s, a, r, s2, d = map(np.stack, zip(*batch))

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            a2, log_prob_2, _ = self.actor(s2)
            q1_t = self.q1_target(s2, a2)
            q2_t = self.q2_target(s2, a2)
            q_min = torch.min(q1_t, q2_t)
            y = r + self.gamma * (1.0 - d) * (q_min - self.alpha * log_prob_2)

        q1_loss = F.mse_loss(self.q1(s, a), y)
        q2_loss = F.mse_loss(self.q2(s, a), y)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
        self.q2_optimizer.step()

        
        a, log_prob, _ = self.actor(s)
        q1_a = self.q1(s, a)
        q2_a = self.q2(s, a)
        q_min = torch.min(q1_a, q2_a)
        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        nn.utils.clip_grad_norm_(self.log_alpha, self.max_grad_norm)
        self.alpha_optimizer.step()
        
        with torch.no_grad():
            for p, p_target in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_target.data.mul_(1 - self.tau).add_((self.tau) * p.data)
            for p, p_target in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_target.data.mul_(1 - self.tau).add_((self.tau) * p.data)

        self.train_step_counter += 1
        return (q1_loss + q2_loss).item() / 2.0
