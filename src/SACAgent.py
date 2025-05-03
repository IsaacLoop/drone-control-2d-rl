# SAC Agent

import random
from collections import deque

from src.ReplayMemory import LSTMReplayMemory

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Apparently typical values for SAC
LOG_STD_MIN, LOG_STD_MAX = -20, 2


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, lstm_hidden_dim):
        super().__init__()

        self.lstm_hidden_dim = lstm_hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(256, self.lstm_hidden_dim, batch_first=True)

        self.μ_layer = nn.Linear(self.lstm_hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(self.lstm_hidden_dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]
    ]:

        is_sequence = state.dim() == 3
        if not is_sequence:
            # (batch, features) -> (batch, 1, features)
            state = state.unsqueeze(1)

        x = self.fc(state)

        lstm_out, new_hidden = self.lstm(x, hidden)

        if not is_sequence:
            lstm_out = lstm_out.squeeze(1)

        μ = self.μ_layer(lstm_out)
        log_std = torch.clamp(self.log_std_layer(lstm_out), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        dist = torch.distributions.Normal(μ, std)
        u = dist.rsample()
        a = torch.tanh(u)

        log_prob = dist.log_prob(u)
        log_prob = log_prob.sum(-1, keepdim=True)
        log_prob -= torch.log(1.0 - a.pow(2) + 1e-6).sum(-1, keepdim=True)

        deterministic_action = torch.tanh(μ)

        # `a` shape: (batch, seq_len, action_dim) or (batch, action_dim)
        # `log_prob` shape: (batch, seq_len, 1) or (batch, 1)
        # `deterministic_action` shape: (batch, seq_len, action_dim) or (batch, action_dim)
        # `new_hidden` shape: tuple of (h, c), each shape (1, batch, hidden_dim)
        return a, log_prob, deterministic_action, new_hidden


class Critic(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        lstm_hidden_dim,
    ):
        super().__init__()
        self.lstm_hidden_dim = lstm_hidden_dim

        self.input_layer = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(256, self.lstm_hidden_dim, batch_first=True)

        self.output_layer = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        is_sequence = state.dim() == 3
        if not is_sequence:
            state = state.unsqueeze(1)
            action = action.unsqueeze(1)

        x = torch.cat([state, action], dim=-1)

        x = self.input_layer(x)

        lstm_out, new_hidden = self.lstm(x, hidden)

        if not is_sequence:
            lstm_out = lstm_out.squeeze(1)

        q_value = self.output_layer(lstm_out)

        # `q_value` shape: (batch, seq_len, 1) or (batch, 1)
        # `new_hidden` shape: tuple of (h, c), each shape (1, batch, hidden_dim)
        return q_value, new_hidden


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lstm_hidden_dim: int = 128,
        memory_episodes_capacity: int = 10_000,
        seq_len: int = 32,
        batch_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float | None = None,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.memory_episodes_capacity = memory_episodes_capacity
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.actor = Actor(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
        ).to(self.device)
        self.q1 = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
        ).to(self.device)
        self.q2 = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
        ).to(self.device)

        self.q1_target = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
        ).to(self.device)
        self.q2_target = Critic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
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

        self.memory = LSTMReplayMemory(n_episodes=self.memory_episodes_capacity)
        self.train_step_counter = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(
        self,
        state: torch.Tensor | np.ndarray,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        deterministic: bool,
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor]]:
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if hidden:
            h, c = hidden
            hidden = (h.to(self.device), c.to(self.device))

        with torch.no_grad():
            a, _, a_det, new_hidden = self.actor(state, hidden)

        action_to_take = a_det if deterministic else a
        action_np = action_to_take.cpu().numpy().squeeze(0)

        new_h, new_c = new_hidden
        new_hidden_detached = (new_h.detach(), new_c.detach())

        return action_np, new_hidden_detached

    def update(self):
        if len(self.memory) < self.batch_size:
            print(f"Waiting for {self.batch_size} episodes, have {len(self.memory)}")
            return None

        try:
            batch = self.memory.sample(self.batch_size, self.seq_len)
        except ValueError as e:
            print(f"Skipping update: {e}")
            return None

        sequences = {}
        h0s = {}
        c0s = {}
        for i in range(0, len(batch), self.seq_len):
            seq_idx = i // self.seq_len
            sequence_steps = batch[i : i + self.seq_len]
            sequences[seq_idx] = sequence_steps
            h0s[seq_idx] = (
                torch.from_numpy(sequence_steps[0].h0_actor).float().squeeze(1)
            )
            c0s[seq_idx] = (
                torch.from_numpy(sequence_steps[0].c0_actor).float().squeeze(1)
            )

        s_list = [[step.state for step in sequences[i]] for i in range(self.batch_size)]
        a_list = [
            [step.action for step in sequences[i]] for i in range(self.batch_size)
        ]
        r_list = [
            [step.reward for step in sequences[i]] for i in range(self.batch_size)
        ]
        s2_list = [
            [step.next_state for step in sequences[i]] for i in range(self.batch_size)
        ]
        d_list = [[step.done for step in sequences[i]] for i in range(self.batch_size)]

        s = torch.FloatTensor(np.array(s_list)).to(self.device)
        a = torch.FloatTensor(np.array(a_list)).to(self.device)
        if a.dim() == 2:
            a = a.unsqueeze(-1)
        r = torch.FloatTensor(np.array(r_list)).unsqueeze(-1).to(self.device)
        s2 = torch.FloatTensor(np.array(s2_list)).to(self.device)
        d = torch.FloatTensor(np.array(d_list)).unsqueeze(-1).to(self.device)

        h0 = (
            torch.stack([h0s[i] for i in range(self.batch_size)], dim=0)
            .permute(1, 0, 2)
            .to(self.device)
        )
        c0 = (
            torch.stack([c0s[i] for i in range(self.batch_size)], dim=0)
            .permute(1, 0, 2)
            .to(self.device)
        )
        initial_hidden = (h0, c0)

        with torch.no_grad():
            _, _, _, hidden_s = self.actor(s, initial_hidden)
            a2, log_prob_2, _, _ = self.actor(s2, hidden_s)
            q1_target_pred, _ = self.q1_target(s2, a2, hidden_s)
            q2_target_pred, _ = self.q2_target(s2, a2, hidden_s)
            q_min = torch.min(q1_target_pred, q2_target_pred)

        y = r + self.gamma * (1.0 - d) * (q_min - self.alpha * log_prob_2)

        q1_pred, _ = self.q1(s, a, initial_hidden)
        q2_pred, _ = self.q2(s, a, initial_hidden)

        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        nn.utils.clip_grad_norm_(self.q1.parameters(), self.max_grad_norm)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        nn.utils.clip_grad_norm_(self.q2.parameters(), self.max_grad_norm)
        self.q2_optimizer.step()

        for p in self.q1.parameters():
            p.requires_grad = False
        for p in self.q2.parameters():
            p.requires_grad = False

        a_pred, log_prob, _, _ = self.actor(s, initial_hidden)

        q1_a, _ = self.q1(s, a_pred, initial_hidden)
        q2_a, _ = self.q2(s, a_pred, initial_hidden)
        q_min = torch.min(q1_a, q2_a)

        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        for p in self.q1.parameters():
            p.requires_grad = True
        for p in self.q2.parameters():
            p.requires_grad = True

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for p, p_target in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_target.data.mul_(1.0 - self.tau)
                p_target.data.add_(self.tau * p.data)
            for p, p_target in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_target.data.mul_(1.0 - self.tau)
                p_target.data.add_(self.tau * p.data)

        self.train_step_counter += 1
        return (q1_loss + q2_loss).item() / 2.0
