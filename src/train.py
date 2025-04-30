import argparse
from collections.abc import Callable
import faulthandler
import random
import shutil

import numpy as np
from tqdm import tqdm

faulthandler.enable()

from src.Episode import AbstractEpisode, StopEpisode, StraightLineEpisode
from src.SACAgent import SACAgent
import torch
from collections import deque
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

N_ENVS = 4  # Parallel episodes processed each step
FPS = 30  # Physics rate (env.dt = 1/FPS)
STATE_DIM = 11  # Fixed state length
ACTION_DIM = 2  # [aL, aR] ∈ [‑1,1]²

LEARNING_STARTS = 5_000
CAPACITY = 100_000
BATCH_SIZE = 256
LR = 3e-4
GAMMA = 0.992
TAU = 0.005
TARGET_ENTROPY = -ACTION_DIM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE} {'🥳' if DEVICE == 'cuda' else '😢'}")

BACKUP_DIR = Path("training/backup")
TENSORBOARD_DIR = Path("training/tb_logs")
SAVE_BACKUP_EVERY = 10_000
TEST_EVERY = 1_000
MA_WINDOW = 1_000

EPISODE_FACTORIES: list[Callable[[bool], AbstractEpisode]] = [
    lambda gui: StopEpisode(duration_steps=FPS * 10, dt=1 / FPS, gui=gui),
    lambda gui: StraightLineEpisode(duration_steps=FPS * 10, dt=1 / FPS, gui=gui),
]


def save_checkpoint(
    step: int,
    best_eval: float,
    episodes_finished: int,
    ma_losses: deque[float],
    ma_alphas: deque[float],
    agent: SACAgent,
):
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best": best_eval,
            "episodes_finished": episodes_finished,
            "ma_losses": list(ma_losses),
            "ma_alphas": list(ma_alphas),
            "actor": agent.actor.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "opt_actor": agent.actor_optimizer.state_dict(),
            "opt_q1": agent.q1_optimizer.state_dict(),
            "opt_q2": agent.q2_optimizer.state_dict(),
            "opt_alpha": agent.alpha_optimizer.state_dict(),
            "replay": list(agent.memory.buffer),
        },
        BACKUP_DIR / "ckpt.pth",
    )


def load_checkpoint(agent: SACAgent, ma_window: int):
    path = BACKUP_DIR / "ckpt.pth"
    if not path.exists():
        values = 0, -float("inf"), 0, deque(maxlen=ma_window), deque(maxlen=ma_window)
        print("\nNo checkpoint found, starting with those default values: ", values)
        print()
        return values
    data = torch.load(path, map_location="cpu", weights_only=False)
    agent.actor.load_state_dict(data["actor"])
    agent.q1.load_state_dict(data["q1"])
    agent.q2.load_state_dict(data["q2"])
    agent.log_alpha.data.copy_(data["log_alpha"])
    agent.actor_optimizer.load_state_dict(data["opt_actor"])
    agent.q1_optimizer.load_state_dict(data["opt_q1"])
    agent.q2_optimizer.load_state_dict(data["opt_q2"])
    agent.alpha_optimizer.load_state_dict(data["opt_alpha"])
    agent.memory.buffer = deque(data["replay"], maxlen=agent.capacity)
    return (
        int(data["step"]),
        float(data["best"]),
        int(data["episodes_finished"]),
        deque(data["ma_losses"], maxlen=ma_window),
        deque(data["ma_alphas"], maxlen=ma_window),
    )


def evaluate(agent: SACAgent, gui: bool) -> tuple[list[float], float]:
    all_rewards = []
    total_reward = 0.0
    for episode_factory in EPISODE_FACTORIES:
        episode = episode_factory(gui)
        while not episode.done:
            state = episode.state
            action = agent.select_action(state, deterministic=True)
            _, reward, _ = episode.step(action)
            all_rewards.append(reward)
            total_reward += reward
    avg_total_reward = total_reward / len(EPISODE_FACTORIES)
    return all_rewards, avg_total_reward


def train(
    learning_starts: int,
    capacity: int,
    batch_size: int,
    lr: float,
    gamma: float,
    tau: float,
    target_entropy: float,
    device: str,
    save_interval_steps: int,
    evaluation_interval_steps: int,
    ma_window: int,
    gui: bool,
    resume: bool = False,
):
    agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        capacity=capacity,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        tau=tau,
        target_entropy=target_entropy,
        device=device,
    )

    if resume:
        step, best_eval, episodes_finished, ma_losses, ma_alphas = load_checkpoint(
            agent, ma_window
        )
    else:
        step = 0
        best_eval = -float("inf")
        episodes_finished = 0
        ma_losses = deque(maxlen=ma_window)
        ma_alphas = deque(maxlen=ma_window)

    if TENSORBOARD_DIR.exists() and not resume:
        shutil.rmtree(TENSORBOARD_DIR)
    writer = SummaryWriter(TENSORBOARD_DIR)

    episodes: list[AbstractEpisode] = [
        random.choice(EPISODE_FACTORIES)(gui=False) for _ in range(N_ENVS)
    ]
    episodes_cum_reward = [0.0] * N_ENVS

    bar = tqdm(desc="Train", unit="t", initial=step, smoothing=0.01)

    while True:
        step += 1
        bar.update(1)

        # Train
        for i, e in enumerate(episodes):
            state = e.state
            if step > learning_starts:
                action = agent.select_action(state, deterministic=False)
            else:
                action = np.random.uniform(-1, 1, size=ACTION_DIM)
            next_state, reward, done = e.step(action)
            agent.store(state, action, reward, next_state, done)
            if done:
                writer.add_scalar(
                    "t/episode_avg_reward",
                    episodes_cum_reward[i] / e.duration_steps,
                    step,
                )
                episodes_finished += 1
                episodes[i] = random.choice(EPISODE_FACTORIES)(gui=False)
                episodes_cum_reward[i] = 0.0
            else:
                episodes_cum_reward[i] += reward

        if step > learning_starts:
            q_loss = agent.update()
            ma_losses.append(q_loss)
            alpha = torch.exp(agent.log_alpha.detach()).item()
            ma_alphas.append(alpha)

        # Evaluate
        if step >= learning_starts and step % evaluation_interval_steps == 0:
            all_rewards, avg_total_reward = evaluate(agent, gui)
            writer.add_scalars(
                "e/reward",
                {
                    "max": np.max(all_rewards),
                    "mean": np.mean(all_rewards),
                    "min": np.min(all_rewards),
                },
                step,
            )
            if avg_total_reward > best_eval:
                best_eval = avg_total_reward

        # Save checkpoint
        if step % save_interval_steps == 0:
            save_checkpoint(
                step, best_eval, episodes_finished, ma_losses, ma_alphas, agent
            )

        # Log to tensorboard
        if len(ma_losses) == ma_window:
            writer.add_scalars(
                "t/loss",
                {
                    "mean": np.mean(ma_losses),
                    "min": np.min(ma_losses),
                    "max": np.max(ma_losses),
                },
                step,
            )
            writer.add_scalars(
                "t/α",
                {
                    "mean": np.mean(ma_alphas),
                    "min": np.min(ma_alphas),
                    "max": np.max(ma_alphas),
                },
                step,
            )

        # tqdm bar update
        bar.set_postfix(
            {
                "E": f"{episodes_finished:,}",
                "S": f"{step:,}",
                "M": f"{len(agent.memory) / capacity:.2f}".rstrip("0").rstrip("."),
                "B": f"{best_eval:.6f}",
                "L": (
                    f"{np.mean(ma_losses):.6f}"
                    if len(ma_losses) == ma_window
                    else "N/A"
                ),
                "α": (
                    f"{np.mean(ma_alphas):.6f}"
                    if len(ma_alphas) == ma_window
                    else "N/A"
                ),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="Disable GUI during training",
        default=False,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the most recent training/backup/ checkpoint if available",
        default=False,
    )
    args = parser.parse_args()

    train(
        learning_starts=LEARNING_STARTS,
        capacity=CAPACITY,
        batch_size=BATCH_SIZE,
        lr=LR,
        gamma=GAMMA,
        tau=TAU,
        target_entropy=TARGET_ENTROPY,
        device=DEVICE,
        save_interval_steps=SAVE_BACKUP_EVERY,
        evaluation_interval_steps=TEST_EVERY,
        ma_window=MA_WINDOW,
        gui=not args.nogui,
        resume=args.resume,
    )
