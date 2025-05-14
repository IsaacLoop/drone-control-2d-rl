import argparse
import faulthandler
import random
import shutil
from collections.abc import Callable

import numpy as np
from tqdm import tqdm

faulthandler.enable()

from collections import deque
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from src.Episode import AbstractEpisode, StopEpisode, StraightLineEpisode
from src.ReplayMemory import LSTMReplayMemory
from src.SACAgent import SACAgent

N_ENVS = 4
FPS = 30  # Physics rate (env.dt = 1/FPS)
STATE_DIM = 9
ACTION_DIM = 2

LEARNING_STARTS = 5_000
MEMORY_EPISODES_CAPACITY = 1_000
SEQ_LEN = 32
BATCH_SIZE = 64
LR = 3e-4
GAMMA = 0.98
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
    ma_train_rewards: deque[float],
    agent: SACAgent,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best": best_eval,
            "episodes_finished": episodes_finished,
            "ma_losses": list(ma_losses),
            "ma_alphas": list(ma_alphas),
            "ma_train_rewards": list(ma_train_rewards),
            "actor": agent.actor.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "opt_actor": agent.actor_optimizer.state_dict(),
            "opt_q1": agent.q1_optimizer.state_dict(),
            "opt_q2": agent.q2_optimizer.state_dict(),
            "opt_alpha": agent.alpha_optimizer.state_dict(),
            "replay": agent.memory,
        },
        path,
    )


def load_checkpoint(agent: SACAgent, ma_window: int, path: Path):
    if not path.exists():
        values = (
            0,
            -float("inf"),
            0,
            deque(maxlen=ma_window),
            deque(maxlen=ma_window),
            deque(maxlen=ma_window),
        )
        print(
            f"\nNo checkpoint found at {path}, starting with those default values: ",
            values,
        )
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
    agent.memory = data["replay"]
    return (
        int(data["step"]),
        float(data["best"]),
        int(data["episodes_finished"]),
        deque(data["ma_losses"], maxlen=ma_window),
        deque(data["ma_alphas"], maxlen=ma_window),
        deque(data.get("ma_train_rewards", []), maxlen=ma_window),
    )


def evaluate(agent: SACAgent, gui: bool) -> tuple[list[float], float]:
    all_rewards = []
    total_reward = 0.0

    for episode_factory in EPISODE_FACTORIES:
        episode = episode_factory(gui)
        hidden = None

        while not episode.done:
            state = episode.state
            action, next_hidden = agent.select_action(state, hidden, deterministic=True)
            hidden = (next_hidden[0].detach(), next_hidden[1].detach())

            _, reward, _ = episode.step(action)
            all_rewards.append(reward)
            total_reward += reward

    avg_total_reward = total_reward / len(EPISODE_FACTORIES)
    return all_rewards, avg_total_reward


def train(
    learning_starts: int,
    memory_episodes_capacity: int,
    seq_len: int,
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
        memory_episodes_capacity=memory_episodes_capacity,
        seq_len=seq_len,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        tau=tau,
        target_entropy=target_entropy,
        device=device,
    )

    if resume:
        (
            step,
            best_eval,
            episodes_finished,
            ma_losses,
            ma_alphas,
            ma_train_rewards,
        ) = load_checkpoint(agent, ma_window, BACKUP_DIR / "ckpt.pth")
    else:
        step = 0
        best_eval = -float("inf")
        episodes_finished = 0
        ma_losses = deque(maxlen=ma_window)
        ma_alphas = deque(maxlen=ma_window)
        ma_train_rewards = deque(maxlen=ma_window)

    if TENSORBOARD_DIR.exists() and not resume:
        shutil.rmtree(TENSORBOARD_DIR)
    writer = SummaryWriter(TENSORBOARD_DIR)

    episodes: list[AbstractEpisode] = [
        random.choice(EPISODE_FACTORIES)(gui=False) for _ in range(N_ENVS)
    ]
    h_states = [
        torch.zeros(1, 1, agent.lstm_hidden_dim).to(DEVICE) for _ in range(N_ENVS)
    ]
    c_states = [
        torch.zeros(1, 1, agent.lstm_hidden_dim).to(DEVICE) for _ in range(N_ENVS)
    ]
    episodes_cum_reward = [0.0] * N_ENVS
    episode_ids = list(range(N_ENVS))
    next_episode_id = N_ENVS

    bar = tqdm(desc="Train", unit="t", initial=step, smoothing=0.01)

    while True:
        step += 1
        bar.update(1)

        for i in range(N_ENVS):
            e = episodes[i]
            current_episode_id = episode_ids[i]
            state = e.state

            h_prev = h_states[i].detach().cpu().numpy()
            c_prev = c_states[i].detach().cpu().numpy()

            current_hidden = (h_states[i], c_states[i])
            if step > learning_starts:
                action, next_hidden = agent.select_action(
                    state, current_hidden, deterministic=False
                )
            else:
                action = np.random.uniform(-1, 1, size=ACTION_DIM)
                with torch.no_grad():
                    _, _, _, next_hidden = agent.actor(
                        torch.from_numpy(state).float().unsqueeze(0).to(DEVICE),
                        current_hidden,
                    )

            next_state, reward, done = e.step(action)

            agent.memory.push(
                episode_id=current_episode_id,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                h0_actor=h_prev,
                c0_actor=c_prev,
            )

            h_states[i] = next_hidden[0].detach()
            c_states[i] = next_hidden[1].detach()

            if done:
                ma_train_rewards.append(episodes_cum_reward[i] / e.duration_steps)
                episodes_finished += 1

                episodes[i] = random.choice(EPISODE_FACTORIES)(gui=False)
                episodes_cum_reward[i] = 0.0
                h_states[i] = torch.zeros(1, 1, agent.lstm_hidden_dim).to(DEVICE)
                c_states[i] = torch.zeros(1, 1, agent.lstm_hidden_dim).to(DEVICE)

                episode_ids[i] = next_episode_id
                next_episode_id += 1
            else:
                episodes_cum_reward[i] += reward

        if step > learning_starts:
            q_loss = agent.update()
            ma_losses.append(q_loss)
            alpha = torch.exp(agent.log_alpha.detach()).item()
            ma_alphas.append(alpha)

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
            if len(ma_train_rewards) == ma_window:
                current_ma_train_reward = np.mean(list(ma_train_rewards))
                if current_ma_train_reward > best_eval:
                    best_eval = current_ma_train_reward
                    print(f"\nNew best training MA reward: {best_eval:.6f} at step {step}. Saving best model.")
                    save_checkpoint(
                        step,
                        best_eval,
                        episodes_finished,
                        ma_losses,
                        ma_alphas,
                        ma_train_rewards,
                        agent,
                        BACKUP_DIR / "best_ckpt.pth",
                    )

        # Save checkpoint
        if step % save_interval_steps == 0:
            save_checkpoint(
                step,
                best_eval,
                episodes_finished,
                ma_losses,
                ma_alphas,
                ma_train_rewards,
                agent,
                BACKUP_DIR / "ckpt.pth",
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
            writer.add_scalars(
                "t/train_reward_ma",
                {
                    "mean": np.mean(list(ma_train_rewards)),
                    "min": np.min(list(ma_train_rewards)),
                    "max": np.max(list(ma_train_rewards)),
                },
                step,
            )

        # tqdm bar update
        bar.set_postfix(
            {
                "E": f"{episodes_finished:,}",
                "S": f"{step:,}",
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
        memory_episodes_capacity=MEMORY_EPISODES_CAPACITY,
        seq_len=SEQ_LEN,
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
