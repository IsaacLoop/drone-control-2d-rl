# drone-control-2d-rl

**Author:** Isaac Nivet

---

The project consists in training an agent to control a small drone in a realistic 2D environment that I made. The environment has realistic physics, such as gravity, air drag, torque, etc. It is honestly pretty hard to control such a drone! (You can try it yourself by running `python src/Game.py`, and controlling the drone with `A`/`Q` for the left propeller, and `P`/`M` for the right one.)

This difficulty makes it very interesting for a reinforcement learning (RL) project, since it is very hard to hand-engineer rules that would stabilize a drone running in an unstable air, let alone make it perform a specific movement.

## Environment

I used Python 3.11 and CUDA 12.6. Installations suggestions are in [env/readme.md](env/readme.md).

## Algorithm

The algorithm used is [SAC (Soft Actor-Critic)](https://spinningup.openai.com/en/latest/algorithms/sac.html), which is a popular algorithm for continuous action spaces. It is fairly tricky to understand and to implement, but once it's set up, it's very powerful. Some of the best reasons for using SAC would be:

- Works on continuous action spaces (most important, mandatory criteria)
- Entropy depends on a learnt parameter, which allows the agent to automatically decide to explore more or less depending on where it thinks it is in the training process. That makes it more convenient to use than others, like PPO, DDPG, etc.
- It is an off-policy algorithm, which means that the agent can learn from past experiences, and not just the ones it has experienced very recently. I tend to have a better feeling with off-policy algorithms, they just feel right to me.

## Reward function

RL algorithms need a reward function to learn. A reward function is a function that takes the current state of the environment and the action taken by the agent, and returns a number that represents how good or bad the action was. Typically, the better the action, the higher the reward.

In this project, we used two kinds of training episodes:

- `StopEpisode`: the agent is rewarded for stabilizing the drone, and stop moving.
- `StraightLineEpisode`: the agent is rewarded for moving in a straight line.

Although simple, I was surprised to find that these two kinds of episodes are enough to train a robust drone controller agent.

## Training

To train the agent, run `python src/train.py`. Since RL training does not really have a clear end, training is never ending. It periodically saves a checkpoint of the last model. You can resume a previous training by running `python src/train.py --resume`. I advise to also run it with `--nogui` to avoid the GUI showing the drone during testing, because it slows down the training.

## Results

As of right now, the training is not yet completely finished. However, we can observe after only 6 hours of training that the agent is capable of maintaining the drone upwards, and to move it in a desired direction. It still lacks a perfect stability, for instance the wind will drag it around a bit, and it struggles to move quickly in the desired direction.

You can visualize the performance of the agent by using the notebook `notebooks/test_model.ipynb`, which will load the last checkpoint created by `train.py` during the training process.

## Future work

Training is still ongoing on a server in my house running a RTX 3060, I intend to let it run for a few days before publishing the results.
