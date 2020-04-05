from ILQR import ILQR
import gym
import sys
import env
import jax.numpy as np

env = gym.make('CartPoleContinuous-v0').env
obs = env.reset()


def final_cost(x):
    return 0.5 * (np.square(1.0 - np.cos(x[2])) + np.square(x[1]) + np.square(x[3]))


def running_cost(x, u):
    return 0.5 * np.sum(np.square(u))


def model(x, u):
    return env._state_eq(x, u)


horizon = 20
per_iter = 3
u_range = [-env.max_force, env.max_force]

ilqr = ILQR(final_cost, running_cost, model, u_range, horizon, per_iter)

u_seq = [np.zeros(1) for _ in range(horizon)]
x_seq = [obs.copy()]
for t in range(ilqr.horizon):
    x_seq.append(env._state_eq(x_seq[-1], u_seq[t]))

cnt = 0
while True:
    env.render(mode='Human')
    u_seq = ilqr.predict(x_seq, u_seq)

    obs, _, _, _ = env.step(u_seq[0])
    x_seq[0] = obs.copy()
    cnt += 1
