import jax
import jax.numpy as jnp
import gymnasium as gym
from lite_agents.vpg import VPGAgent, ReplayBuffer
import gym_explore
import matplotlib.pyplot as plt


# SETUP
key = jax.random.PRNGKey(19)
env = gym.make('Escaper-v0')
buffer = ReplayBuffer(
    capacity=4096,
    obs_shape=env.observation_space.shape,
    act_shape=env.action_space.shape,
    num_act=env.action_space.n,
)
agent = VPGAgent(
    seed=19,
    obs_shape=env.observation_space.shape,
    act_shape=env.action_space.shape,
    num_act=env.action_space.n,
)
params = agent.init_params()
agent.init_optimizer(params)

num_epochs = 100
ep, ep_return = 0, 0
deposit_return, average_return = [], []
pobs, _ = env.reset()
for e in range(num_epochs):
    for st in range(buffer.capacity):
        act, logp = agent.make_decision(
            params,
            jnp.expand_dims(pobs, axis=0),
        )
        # print(act, logp)
        nobs, rew, term, trunc, _ = env.step(int(act))
        buffer.store(pobs, act, rew)
        ep_return += rew
        pobs = nobs
        if term or trunc:
            buffer.finish_episode()
            deposit_return.append(ep_return)
            average_return.append(sum(deposit_return) / len(deposit_return))
            print(f"episode: {ep+1}, steps: {st+1}, return: {ep_return}")
            ep += 1
            ep_return = 0
            pobs, _ = env.reset()
    buffer.finish_episode()
    replay = buffer.dump()
    # loss_val = loss_fn(params, rep.obs, rep.act, rep.ret)
    params, loss_val = agent.train_epoch(params, replay)
    print(f"\n---epoch {e+1} loss: {loss_val}---\n")
env.close()
plt.plot(average_return)
plt.show()

# VALIDATION
env = gym.make('Escaper-v0', render_mode='human')
pobs, _ = env.reset()
term, trunc = False, False
for _ in range(1000):
    act, qvals = agent.make_decision(
        params,
        jnp.expand_dims(pobs, axis=0),
    )
    nobs, rew, term, trunc, _ = env.step(int(act))
    ep_return += rew
    pobs = nobs
    if term or trunc:
        print(f"\n---return: {ep_return}---\n")
        break
env.close()


