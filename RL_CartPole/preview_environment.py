import gym

env = gym.make('CartPole-v1')
env.reset()

for _ in range(2000):
    env.render()
    s, r, done, info = env.step(env.action_space.sample())
    if _ % 300 == 0:
        env.reset()

env.close()
print('done')
