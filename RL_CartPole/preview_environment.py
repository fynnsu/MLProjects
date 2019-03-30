import gym

env = gym.make('CartPole-v1')
env.reset()

for _ in range(2000):
    env.render()
    s, r, done, info = env.step(env.action_space.sample())
    if _ % 300 == 0:
        # restarts environment occasionally
        # Note: the game is usually terminated long before restart
        # as the action game mecanism return done when the pole tilts
        # more than 15 degrees from vertical.
        env.reset()

env.close()
print('done')
