import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.reset()

for _ in range(1000):
	env.render()
	env.step(env.action_space.sample())
env.close()