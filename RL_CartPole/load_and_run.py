import numpy as np
import gym
from rl_model import rl_model

NUM_TRAIN_GAMES = 25000
NUM_TEST_GAMES = 100
NUM_TEST_VISUAL_GAMES = 10
MAX_GAME_STEPS = 500
LOSS_PENALTY = -100
RANDOM_SEED = 0
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 256
GYM_ENVIRONMENT = 'CartPole-v1'
LEARNING_RATE = 1e-2
DISCOUNT_FACTOR = 0.8
START_EPSILON = 1.0
MIN_EPSILON = 0.5
EPSILON_DECAY = 0.999

model = rl_model(LEARNING_RATE, DISCOUNT_FACTOR, RANDOM_SEED)
env = gym.make(GYM_ENVIRONMENT)

def run_test(render=False, num_games=NUM_TEST_GAMES):
    times = []
    sides = [0, 0]
    t = 0
    for i in range(num_games):
        state = env.reset()
        while(True):
            if render:
                env.render()
            action = np.argmax(model.predict([state]), axis=1)[0]
            next_state, reward, done, info = env.step(action)
            sides[action] += 1
            
            state = next_state
            t += 1
            if done:
                break
        times.append(t)
        state = env.reset()
        t = 0
#     print(sides)
    return times


model.load(441)

print(run_test(render=True, num_games=NUM_TEST_VISUAL_GAMES))

env.close()
print("Done")
