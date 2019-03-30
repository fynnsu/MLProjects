import numpy as np
import gym
from rl_model import rl_model
from memory import rl_memory

NUM_TRAIN_GAMES = 5000
NUM_TEST_GAMES = 100
NUM_TEST_VISUAL_GAMES = 3
MAX_GAME_STEPS = 500
NUM_RANDOM_GAMES = 50
TEST_FREQUENCY = 200
LOSS_PENALTY = -100
RANDOM_SEED = 0
MEMORY_CAPACITY = 10000000
BATCH_SIZE = 256
GYM_ENVIRONMENT = 'CartPole-v1'
LEARNING_RATE = 3e-3
DISCOUNT_FACTOR = 0.8
START_EPSILON = 1.0
MIN_EPSILON = 0.5
EPSILON_DECAY = 0.999

model = rl_model(LEARNING_RATE, DISCOUNT_FACTOR, RANDOM_SEED)
memory = rl_memory(MEMORY_CAPACITY, BATCH_SIZE, RANDOM_SEED)
env = gym.make(GYM_ENVIRONMENT)

def run_test(render=False, num_games=NUM_TEST_GAMES):
    times = []
    sides = [0, 0]
    for i in range(num_games):
        state = env.reset()
        for j in range(MAX_GAME_STEPS):
            if render:
                env.render()
            action = np.argmax(model.predict([state]), axis=1)[0]
            next_state, reward, done, info = env.step(action)
            sides[action] += 1
            
            state = next_state

            if done:
                break
        times.append(j)
        state = env.reset()
#     print(sides)
    return times

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(model.predict([state]), axis=1)[0]
#         return np.random.choice(2, p=softmax(model.predict([state])[0]))

def run_train():
    loss = 0
    prev_avg_time = 0
    epsilon = START_EPSILON
    for i in range(NUM_TRAIN_GAMES):
        state = env.reset()
        for j in range(MAX_GAME_STEPS):
            action = get_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            

            if done:
                if i > NUM_RANDOM_GAMES and epsilon > MIN_EPSILON:
                    epsilon *= EPSILON_DECAY
                reward = LOSS_PENALTY

            memory.add(state, action, reward, next_state)
            state = next_state

            if done:
                break
        if i > NUM_RANDOM_GAMES and i % 1 == 0:
            loss += model.train(memory.get_batch(BATCH_SIZE))
        if i > NUM_RANDOM_GAMES and i % TEST_FREQUENCY == 0:
            times = run_test()
            avg_time = sum(times)/len(times)
            if avg_time > 200 and int(avg_time) > prev_avg_time:
                prev_avg_time = avg_time
                print(model.save(avg_time))
            print('Game:', i)
            print('Loss:', loss / TEST_FREQUENCY)
            print('Time:', avg_time)
#             print('Epsilon', epsilon)
            print()
            loss = 0

                

run_train()
# run_test(render=True, num_games=NUM_TEST_VISUAL_GAMES)
print(run_test())
env.close()

print("Done")