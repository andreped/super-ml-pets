"""
Main application to train SAIP
"""

import sys, os
from collections import deque
import time
import random
import pickle, csv

import tensorflow as tf
import numpy as np
from tensorflow import keras

import sap

class Data():
    # Save the teams from every level, refresh every generation to fight against
    past_teams = [[]]

    total_wins = 0
    total_losses = 0
    total_draws = 0

    logs = []

# An episode a full game
start_episode = 0
train_episodes = 100000
test_episodes = 100

def agent(state_shape, action_shape):
    """ The agent maps X-states to Y-actions
    e.g. The neural network output is [.1, .7, .1, .3]
    The highest value 0.7 is the Q-Value.
    The index of the highest action (0.7) is action #1.
    """
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
        Y.append(current_qs)
    model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)


def save_logs(data):
    with open('past_teams', 'w', newline='') as f:
        a = csv.writer(f)
        a.writerows(data.past_teams)

    with open('past_teams_bin', 'wb') as f:
        pickle.dump(data.past_teams, f)

    with open('logs', 'w', newline='') as f:
        a = csv.writer(f)
        for l in data.logs:
            a.writerow([str(l)])

    data.logs = []

def main():
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    data = Data()

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    if False:
        model = agent((50, ), 69)
    else:
        model = keras.models.load_model('ckpt/ckpt-600000')
        data.total_wins = 2559962
        data.total_draws = 3490824
        data.total_losses = 2574569
        start_episode = 600001

        with open('past_teams_bin', 'rb') as f:
            data.past_teams = pickle.load(f)

        print("loaded")

    
    # Target Model (updated every 100 steps)
    target_model = agent((50, ), 69)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    target_update_counter = 0

    # X = states, y = actions
    X = []
    y = []

    steps_to_update_target_model = 0

    try:
        for episode in range(start_episode, start_episode+train_episodes):
            total_training_rewards = 0
            env = sap.SAP(data)
            observation = env.get_scaled_state()
            done = False
            while not done:
                steps_to_update_target_model += 1

                random_number = np.random.rand()
                # 2. Explore using the Epsilon Greedy Exploration Strategy
                if random_number <= epsilon:
                    # Explore
                    action = [0]*69
                    action[np.random.randint(0, 69)] = 1
                else:
                    # Exploit best known action
                    # model dims are (batch, env.observation_space.n)
                    encoded = observation
                    encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                    action = model.predict(encoded_reshaped).flatten()

                env.step(action)

                action = np.argmax(action)

                new_observation = env.get_scaled_state()
                reward = env.score 
                done = env.isGameOver()
                info = None
                replay_memory.append([observation, action, reward, new_observation, done])

                # 3. Update the Main Network using the Bellman Equation
                if steps_to_update_target_model % 4 == 0 or done:
                    train(env, replay_memory, model, target_model, done)

                observation = new_observation
                total_training_rewards += reward

                if done:
                    print('After n steps = {} : Total training rewards: {} '.format(episode, total_training_rewards))
                    total_training_rewards += 1

                    if steps_to_update_target_model >= 100:
                        print('Copying main network weights to the target network weights')
                        target_model.set_weights(model.get_weights())
                        steps_to_update_target_model = 0
                    break

            data.total_wins += env.wins
            data.total_losses += env.losses
            data.total_draws += env.draws
            
            print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)

            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * episode)

            if episode % 100 == 0:
                save_logs(data)
                model.save('ckpt/ckpt-'+str(episode))

    except KeyboardInterrupt:
        print('Interrupted')
        save_logs(data)
        model.save('ckpt/ckpt-'+episode)
        print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    
    save_logs(data)
    model.save('ckpt/ckpt-'+str(episode))

if __name__ == '__main__':
    main()