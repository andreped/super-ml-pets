"""
Main application to train RL model in simulated environment
"""

import sys, os
from collections import deque
import time
import random
import pickle, csv

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tqdm import tqdm
import random
import traceback
import numpy as np

import sapai
from sapai import Player
from sapai import Food
from sapai import Team
from sapai.battle import Battle

class Data():
    # Save the teams from every level, refresh every generation to fight against
    past_teams = [[]]

    total_wins = 0
    total_losses = 0
    total_draws = 0

    logs = []

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
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate))  # , metrics=['accuracy'])
    return model

def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]), verbose=0)[0]


class SAP(object):
    def __init__(self, data):
        self.player = Player(pack="StandardPack")
        self.score = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.turns = 1
        self.actions_taken_this_turn = 0
        self.past_teams = data.past_teams
        self.logs = data.logs

    def step(self, action):
        """
        Update the system state using the best of action (0-68)
        """
        action = np.argmax(action)

        self.actions_taken_this_turn += 1

        self.score = 0

        if self.actions_taken_this_turn > 20:
            self.score -= 10

        try:
            if action < 35:
                # buyshop
                tm_idx = int(action/7)
                shp_idx = action % 7
                tm_slot = self.player.team[tm_idx]
                shp_slot = self.player.shop[shp_idx]

                self.score += 1

                # buy pet (always puts in last slot), buy combine
                if shp_slot.slot_type == "pet":
                    if tm_slot.empty:
                        self.player.buy_pet(shp_slot)
                        self.player.team.move(len(self.player.team)-1, tm_idx)
                    else:
                        self.player.buy_combine(shp_slot, tm_slot)
                else:
                    self.player.buy_food(shp_slot, tm_slot)

            elif action < 55:
                # moveteam
                action -= 35
                tm1_idx = int(action/5)
                tm2_idx = action % 5

                self.score -= 1

                if not self.player.team[tm1_idx].empty and self.player.team[tm1_idx].pet.name == self.player.team[tm2_idx].pet.name:
                    self.player.combine(tm2_idx, tm1_idx)
                else:
                    self.player.team.move(tm1_idx, tm2_idx)
            elif action < 60:
                # sellteam
                action -= 55
                tm_slot = self.player.team[action]

                self.score -= 1

                self.player.sell(tm_slot)
            elif action < 67:
                # freezeshop
                action -= 60
                shp_slot = self.player.shop[action]

                self.score -= 1

                self.player.freeze(shp_slot)
            elif action < 68:
                # rollshop
                self.player.roll()

                self.score += 1
            else:
                # endturn
                self.actions_taken_this_turn = 0
                self.player.end_turn()

                prev_team = Team([])
                while len(self.past_teams) <= self.turns:
                    self.past_teams.append([])

                if len(self.past_teams[self.turns]) == 0:
                    self.past_teams[self.turns].append(Team([]))


                if len(self.past_teams[self.turns]) > 500:
                    self.past_teams[self.turns] = self.past_teams[self.turns][150:]

                prev_team = self.past_teams[self.turns][random.randint(
                    0, len(self.past_teams[self.turns])-1)]

                battle = Battle(self.player.team, prev_team)
                winner = battle.battle()

                if winner == 0:
                    self.wins += 1
                    self.score += 50
                elif winner == 1:
                    self.losses += 1
                    self.score += 5
                    if self.turns <= 2:
                        self.player.lives -= 1
                    elif self.turns <= 4:
                        self.player.lives -= 2
                    else:
                        self.player.lives -= 3
                else:
                    self.draws += 1
                    self.score += 20

                self.past_teams[self.turns].append(self.player.team)
                self.turns += 1
        
        except Exception:
            self.logs.append(traceback.format_exc())

            self.score -= 5

    def get_scaled_state(self):
        """
        Get full state, scaled into (approximately) [0, 1].
        State is: 
        team states {id, exp, atk, def, food},
        shop states {id, atk, def},
        money, turn, lives, wins
        """

        state = []
        for teamslot_state in self.player.team.state["team"]:
            pet = teamslot_state["pet"]
            if pet["name"] == "pet-none":
                state.extend([89/len(sapai.data["pets"]), 0, 0, 0, 1])
            else:
                exp = pet["experience"]
                lvl = pet["level"]
                if lvl == 2:
                    exp += 2
                elif lvl == 3:
                    exp = 5

                if pet["status"] != 'none':
                    state.extend([(list(sapai.data["pets"].keys()).index(
                        pet["name"]))/len(sapai.data["pets"]), exp/6, pet["attack"]/50, pet["health"]/50,
                        (list(sapai.data["statuses"].keys()).index(pet["status"]))/(len(sapai.data["statuses"])+1)])

                else:
                    state.extend([(list(sapai.data["pets"].keys()).index(
                        pet["name"]))/len(sapai.data["pets"]), exp/6, pet["attack"]/50, pet["health"]/50,
                        (11)/(len(sapai.data["statuses"])+1)])

        for shopslot_state in self.player.shop.state["shop_slots"]:
            item = shopslot_state["item"]
            if item["name"] == "pet-none" or item["name"] == "food-none":
                state.extend([89/len(sapai.data["pets"]), 0, 0])
            elif item["type"] == "Food":
                state.extend([(list(sapai.data["foods"].keys()).index(item["name"])+len(sapai.data["pets"]))
                              / (len(sapai.data["foods"])+len(sapai.data["pets"])), item["attack"]/50, item["health"]/50])
            else:
                state.extend([(list(sapai.data["pets"].keys()).index(
                            item["name"]))/len(sapai.data["pets"]), item["attack"]/50, item["health"]/50])

        for i in range(7-len(self.player.shop)):
            state.extend([89/len(sapai.data["pets"]), 0, 0])

        state.extend([self.player.gold, self.player.turn,
                     self.player.lives, self.wins])

        return np.array(state)

    def isGameOver(self):
        if self.player.lives <= 0 or self.wins >= 10 or self.turns >= 30 or self.actions_taken_this_turn >= 30:
            return True
        
        return False

def train(env, replay_memory, model, target_model, done):
    learning_rate = 0.7 # Learning rate
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 1000
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states, verbose=0)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states, verbose=0)

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

def main(start_episode, verbose_step):
    epsilon = 1 # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start
    max_epsilon = 1 # You can't explore more than 100% of the time
    min_epsilon = 0.01 # At a minimum, we'll always explore 1% of the time
    decay = 0.01

    data = Data()

    # 1. Initialize the Target and Main models
    # Main Model (updated every 4 steps)
    if not finetune:
        model = agent((50, ), 69)
        start_episode = 1
    else:
        model = keras.models.load_model('ckpt/ckpt-' + str(start_episode))
        data.total_wins = 0
        data.total_draws = 0
        data.total_losses = 0
        #start_episode = 12000

        with open('past_teams_bin', 'rb') as f:
            data.past_teams = pickle.load(f)

        print("loaded")

    
    # Target Model (updated every 100 steps)
    target_model = agent((50, ), 69)
    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=50_000)

    # X = states, y = actions
    X = []
    y = []
    target_update_counter = 0
    steps_to_update_target_model = 0

    try:
        for episode in tqdm(range(start_episode, start_episode + train_episodes), "Iter:"):
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
                    action = [0]*69  # @TODO: Is this OK to do in Python 3.8?
                    action[np.random.randint(0, 69)] = 1
                else:
                    # Exploit best known action
                    # model dims are (batch, env.observation_space.n)
                    encoded = observation
                    encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                    action = model.predict(encoded_reshaped, verbose=0).flatten()

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
                    if episode % verbose_step == 0:
                        print('After n steps = {} : Total training rewards: {} '.format(episode, total_training_rewards))
                    total_training_rewards += 1

                    if steps_to_update_target_model >= 100:
                        # print('Copying main network weights to the target network weights')
                        target_model.set_weights(model.get_weights())
                        steps_to_update_target_model = 0
                    break

            data.total_wins += env.wins
            data.total_losses += env.losses
            data.total_draws += env.draws
            
            if episode % verbose_step == 0:
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

def test_train():
    # An episode a full game
    start_episode = 0
    train_episodes = 20
    test_episodes = 100
    finetune = False  # whether to finetune or not
    verbose_step = 2

    main(start_episode=start_episode, verbose_step=verbose_step)

