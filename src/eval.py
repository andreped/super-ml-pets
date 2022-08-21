"""
Script for evaluating trained RL model. Run battles of new models against
older models, record battle results, and calculate metrics.
"""

from tensorflow.keras.models import load_model
from utils import Data
import sap
from tqdm import tqdm
import numpy as np
from collections import deque


class Evaluate:
    """
    @TODO: Not finished yet!
    """
    def __init__(best_model_path: str, target_model_path: str, test_episodes: int):
        self.best_model_path = best_model_path
        self.target_model_path = target_model_path
        self.test_episodes = test_episodes

    def load_models(self):
        self.best_model = load_model(self.best_model_path, compile=False)
        self.target_model = load_model(self.target_model_path, compile=False)

    def setup(self):
        pass

    def battle(self):
        pass


def apply(best_model_path: str, target_model_path: str, test_episodes: int):
    """
    method for evaluating agent in simulated Super Auto Pets environment
    """
    # load models
    best_model = load_model(best_model_path, compile=False)
    target_model = load_model(target_model_path, compile=False)

    # initialize environment
    data = Data()
    env = sap.SAP(data)
    print(env)
    exit()
    replay_memory = deque(maxlen=50_000)
    verbose_step = 1

    for episode in tqdm(range(test_episodes), "Iter:"):
        total_training_rewards = 0
        env = sap.SAP(data)
        observation = env.get_scaled_state()
        done = False

        while not done:
            # Exploit best known action
            # model dims are (batch, env.observation_space.n)
            encoded = observation
            encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
            action = best_model.predict(encoded_reshaped, verbose=0).flatten()

            env.step(action)

            action = np.argmax(action)

            new_observation = env.get_scaled_state()
            reward = env.score
            done = env.isGameOver()
            info = None
            replay_memory.append([observation, action, reward, new_observation, done])

            observation = new_observation
            total_training_rewards += reward

            if done:
                if episode % verbose_step == 0:
                    print('After n steps = {} : Total training rewards: {} '.format(episode, total_training_rewards))
                total_training_rewards += 1
                break

        data.total_wins += env.wins
        data.total_losses += env.losses
        data.total_draws += env.draws

        if episode % verbose_step == 0:
            print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)


if __name__ == "__main__":
    best_model_path = "./ckpt/ckpt-63100"
    target_model_path = "./ckpt/ckpt-47000"
    apply(best_model_path, target_model_path, test_episodes=100)
