"""
Script for evaluating trained RL model. Run battles of new models against
older models, record battle results, and calculate metrics.
"""

from utils import opponent_generator
import sapai
from tqdm import tqdm
import numpy as np
from collections import deque
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sapai_gym import SuperAutoPetsEnv
from sapai_gym.opponent_gen.opponent_generators import random_opp_generator, biggest_numbers_horizontal_opp_generator


class Evaluate:
    """
    @TODO: Not finished yet!
    """

    def __init__(self, best_model_path: str, target_model_path: str, test_episodes: int):
        self.best_model_path = best_model_path
        self.target_model_path = target_model_path
        self.test_episodes = test_episodes
        self.best_model = None
        self.target_model = None
        self.env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)

    def load_models_and_set_env(self):
        self.best_model = MaskablePPO.load(self.best_model_path, compile=False)
        self.target_model = MaskablePPO.load(self.target_model_path, compile=False)

        self.best_model.set_env(self.env)
        self.target_model.set_env(self.env)

    def battle(self):
        for episode in tqdm(range(self.test_episodes), "Iter:"):
            total_training_rewards = 0
            observation = self.env.get_scaled_state()
            verbose_step = 10
            done = False

            while not done:
                # Exploit best known action
                # model dims are (batch, env.observation_space.n)
                encoded = observation
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                action = self.best_model.predict(encoded_reshaped, verbose=0).flatten()

                self.env.step(action)

                action = np.argmax(action)
                new_observation = self.env.get_scaled_state()
                reward = self.env.score
                done = self.env.isGameOver()
                info = None
                # replay_memory.append([observation, action, reward, new_observation, done])

                observation = new_observation
                total_training_rewards += reward

                if done:
                    if episode % verbose_step == 0:
                        print(
                            'After n steps = {} : Total training rewards: {} '.format(episode, total_training_rewards))
                    total_training_rewards += 1
                    break

            #data.total_wins += self.env.wins
            #data.total_losses += self.env.losses
            #data.total_draws += self.env.draws

            #if episode % verbose_step == 0:
            #    print("stats: ", data.total_wins, "/", data.total_draws, "/", data.total_losses)


def apply(best_model_path: str, target_model_path: str, test_episodes: int):
    """
    method for evaluating agent in simulated Super Auto Pets environment
    """
    # load models
    evaluator = Evaluate(best_model_path, target_model_path, 100)
    evaluator.load_models_and_set_env()
    evaluator.battle()


if __name__ == "__main__":
    model_path = "./best_models/model_sap_gym_sb3_280822_finetuned_641057_steps"
    target_path = "./best_models/model_sap_gym_sb3_200822_422718_steps"
    #apply(model_path, target_path, test_episodes=100)
