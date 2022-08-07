from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.logger import configure
from sapai_gym import SuperAutoPetsEnv
from sapai_gym.opponent_gen.opponent_generators import random_opp_generator, biggest_numbers_horizontal_opp_generator
from tqdm import tqdm
import numpy as np

def opponent_generator(num_turns):
    # Returns teams to fight against in the gym 
    opponents = biggest_numbers_horizontal_opp_generator(25)
    return opponents

def train_with_masks(nb_timesteps: int, nb_games: int):
    # initialize environment
    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)

    # setup logger
    logger = configure("./history/sb3_log/")

    # train
    try:
        model = MaskablePPO("MlpPolicy", env, verbose=1)
        model.set_logger(logger)
        model.learn(total_timesteps=nb_timesteps)
        evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=1, warn=False)
        obs = env.reset()
    except AssertionError as e:
        print(e)
    
    # predict
    obs = env.reset()
    rewards = []
    for i in tqdm(range(nb_games), "Games:"):
        # Predict outcome with model
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        print("---")
        print(obs)
        print(reward)
        print(info)
        print()
        rewards.append(reward)
    print(sum(rewards), len(rewards), np.mean(rewards))
    env.close()

if __name__ == "__main__":
    train_with_masks(nb_timesteps=1e6, nb_games=1e2)
