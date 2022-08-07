"""
Methods for agent interaction using sapai-gym and stable basline 3
"""
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sapai_gym import SuperAutoPetsEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sapai_gym.opponent_gen.opponent_generators import random_opp_generator, biggest_numbers_horizontal_opp_generator
from sapai_gym.ai import baselines
from sapai import *
from sapai.shop import *
from image_detection import *
from actions import *
import keyboard

def pause():
    while True:
        if keyboard.read_key() == 'space':
            # If you put 'space' key
            # the program will resume.
            break

def get_action_name(k: int) -> str:
    name_val = list(SuperAutoPetsEnv.ACTION_BASE_NUM.items())

    assert k >= 0
    for (start_name, _), (end_name, end_val) in zip(name_val[:-1], name_val[1:]):
        if k < end_val:
            return start_name
    else:
        return end_name

def remove_nothing(pet_list):
    pets = []
    for i in pet_list:
        if i != 'nothing':
            pets.append(i)
    return pets

def opponent_generator(num_turns):
    # Returns teams to fight against in the gym 
    opponents = biggest_numbers_horizontal_opp_generator(25)
    return opponents

def create_new_PPO():
    env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
    model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
    model.learn(5000)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

    model.save("ppo_sapai_070822")

def run():
    interface = SuperAutoPetsMouse()
    action_dict = interface.actionDict()

    create_new_PPO()
    model = MaskablePPO.load("ppo_sapai_070822")
    #model = MaskablePPO.load("ppo_sapai_3")

    env = SuperAutoPetsEnv(opponent_generator, valid_actions_only=True)
    obs = env.reset()

    num_turns = 20
    while num_turns:
        pause()
        pets, _ = find_the_animals(directory = '.\\SAP_res\\')
        pets = remove_nothing(pets)
        env.player.shop = Shop(pets)
        if env.player.lives <= 3:
            env.player.lives += 3
        action_masks = get_action_masks(env)
        obs = env._encode_state()
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        s = env._avail_actions()
        # print(s[action][1:])
        pause()
        print("Action")
        print(action)
        print(get_action_name(action))
        print(s[action][0])
        print(s[action][1:])
        if env._is_valid_action(action):
            if get_action_name(action) == 'buy_food':
                num_pets = 0
                num_food = 0
                for shop_slot in env.player.shop:
                    if shop_slot.slot_type == "pet":
                        num_pets += 1
                    if shop_slot.slot_type == "food":
                        num_food += 1
                action_dict[get_action_name(action)](s[action][1:], num_pets - num_food%2)
            else:
                if get_action_name(action) == 'roll':
                    action_dict[get_action_name(action)]()
                else:
                    action_dict[get_action_name(action)](s[action][1:])
        obs, reward, done, info = env.step(action)
        if get_action_name(action) == 'end_turn':
            num_turns -= 1
            pause()
        # if done:
        #     obs = env.reset()
        #     break
    print(s[action][0])
    env.close()

if __name__ == "__main__":
    pause()
    run()
