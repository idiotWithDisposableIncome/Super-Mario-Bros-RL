import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import os
import re
from utils import *
import logging

#from PIL import Image
#look into passing gpu to container - should be fine
#mount docker volume 
# Configure the logging system - docker will manage movement 
#use 
logging.basicConfig(filename='logs/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 1_000
NUM_OF_EPISODES = 10_000
SAVE_FRAMES_INTERVAL = 2
#controllers = [Image.open(f"controllers/{i}.png") for i in range(5)]
#video_save_path = os.path.join("video-", get_current_date_time_string())
#os.makedirs(video_save_path, exist_ok=True)

env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
env.metadata['render.modes'] = ['human','rgb_array']

env = JoypadSpace(env, SIMPLE_MOVEMENT)

env = apply_wrappers(env)
agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

models_dir = "models"
subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]

if subdirs:
    # Find the most recent folder using the date and time in the folder name
    most_recent_folder = max(subdirs, key=lambda x: (re.search(r"\d{4}-\d{2}-\d{2}-\d{2}_\d{2}_\d{2}", x) or "").group())
    most_recent_folder_path = os.path.join(models_dir, most_recent_folder)

    # Check if there is a saved model in the most recent folder
    model_files = [f for f in os.listdir(most_recent_folder_path) if f.endswith('.pt')]

    if model_files:
        # Load the most recent model
        model_files.sort()  # Sort to get the most recent model
        latest_model_path = os.path.join(most_recent_folder_path, model_files[-1])
        agent.load_model(latest_model_path)

        print(f"Resuming training from the most recent model: {latest_model_path}")
    else:
        print(f"No saved models found in the most recent folder. Starting fresh.")
else:
    
    print("No existing models. Starting fresh.")

model_path = os.path.join("models", get_current_date_time_string())
#frames_save_path = os.path.join("frames", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)


if not SHOULD_TRAIN:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

for i in range(NUM_OF_EPISODES):    
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info  = env.step(a)
        total_reward += reward
        #env.record_frame()
        

        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        state = new_state


    logging.info(f"Episode {i + 1}: Total Reward = {total_reward}, Epsilon = {agent.epsilon}, Replay Buffer Size = {len(agent.replay_buffer)}")
    print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

    print("Total reward:", total_reward)

env.close()
