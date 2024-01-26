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
import multiprocessing as mp


#from PIL import Image
#look into passing gpu to container - should be fine
#mount docker volume 
# Configure the logging system - docker will manage movement 
#use 
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def environment_worker(env_id, pipe):
    try:
        print("Worker started")
        env = gym_super_mario_bros.make(env_id, render_mode= 'rgb_array', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = apply_wrappers(env)

        print("Environment initialized")

        state = env.reset()
        pipe.send((state))  # Send initial state to the main process

        print("Environment reset")  # Confirm environment reset
        while True:
            action = pipe.recv()  # Receive action from the main process
            if action is None:    # None action indicates shutdown
                break
            next_state, reward, done, trunc, info = env.step(action)
            if done:
                next_state, _ = env.reset()
            # Send observation, reward, and done flag back to the main process
            pipe.send((next_state, reward, done, trunc, info))
    except Exception as e:
        print(f"Worker encountered an error: {e}")
        pipe.send(("error",str(e)))  # Send shutdown signal 
    finally:
        env.close()

def start_environments():
    num_envs = 4  # Number of parallel environments
    env_processes = []
    parent_conns = []
    child_conns = []
    env_id = 'SuperMarioBros-1-1-v0'  # Environment ID

    for _ in range(num_envs):
        parent_conn, child_conn = mp.Pipe()
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
        process = mp.Process(target=environment_worker, args=(env_id, child_conn))
        process.start()
        env_processes.append(process)

    return env_processes, parent_conns

if __name__ == '__main__':
    
    LOGGING_PATH = 'logs'

    create_directory(LOGGING_PATH)

    logging.basicConfig(filename='logs/training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if torch.cuda.is_available():
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available")

    ENV_NAME = 'SuperMarioBros-1-1-v0'
    CKPT_SAVE_INTERVAL = 5_000
    NUM_OF_EPISODES = 100_000

    env = gym_super_mario_bros.make(ENV_NAME, render_mode='rgb_array', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = apply_wrappers(env)
    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

    models_dir = "models"

    create_directory(models_dir)

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

    env_processes, parent_conns = start_environments()

    for episode in range(NUM_OF_EPISODES):

        # Reset states for all environments
        total_rewards = [0] * len(parent_conns)
        all_done = [False] * len(parent_conns)
        states = [None] * len(parent_conns)

        for index, parent_conn in enumerate(parent_conns):
            #parent_conn.send(None)  # Sending a dummy action to start the environment
            states[index] = parent_conn.recv()

        while not all_done:
            actions = [agent.choose_action(state) if not done else None for state, done in zip(states, all_done)]
            for action, parent_conn in zip(actions, parent_conns):
                if action is not None:
                    parent_conn.send(action)

            for index, parent_conn in enumerate(parent_conns):
                if not all_done[index]:
                    next_state, reward, done, trunc, info = parent_conn.recv()
                    total_rewards[index] += reward
                    if done:
                        next_state, _ = parent_conn.recv()  # Reset and get new initial state
                        all_done[index] = True
                    experiences = (states[index], actions[index], reward, next_state, done)
                    agent.handle_experiences([experiences])
                    states[index] = next_state

        logging.info(f"Episode {episode}: Total Average Reward = {sum(total_rewards)/len(total_rewards)}, Epsilon = {agent.epsilon}, Learn Rate = {agent.scheduler.get_last_lr()[0]}")
        
        if (episode + 1) % CKPT_SAVE_INTERVAL == 0:
            agent.save_model(os.path.join(model_path, f"model_{episode + 1}_iter.pt"))

    for parent_conn in parent_conns:
        parent_conn.send(None)  # Send shutdown signal

    for process in env_processes:
        process.join()
