import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from agent import Agent
from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers
import os
import re
import numpy as np
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

def environment_worker(env_id, pipe, index):
    try:
        env = gym_super_mario_bros.make(env_id, render_mode='rgb_array', apply_api_compatibility=True)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = apply_wrappers(env, index=index)
        state, _ = env.reset()
        pipe.send(state)  # Send initial state to the main process

        while True:
            action = pipe.recv()  # Receive action from the main process
            if action == None:
                break  # Shutdown signal received
            #print(f"worker {index} received Action: {action} ")
            next_state, reward, done, trunc, info = env.step(action)
            #print(f"worker done state after step: {done} environmentId: {index}")
            if done:
                if not trunc:
                    next_state, _ = env.reset()
                    #print(f"Worker {index} resetting environment.")
                else:
                    print(f"Worker {index} forced reset")
                
                pipe.send((state, action, reward, next_state, done))  # Send experience back to the main process
                #print(f"worker done state after reset: {done} environmentId: {index}")
                state = next_state  # Update state to the new initial state
            else:
                pipe.send((state, action, reward, next_state, done))  # Send experience back to the main process
                state = next_state
    except Exception as e:
        print(f"Worker {index} encountered an error: {e}")
        pipe.send(("error",str(e)))  # Send shutdown signal 
    finally:
        env.close()

def start_environments():
    num_envs = 4  # Number of parallel environments
    env_processes = []
    parent_conns = []
    child_conns = []
    env_id = 'SuperMarioBros-1-1-v0'  # Environment ID

    for i in range(num_envs):
        parent_conn, child_conn = mp.Pipe()
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)
        process = mp.Process(target=environment_worker, args=(env_id, child_conn, i))
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
    CKPT_SAVE_INTERVAL = 20_000
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
    total_episodes_played = [0] * len(parent_conns)
    total_rewards_current_episode = [0] * len(parent_conns)
    total_reward = 0
    average_total_reward = 0
    env_average_total_reward = [0] * len(parent_conns)
    env_total_reward = [0] * len(parent_conns)
    # Initial receipt of state from each environment
    env_dif = [0] * len(parent_conns)
    states = [parent_conn.recv() for parent_conn in parent_conns]

    while sum(total_episodes_played) < NUM_OF_EPISODES:
        # Collect experiences from each environment
        for index, parent_conn in enumerate(parent_conns):

            action = agent.choose_action(states[index])
            if env_dif[index] == 0:
                env_dif[index] += 1
                #print(f"Main loop sending action in top loop for worker {index} action: {action}")
                parent_conn.send(action)  # Send action to the environment worker

            if parent_conn.poll():  # Check if there's a message from the worker
                env_dif[index] += -1
                message = parent_conn.recv()
                
                if isinstance(message, tuple) and isinstance(message[0], str):
                    if message[0] == "error":
                        # Handle error
                        print(f"Error received from worker {index}: {message[1]}")

                state, action, reward, next_state, done = message
                #print(f"Main loop recv exp from worker {index} done: {done} action: {action}")
                agent.handle_experiences([(state, action, reward, next_state, done)])
                total_rewards_current_episode[index] += reward
                states[index] = next_state
                if done:
                    env_dif[index] = 0
                    # total episodes played for this environment
                    total_episodes_played[index] += 1
                    #overall total reward
                    total_reward += total_rewards_current_episode[index]
                    #set env total reward
                    env_total_reward[index] += total_rewards_current_episode[index]
                    #overall average reward
                    average_total_reward = total_reward / sum(total_episodes_played)
                    #average reward for this environment
                    env_average_total_reward[index] = env_total_reward[index] / total_episodes_played[index]
                    #log the rewards and save the model if it is time
                    if sum(total_episodes_played) % CKPT_SAVE_INTERVAL == 0:
                        agent.save_model(os.path.join(model_path, f"model_processor_{index}_episode_{total_episodes_played[index]}.pt"))
                    logging.info(f"Processor {index}, Episode {total_episodes_played[index]}:  Env Average Reward = {env_average_total_reward[index]}, Total Episode Reward = {total_rewards_current_episode[index]}, Model Average Total Reward = {average_total_reward}")
                    print(f"Processor {index}, Episode {total_episodes_played[index]}:  Env Average Reward = {env_average_total_reward[index]}, Total Episode Reward = {total_rewards_current_episode[index]}, Model Average Total Reward = {average_total_reward}")
                    #reset reward for this environment
                    total_rewards_current_episode[index] = 0

    for parent_conn in parent_conns:
        parent_conn.send(None)  # Send shutdown signal

    for process in env_processes:
        process.join()
