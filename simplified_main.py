import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import apply_wrappers
from agent import Agent

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = False
DISPLAY = True
NUM_OF_EPISODES = 100


env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

agent.load_model("models/model_processor_0_episode_25052.pt")
#agent.epsilon = 0.2
#agent.eps_min = 0.0
#agent.eps_decay = 0.0

for i in range(NUM_OF_EPISODES):
    done = False
    state, _ = env.reset()
    while not done:
        a = agent.choose_action(state)
        new_state, reward, done, truncated, info  = env.step(a)
        
        agent.store_in_memory(state, a, reward, new_state, done)
        agent.learn()

        state = new_state

env.close()
