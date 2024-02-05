import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


class SkipFrame(Wrapper):
    def __init__(self, env, skip, index=None):
        super().__init__(env)
        self._skip = skip
        self.index = index
        self.counter = 0
    
    def step(self, action):
        total_reward = 0.0
        #done = False

        for _ in range(self._skip):
            try:
                next_state, reward, done, trunc, info = self.env.step(action)
                #print(f"skipwrapper for worker: {self.index}  done state after step: {done} action: {action}")
                total_reward += reward
                if done:
                    break
            except Exception as e:
                print(f"Error in SkipFrame for worker {self.index}: {e}. sending Reset Signal to main loop.")
                next_state, _ = self.env.reset()  # Reset the environment
                done = True  # Set done to True as the episode has ended
                trunc = True
                info = {}
                break 
        return next_state, total_reward, done, trunc, info
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        return state, info

def apply_wrappers(env, index=None):
    env = SkipFrame(env, skip=4, index=index) # Num of frames to apply one action to
    env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True) # May need to change lz4_compress to False if issues arise
    return env
