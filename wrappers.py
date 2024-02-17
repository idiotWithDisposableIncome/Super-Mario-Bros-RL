import numpy as np
from gym import Wrapper
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack


class SkipFrame(Wrapper):
    def __init__(self, env, skip, index=None):
        super().__init__(env)
        self._skip = skip
        self.index = index
        self.counter = 0
        self.prev_info = None
        self.furthest_x = 0
    
    def step(self, action):
        total_reward = 0.0
        #done = False

        for _ in range(self._skip):
            try:
                next_state, reward, done, trunc, info = self.env.step(action)
                #print(f"skipwrapper for worker: {self.index}  done state after step: {done} action: {action}")
                if self.prev_info is not None:
                    total_reward += self.calculate_reward(info, done)
                self.prev_info = info
                if done:
                    break
            except Exception as e:
                print(f"Error in SkipFrame for worker {self.index}: {e}. sending Reset Signal to main loop.")
                next_state, _ = self.env.reset()  # Reset the environment
                done = True  # Set done to True as the episode has ended
                trunc = True
                info = {}
                total_reward -= 100
                break 
        return next_state, total_reward, done, trunc, info
    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        self.prev_info = None
        self.furthest_x = 0
        return state, info

    def calculate_reward(self, info, done):
        reward = 0
        # Progression
        reward += max(-25, min( ( (info['x_pos'] - self.furthest_x ) * 3 ) , 25) )
        #set new furthest
        if self.furthest_x < info['x_pos']:
            self.furthest_x = info['x_pos']

        # Coins
        reward += max(-5,min( (info['coins'] - self.prev_info['coins']) * 1, 5) )

        # Flag Get (end of level)
        if info['flag_get']:
            reward += 100

        # Lives
        reward -= (self.prev_info['life'] - info['life']) * 100

        # Score
        reward += max(-25, min( int((info['score'] - self.prev_info['score'] ) * 0.01), 25))

        # Time Penalty
        reward -= (self.prev_info['time'] - info['time']) * 1

        # Check for power-up status change (small, tall, fire)
        if info['status'] != self.prev_info['status']:
            if info['status'] != 'small':
                reward += 30
            else:
                reward -= 35

        # Vertical Movement (if needed)
        # reward += (info['y_pos'] - self.prev_y_pos) * VERTICAL_MOVEMENT_REWARD

        # Adjust reward for death
        if done and info['life'] < self.prev_info['life']:
            reward -= 100


        # Clip the reward to a reasonable range to prevent any single event from having too much influence
        reward = max(-100, min(reward, 100))

        if self.prev_info['life'] > info['life']:
            self.furthest_x = 0
            self.prev_info['x_pos'] = 0
        if info['world'] > self.prev_info['world']:
            self.furthest_x = 0
            self.prev_info['x_pos'] = 0
        return reward

def apply_wrappers(env, index=None):
    env = SkipFrame(env, skip=4, index=index) # Num of frames to apply one action to
    env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True) # May need to change lz4_compress to False if issues arise
    return env
