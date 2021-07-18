import numpy as np
from copy import deepcopy as dc
import random

import config
import normalizer
state_norm = normalizer.GlobalNormalizerWithTime(config.STATE_SIZE)
state_norm.share_memory()


class Memory(object):
    def __init__(self, max_size, _k_future, _env, state_dim=config.STATE_SIZE, action_dim=config.ACTION_SIZE):
        max_size = int(1e7)
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.size

    def add(self, episode_dict):
        norm_ind = self.ptr
        action = episode_dict["action"]
        reward = np.asarray(episode_dict["reward"])
        state = np.concatenate([ dc( episode_dict["state"] ), dc( episode_dict["desired_goal"] ) ], 1)
        next_state = np.concatenate([ dc( episode_dict["next_state"] ), dc( episode_dict["desired_goal"][1:] ) ], 1)

        for _ in range(200):
            for j in range(len(action) - 2):
                state_, next_state_ = dc ( state[j] ), dc( next_state[j] )
                g = len(episode_dict["desired_goal"][0])
                assert config.GOAL_SIZE == g
                ag = dc (random.choice(episode_dict["next_achieved_goal"][j:]) )
                state_[-g:] = ag
                next_state_[-g:] = ag
                reward_ = -1. * (np.linalg.norm(episode_dict["next_achieved_goal"][j][:g] - state_[-g:]) > .05)

                self._add(state_, action[j], next_state_, reward_)

            if random.random() < config.HER_RATIO:
                continue
            for i in range(len(action)):
                self._add(state[i], action[i], next_state[i], reward[i])

        if self.ptr > norm_ind:
            state_norm.update(torch.from_numpy(self.state[norm_ind:self.ptr]))

    def _add(self, state, action, next_state, reward):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            state_norm(torch.from_numpy(self.state[ind])),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            state_norm(torch.from_numpy(self.next_state[ind])),
            )

    def normalize_state(self, state, goal):
        if False:#not config.NORMALIZE:
            return states
        return state_norm( torch.cat([torch.from_numpy(state), torch.from_numpy(goal)]).view(1, -1) )

    @staticmethod
    def clip_obs(x):
        return np.clip(x, -200, 200)
