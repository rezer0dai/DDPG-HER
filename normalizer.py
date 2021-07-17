"""
Based on code from Marcin Andrychowicz

                    **** IMPORTANT ****

-> copy-pasted to my project from : https://github.com/vitchyr/rlkit
  >> encouraged to check that nice project
"""
import torch
import torch.nn as nn
import numpy as np

class Normalizer(torch.nn.Module):
    def __init__(
            self,
            size,
            eps=1e-8,
            default_clip_range=torch.tensor(np.inf),
            mean=0,
            std=1,
    ):
        super().__init__()
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range

        self.sum = torch.nn.Parameter(torch.zeros(self.size))
        self.sumsq = torch.nn.Parameter(torch.zeros(self.size))
        self.count = torch.nn.Parameter(torch.ones(1))
        self.mean = torch.nn.Parameter(mean + torch.zeros(self.size))
        self.std = torch.nn.Parameter(std * torch.ones(self.size))

        self.synchronized = True

    def update(self, v):
        if 1 == len(v):
            return
        self.sum.data = self.sum.data + v.sum(0)
        self.sumsq.data = self.sumsq.data + (v ** 2).sum(0)
        self.count[0] = self.count[0] + v.shape[0]
        self.synchronized = False

    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self._synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range

        # convert back to numpy ( torch is just for sharing data between workers )
        std = self.std.detach()
        mean = self.mean.detach()

        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def _synchronize(self):
        self.mean.data = self.sum.detach() / self.count[0]
        self.std.data = torch.sqrt(
            torch.max(
                torch.tensor(self.eps ** 2).to(self.std.device),
                self.sumsq.detach() / self.count.detach()[0] - (self.mean.detach() ** 2)
            )
        )
        self.synchronized = True

import config

import torch
import torch.nn as nn

class RunningNorm(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.add_module("norm", Normalizer(size_in))

        self.stop = False
        for p in self.norm.parameters():
            p.requires_grad = False

        self.register_parameter("dummy_param", nn.Parameter(torch.empty(0)))
    def device(self):
        return self.dummy_param.device

    def stop_norm(self):
        self.stop = True

    def active(self):
        return not self.stop

    def forward(self, states, update):
        shape = states.shape
        states = states.to(self.device()).view(-1, self.norm.size)
        if update and not self.stop:
            self.norm.update(states)
        states = states.to(self.device()).view(-1, self.norm.size)
        return self.norm.normalize(states).view(shape)

goal_norm = RunningNorm(config.GOAL_SIZE)
goal_norm.share_memory()

class GlobalNormalizerWithTime(nn.Module):
    def __init__(self, size_in, lowlevel=True):
        super().__init__()

        self.lowlevel = lowlevel
        
        enc = RunningNorm((size_in if config.LEAK2LL or not lowlevel else config.LL_STATE_SIZE) - config.TIMEFEAT)
        
        self.add_module("enc", enc)
        self.add_module("goal_encoder", goal_norm)

        self.register_parameter("dummy_param", nn.Parameter(torch.empty(0)))
    def device(self):
        return self.dummy_param.device

    def stop_norm(self):
        self.enc.stop_norm()

    def active(self):
        return self.enc.active()

    def update(self, states):
        self.forward(states, update=True)

    def forward(self, states, update=False):
        states = states.to(self.device()).float()

        if config.TIMEFEAT:
            tf = states[:, -config.TIMEFEAT:].view(-1, 1)
            states = states[:, :-config.TIMEFEAT]

        #print("\n my device", self.device(), self.goal_encoder.device(), goal_norm.device())
        enc = lambda data: self.goal_encoder(data, update).view(len(data), -1)
        pos = lambda buf, b, e: buf[:, b*config.GOAL_SIZE:e*config.GOAL_SIZE]

        # goal, {current, previous} arm pos 
        arm_pos = pos(states, 1, 3)
        if config.LEAK2LL or not self.lowlevel:# achieved goal and actual goal leaking hint what is our goal to low level
            arm_pos = enc(
                    torch.cat([pos(states, 0, 1), arm_pos, pos(states, -1, 10000)], 1)
                    )[:, :-config.GOAL_SIZE] # skip goal from here, just add hint via norm, also achieved one will be not used by NN
        else:
            arm_pos = torch.cat([pos(arm_pos, 0, 1),
                enc( arm_pos )
                ], 1)# first arm_pos aka fake achieved will be not used anyway

        obj_pos_w_goal = pos(states, -(2*config.PUSHER+1), 10000)
        if config.PUSHER and not self.lowlevel: # object pos, prev pos, goal --> only high level
            obj_pos_w_goal = enc(obj_pos_w_goal)

        state = states
        if self.lowlevel and not config.LEAK2LL:
            state = states[:, config.GOAL_SIZE:][:, :config.LL_STATE_SIZE-config.TIMEFEAT]

        encoded = self.enc(state, update)

        if self.lowlevel and not config.LEAK2LL:
            encoded = torch.cat([states[:, :config.GOAL_SIZE], encoded, states[:, config.GOAL_SIZE+config.LL_STATE_SIZE-config.TIMEFEAT:]], 1)

        encoded = pos(encoded, 3, -(2*config.PUSHER+1)) # skip object + goal positions, those will be added by goal_encoder

        # note : achievedel goal, and goal will be skipped from NN ( this norm is used in encoders just for puting it trough NN )
        #print("\n sumarize", arm_pos.device, encoded.device, obj_pos_w_goal.device)
        encoded = torch.cat([arm_pos, encoded, obj_pos_w_goal], 1)

        if states.shape[-1] != encoded.shape[-1]:
            print("\nshappezzz:", states.shape[-1], encoded.shape[-1], self.lowlevel, obj_pos_w_goal.shape)
        assert states.shape[-1] == encoded.shape[-1]
        if config.TIMEFEAT:
            encoded = torch.cat([encoded, tf], 1)
        return encoded
